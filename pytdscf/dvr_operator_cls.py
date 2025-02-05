"""Discrete Variable Representation Operator Class"""

from __future__ import annotations

import _pickle  # type: ignore
import math
import os
import pickle
from collections import Counter
from itertools import combinations, product
from typing import Callable

import numpy as np
import polars as pl
import scipy.linalg
from discvar.abc import DVRPrimitivesMixin
from loguru import logger

from pytdscf import units
from pytdscf._helper import from_dbkey, to_dbkey
from pytdscf._mpo_cls import to_mpo

try:
    from ase.db import connect
    from ase.units import Hartree
except ImportError:
    print("Failed to import ase. You cannot use database.")


def deepcopy(item):
    """copy.deepcopy() is too lazy"""
    return _pickle.loads(_pickle.dumps(item, -1))


logger = logger.bind(name="main")

debye_in_ase = units.au_in_angstrom / units.au_in_debye


class TensorOperator:
    r""" Tensor Operator class

    Attributes:
        shape (Tuple[int]) : Tensor shape
        tensor_orig (np.ndarray) : Original tensor
        tensor_decomposed (np.ndarray) : Decomposed tensor
        tensor_full (np.ndarray) : if ``only_diag==False``, \
            ``tensor_orig`` is zero-filled.
        only_diag (bool) : Only diagonal terms are non-zero. \
            Defaults to ``False``.
        legs (Tuple[int]) : Tensor legs
        name (str) : Tensor Operator name. Defaults to ``None``.

    Tensor Example:

    .. math::
       \begin{array}{c}
       j_0^\prime &   & j_1^\prime &   &   &   & j_3^\prime \\
       |          &   & |          &   &   &   & | \\
       W(0)       & - & W(1)       & - & - & - & W(3) \\
       |          &   & |          &   &   &   & | \\
       j_0        &   & j_1        &   &   &   & j_3 \\
       \end{array}\\

    .. math::
       j_0 = 0, 1, .. n-1,
       j_1 = 0, 1, .. m-1,
       j_3 = 0, 1, .. l-1

    ``tensor = np.array([W0, W1, W3])``.

    When ``only_diag=False``, above example is ``shape = (n, n, m, m, l, l)``,\
     ``legs=(0, 0, 1, 1, 3, 3)``,  (even index is bra, odd index is ket).

    Otherwise, ``only_diag=True``, ``shape=(n,m,l)``, ``legs=(0, 1, 3)``.

    ``W_1[beta_01,j_1^prime,j_1,beta_13] =``
    :math:`W{\small \beta_{0,1}}_{j_1}^{j_1^\prime}{\small \beta_{1,3}}`

    """

    legs: tuple[int, ...]
    tensor_orig: np.ndarray
    shape: tuple[int, ...]
    only_diag: bool
    tensor_decomposed: list[np.ndarray]
    tensor_full: np.ndarray
    name: str
    bond_dimension: list[int]
    full_bond_dimension: list[int]

    def __init__(
        self,
        *,
        shape: tuple[int, ...] | None = None,
        tensor: np.ndarray | None = None,
        only_diag: bool = False,
        legs: tuple[int, ...] | None = None,
        name: str | None = None,
        mpo: list[np.ndarray] | None = None,
    ) -> None:
        if name is not None:
            self.name = name
        else:
            if mpo is not None:
                assert isinstance(mpo, list)
                only_diag = all(len(core.shape) == 3 for core in mpo)
                bond_dimension = [1]
                for core in mpo:
                    bond_dimension.append(core.shape[-1])
                bond_dimension.append(1)
                self.bond_dimension = bond_dimension
                shape = tuple([i for _core in mpo for i in _core.shape[1:-1]])
                self.tensor_decomposed = mpo

            if shape is None and tensor is None:
                raise ValueError(
                    "You must give argument either shape or tensor"
                )
            if tensor is None:
                assert isinstance(shape, tuple)
                self.shape = shape
                if mpo is None:
                    self.tensor_orig = np.zeros(shape)
            else:
                self.shape = tensor.shape
                self.tensor_orig = tensor
            self.only_diag = only_diag
            if legs is None:
                if isinstance(mpo, list):
                    _legs = []
                    for i, core in enumerate(mpo):
                        if core.ndim == 3:
                            _legs.append(i)
                        elif core.ndim == 4:
                            _legs.extend([i, i])
                        else:
                            raise ValueError(
                                f"core.ndim must be 3 or 4, but {core.ndim}"
                            )
                    self.legs = tuple(_legs)
                else:
                    if only_diag:
                        self.legs = tuple([i for i in range(len(self.shape))])
                    else:
                        raise ValueError(
                            "leg is ambiguous. Please give leg argument."
                        )
            else:
                self.legs = legs
                assert (
                    len(self.legs) == len(self.shape)
                ), f"Tensor shape {self.shape} and legs {self.legs} are different"

    @property
    def dtype(self) -> np.dtype:
        try:
            return self.tensor_orig.dtype
        except AttributeError:
            return self.tensor_decomposed[0].dtype

    def __str__(self) -> str:
        dum = ""
        if hasattr(self, "tensor_orig"):
            dum += self.tensor_orig.__str__()
        if hasattr(self, "tensor_decomposed"):
            dum += "\n" + self.tensor_decomposed.__str__()
        return dum

    def __repr__(self) -> str:
        return self.__str__()

    def __add__(self, tensor_op) -> TensorOperator:
        """merge tensor operatos"""
        tensor_op1 = self.add_dammy_legs(
            add_legs=tensor_op.legs, add_shape=tensor_op.shape
        )
        tensor_op2 = tensor_op.add_dammy_legs(
            add_legs=self.legs, add_shape=self.shape
        )
        if tensor_op1.only_diag ^ tensor_op2.only_diag:
            raise NotImplementedError
        else:
            if (
                tensor_op1.shape == tensor_op2.shape
                and tensor_op1.legs == tensor_op2.legs
            ):
                return TensorOperator(
                    tensor=tensor_op1.tensor_orig + tensor_op2.tensor_orig,
                    only_diag=tensor_op1.only_diag,
                    legs=tensor_op1.legs,
                )
            else:
                raise ValueError(
                    "Tensor shapes or legs are different."
                    + f" shapes : {tensor_op1.shape} vs {tensor_op2.shape}"
                    + f" legs : {tensor_op1.legs} vs {tensor_op2.legs}"
                )

    def add_dammy_legs(
        self, add_legs: tuple[int, ...], add_shape: tuple[int, ...]
    ) -> TensorOperator:
        r"""Dammy legs addition

        Args:
            add_legs(Tuple[int]) : additional tensor legs. \
                e.g. add :math:`j_1 ` is ``add_legs=(1,)``.
            add_shape(Tuple[int]) : additional tensor shape. \
                e.g. add :math:`j_1 = 0,1,\ldots,3` is ``add_shape=(4,)``.

        Returns:
            TensorOperator : Filled Tensor

        e.g.
        ``from legs = (0,1)`` to ``legs=(0,1,2)``, where dammy legs is 2.

        """
        if len(add_legs) != len(add_shape):
            raise ValueError("additional legs and shape is not the same")
        _after_legs = list(set(self.legs + add_legs))
        _after_legs.sort()
        after_legs = tuple(_after_legs)
        if not self.only_diag:
            after_legs = self._repeat_leg(after_legs)

        if tuple(after_legs) == self.legs:
            return deepcopy(self)

        add_index: list[int] = []
        add_index_not_orig: list[int] = []
        orig_index: list[int] = []
        for ind, leg in enumerate(after_legs):
            if leg in self.legs:
                orig_index.append(ind)
            else:
                add_index_not_orig.append(ind)
            if leg in add_legs:
                add_index.append(ind)
        after_shape: list[int] = [None for _ in range(len(after_legs))]  # type: ignore
        for shape, ind in zip(add_shape, add_index, strict=True):
            after_shape[ind] = shape
        for shape, ind in zip(self.shape, orig_index, strict=True):
            after_shape[ind] = shape
        to = TensorOperator(
            shape=tuple(after_shape), legs=after_legs, only_diag=self.only_diag
        )

        add_shape_diag = add_shape
        if len(add_legs) % 2 == 0:
            if add_legs[::2] == add_legs[1::2]:
                add_shape_diag = add_shape[0::2]

        _iter = [range(add_shape_diag[p]) for p in add_index_not_orig]
        for add_leg in product(*_iter):
            after_leg = [slice(s) for s in after_shape]
            if self.only_diag:
                for leg, ind in zip(add_leg, add_index_not_orig, strict=True):
                    after_leg[ind] = leg
            else:
                for leg, ind in zip(add_leg, add_index_not_orig, strict=True):
                    """maybe wrong (not debugging)"""
                    after_leg[ind] = leg
                    after_leg[ind + 1] = leg
            to.tensor_orig[tuple(after_leg)] = deepcopy(self.tensor_orig)
        return to

    def get_tensor_full(self) -> np.ndarray:
        """fill off-diagonal term zero if only_diag=True"""
        if not hasattr(self, "tensor_full"):
            if self.only_diag:
                dum = np.zeros(self._repeat_leg(self.shape))
                for leg in product(*[range(n) for n in self.shape]):
                    dum[self._repeat_leg(leg_diag=leg)] += self.tensor_orig[leg]
                self.tensor_full = dum
            else:
                self.tensor_full = deepcopy(self.tensor_orig)
        return self.tensor_full

    def _repeat_leg(self, leg_diag: tuple[int, ...]) -> tuple[int, ...]:
        dum = [(l1, l2) for l1, l2 in zip(leg_diag, leg_diag, strict=True)]
        leg2: tuple[int, ...] = ()
        for d in dum:
            leg2 += d
        return leg2

    def _trans_J_to_1d(self, J: tuple[int, ...]) -> int:
        """
        Args:
            Tensor-index J (tuple) : (j_1, j_2, ..., j_f)
        Returns:
            int : 1d array index
        """
        dum = 0
        digits = np.flip(np.append(1, np.cumprod(self.shape[:0:-1])))
        for j, d in zip(J, digits, strict=True):
            dum += j * d
        return int(dum)

    def _iter_J(self, N: tuple[int, ...] | None = None):
        """
        Iterate all patterns of tensor index J = (j_1, j_2, ..., j_f)
        """
        if N is None:
            N = self.shape
        for J in product(*[[k for k in range(n)] for n in N]):
            yield J

    def _reshape_row_to_col(self, matrix: np.ndarray, index: int):
        """
        Reshape
        A(n^k, n^l) -> A(n^{k-1}, n^{l+1})

        Args:
            matrix (np.ndarray) : Original matrix
            index (int) : mode which move row to column (0-index)
        Returns:
            np.ndarray : Transformed matrix
        """
        row = int(self.shape[index] * matrix.shape[0])
        col = int(np.prod(self.shape[index + 1 :]))

        new_matrix = np.zeros((row, col))
        for j in range(self.shape[index]):
            new_matrix[matrix.shape[0] * j : matrix.shape[0] * (j + 1), :] = (
                matrix[:, col * j : col * (j + 1)]
            )

        return new_matrix

    def _QRD(self) -> list[np.ndarray]:
        """QR decomposition-based MPO"""
        self.tensor_decomposed = []
        if self.only_diag:
            r = np.zeros((1, np.prod(self.shape)), dtype=np.float64)
            for J in self._iter_J():
                r[0, self._trans_J_to_1d(J)] = self.tensor_orig[J]
            for i in range(len(self.shape)):
                r = self._reshape_row_to_col(
                    r[: self.full_bond_dimension[i], :], i
                )
                q, r = scipy.linalg.qr(r)
                self.tensor_decomposed.append(
                    np.array(
                        [
                            q[
                                tau :: self.full_bond_dimension[i],
                                : self.full_bond_dimension[i + 1],
                            ]
                            for tau in range(self.full_bond_dimension[i])
                        ],
                        dtype=np.float64,
                    )
                )
            self.tensor_decomposed[-1] *= r[0, 0]
        else:
            raise NotImplementedError("non-diag case")
        return self.tensor_decomposed

    def _SVD(self):
        """Singular Value Decomposition-based MPO"""
        self.tensor_decomposed = []
        if self.only_diag:
            r = np.zeros((1, np.prod(self.shape)))
            for J in self._iter_J():
                r[0, self._trans_J_to_1d(J)] = self.tensor_orig[J]
            for i in range(len(self.shape)):
                r = self._reshape_row_to_col(r[: self.bond_dimension[i], :], i)
                if (
                    1 < self.bond_dimension[i + 1] < min(r.shape)
                    and min(r.shape) > 10000
                    and not hasattr(self, "rate")
                ):
                    logger.warning("matrix is too large to execute full-SVD")
                    U, s, Vh = scipy.sparse.linalg.svds(
                        r, k=self.bond_dimension[i + 1]
                    )
                else:
                    try:
                        U, s, Vh = scipy.linalg.svd(r, full_matrices=False)
                    except np.linalg.LinAlgError:
                        U, s, Vh = scipy.linalg.svd(
                            r, full_matrices=False, lapack_driver="gesvd"
                        )

                    if hasattr(self, "rate"):
                        if self.square_sum:
                            total_val = np.tensordot(
                                r, r.T, axes=[[1, 0], [0, 1]]
                            )
                        else:
                            total_val = np.sum(s)
                        cum_s_val = 0.0
                        rank = 0
                        while (
                            cum_s_val / total_val < self.rate
                            and rank < self.bond_dimension[i + 1]
                        ):
                            if self.square_sum:
                                cum_s_val += s[rank] ** 2
                            else:
                                cum_s_val += s[rank]
                            rank += 1
                        self.bond_dimension[i + 1] = rank

                self.tensor_decomposed.append(
                    np.array(
                        [
                            U[
                                tau :: self.bond_dimension[i],
                                : self.bond_dimension[i + 1],
                            ]
                            for tau in range(self.bond_dimension[i])
                        ]
                    )
                )
                dim = min(self.bond_dimension[i + 1], Vh.shape[0])
                r = np.diag(s[:dim]) @ Vh[:dim, :]
            self.tensor_decomposed[-1] *= r[0, 0]
        else:
            raise NotImplementedError("non-diag case")
        return self.tensor_decomposed

    def decompose(
        self,
        bond_dimension: list[int] | int | None = None,
        decompose_type: str = "SVD",
        rate: float | None = None,
        square_sum: bool = True,
        overwrite: bool = False,
    ) -> list[np.ndarray]:
        r"""MPO Decomposition

        Args:
            bond_dimension (List[int] or int) : MPO bond dimension
            decompose_type (str) : Tensor Train Decomposition \
                Type ``'QRD'`` or ``'SVD'``. Defaults to ``'SVD'``.
            rate (float) : SVD decomposition contribution rate truncation. \
                Defaults to ``None``.
            square_sum (bool) : Whether ``rate`` is based on the sum of \
                squares of the eigenvalues, or simply sum of the eigenvalues. \
                Square sum can calculate easily by trace of matrix, \
                but need high ``rate`` percentage. We recommend
                ``rate=0.99999999999``, if ``square_sum=True``.

        Returns:
            List[np.ndarray] : MPO

        """
        if not overwrite and hasattr(self, "tensor_decomposed"):
            return self.tensor_decomposed
        if (not hasattr(self, "bond_dimension")) or bond_dimension is not None:
            if decompose_type.lower() in ["qrd", "qr"]:
                bond_dimension = None
            self.get_bond_dimension(bond_dimension=bond_dimension)
        if rate is not None:
            if 0.0 < rate < 1.0:
                self.rate = rate
                self.square_sum = square_sum
            else:
                raise ValueError(
                    f"Contribution rate must be in (0.0, 1.0), but {rate}"
                )
        if len(set(self.legs)) >= 2:
            if decompose_type.lower() in ["qrd", "qr"]:
                return self._QRD()
            elif decompose_type.lower() in ["svd", "sv"]:
                return self._SVD()
            else:
                raise ValueError('decompose_type must be "QRD" or "SVD"')
        else:
            if self.only_diag:
                dum = np.zeros(
                    (1, self.tensor_orig.shape[0], 1),
                    dtype=self.tensor_orig.dtype,
                )
                dum[0, :, 0] = deepcopy(self.tensor_orig)
            else:
                dum = np.zeros(
                    (1, *self.tensor_orig.shape[0:2], 1),
                    dtype=self.tensor_orig.dtype,
                )
                dum[0, :, :, 0] = deepcopy(self.tensor_orig)

            self.tensor_decomposed = [dum]
            return self.tensor_decomposed

    def get_bond_dimension(self, bond_dimension: list[int] | int | None = None):
        """get MPO bond-dimension

        Args:
            bond_dimension(List[int] | int) : MPO bond dimension

        """
        from_left = list(np.cumprod([1] + list(self.shape)))
        if not self.only_diag:
            from_left = from_left[0::2]
        from_right = list(reversed(from_left))
        self.full_bond_dimension = [
            min(m1, m2) for m1, m2 in zip(from_left, from_right, strict=True)
        ]
        if bond_dimension is None:
            self.bond_dimension = deepcopy(self.full_bond_dimension)
        elif isinstance(bond_dimension, int):
            self.bond_dimension = [
                min(m, bond_dimension) for m in self.full_bond_dimension
            ]
        elif isinstance(bond_dimension, list):
            if len(bond_dimension) != len(from_left):
                raise ValueError(
                    "bond_dimension length is wrong. \
                    Note that terminal bond dimension are fixed to 1."
                )
            self.bond_dimension = [
                min(m1, m2)
                for m1, m2 in zip(
                    self.full_bond_dimension, bond_dimension, strict=True
                )
            ]
        else:
            raise ValueError(
                f"bond dimension type is wrong {type(bond_dimension)}"
            )
        return self.bond_dimension

    def get_num_of_elements(self) -> tuple[int, int | None]:
        """Get number of decomposed tensor operator element

        Returns:
            Tuple[int, int] : Number of tensor elements \
                (before, after) decomposition

        """
        before = self.tensor_orig.size
        if hasattr(self, "bond_dimension"):
            after = sum(
                [
                    self.shape[i]
                    * self.bond_dimension[i]
                    * self.bond_dimension[i + 1]
                    for i in range(len(self.shape))
                ]
            )
        else:
            after = None
        return (before, after)

    def restore_from_decoposed(self) -> np.ndarray:
        """Restore ``tensor_orig`` from MPO (`tensor_decomposed`).
        This routine is almost the same as ``mps_cls.buid_CICoef``
        """
        dum = np.array(self.tensor_decomposed[0])
        for isite in range(1, len(self.tensor_decomposed)):
            dum = np.tensordot(dum, self.tensor_decomposed[isite], axes=[-1, 0])
        return dum[0, ..., 0]

    def get_frobeinus_norm(self) -> float:
        """Get Frobenius norm of tensor operator"""
        return float(np.linalg.norm(self.tensor_orig))

    def estimate_error(self, tensor: np.ndarray | None = None):
        """
        Estimate Frobenius norm between self and given-tensor \
            or renormalized tensor

        Args:
            tensor (np.ndarray) : given tensor (option)

        Returns:
            float : Error %

        """
        if tensor is None:
            tensor = self.restore_from_decoposed()
        if tensor.shape == self.shape:
            dif = self.tensor_orig - tensor
            error_percent = (
                1.0 - np.linalg.norm(dif) / np.linalg.norm(self.tensor_orig)
            ) * 100
            return error_percent
        else:
            raise ValueError(
                f"Tensor Shape if different {tensor.shape} VS {self.shape}"
            )

    def _default_name(self) -> str:
        name = "legs"
        for leg in self.legs:
            name += f"_{leg}"
        return name

    def save(self, name: str | None = None) -> None:
        """
        Save Tensor Operator Object as binary
        file in ./tensor_operators directory

        Args:
            name(str) : file name not including extension

        """
        if name is None:
            name = self._default_name()
        self.name = name
        if not os.path.exists("./tensor_operators"):
            os.makedirs("./tensor_operators")
        with open("./tensor_operators/" + name + ".pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, name) -> TensorOperator:
        """
        Load Tensor Operator Object from binary
        file in ./tensor_operators directory
        """
        with open("./tensor_operators/" + name + ".pkl", "rb") as f:
            TO = pickle.load(f)
        return cls(
            shape=TO.shape,
            tensor=TO.tensor_orig,
            only_diag=TO.only_diag,
            legs=TO.legs,
        )
        # self.__init__(
        #     shape=TO.shape,
        #     tensor=TO.tensor_orig,
        #     only_diag=TO.only_diag,
        #     legs=TO.legs,
        # )


class PotentialFunction:
    """Analytical Model Potential from polynomial-based PES

    Args:
        DOFs (List[int]) : Degree of freedoms used for potential function. \
            1-index.
        polynomial_coef (Dict[Tuple[int], float]) : Polynomial PES coefficient.
        cut_off (float) : potential coefficient cut-off. defaults to no-cutoff.
        div_factorial(bool) : Whether or not divided by factorial term. \
            Defaults to ``True``.
        dipole (Optional[bool]) : Use dipole moment surface (3D vector).
        efiled (Optional[tuple[float, float, float]]) : Electronic field. (only dipole)

    """

    def __init__(
        self,
        DOFs: list[int],
        polynomial_coef: dict[tuple[int, ...], float],
        cut_off: float = -1.0,
        div_factorial: bool = True,
        dipole: bool = False,
        efield: tuple[float, float, float] = (1.0e-02, 1.0e-02, 1.0e-02),
    ) -> None:
        if polynomial_coef is None:
            raise NotImplementedError
        else:
            self.DOFs_0index = [p - 1 for p in DOFs]  # 0-index
            self.DOFs_1index = DOFs  # 1-index
            self.mode_index = {}
            for i, p in enumerate(self.DOFs_0index):
                self.mode_index[p] = i
            self.cut_off = cut_off
            self.div_factorial = div_factorial
            self.dipole = dipole
            self.efield = efield
            self.polynomial = dict()
            for key, coef in polynomial_coef.items():
                if set(key) & set(self.DOFs_1index) == set(key):
                    c_key = Counter(key)
                    if dipole:
                        coef = np.inner(coef, efield)
                    if self.div_factorial:
                        for order in c_key.values():
                            coef /= math.factorial(order)
                    if abs(coef) > self.cut_off:
                        self.polynomial[key] = coef

    def __call__(self, *args) -> float:
        dum = 0.0
        if () in self.polynomial:
            dum += self.polynomial[()]
        for key, coef in self.polynomial.items():
            term = deepcopy(coef)
            c_key = Counter(key)
            for idof, order in c_key.items():
                term *= pow(args[self.mode_index[idof - 1]], order)
            dum += term
        return dum


def construct_nMR_recursive(
    dvr_prims: list[DVRPrimitivesMixin],
    nMR: int = 3,
    ndof: int | None = None,
    func: dict[tuple[int, ...], Callable] | None = None,
    db: str | None = None,
    df: pl.DataFrame | None = None,
    active_dofs: list[int] | None = None,
    site_order: dict[int, int] | None = None,
    zero_indices: list[int] | None = None,
    return_tensor: bool = False,
    include_const_in_mpo: bool = False,
    ref_ene: float | None = None,
    dipole: bool = False,
    efield: tuple[float, float, float] = (1.0, 1.0, 1.0),
    rate: float = 1.0,
    k: int = 200,
    nsweep: int = 4,
) -> list[np.ndarray] | dict[tuple[int, ...], TensorOperator]:
    r"""Construct n-Mode Representation Operator

    n-Mode Representation (nMR) are used for reducing grid points
    for ab initio calculation.
    To avoid duplicate addition of the same coordinates,
    use principle of inclusion and exclusion.
    (It is easy to understand by Venn-diagram)

    Args:
        dvr_prims (List[DVRPrimitivesMixin]) : DVR functions
        nMR (Optional, int) : Mode Representation Number. Defaults to ``3``.
        ndof (Optional, int) : number of mode including in database.
        func (Optional, Dict[Tuple[int], Callable]) : E.g. Potential energy function
        db (Optional, str) : Electronic structure calculation database path, \
            such as 'foo/hoge.db'
        df (Optional, pl.DataFrame) : Electronic structure calculation dataframe.
        active_dofs (Optional, List[int]) : Active modes Defaults to ALL.
        site_order (Optional, Dict[int, int]) : MPO site order. \
            Defaults to DOF index order.
        zero_indices (Optional, List[int]) : nMR criteria coordinate grid.
        return_tensor (bool) : Return before decomposition. Defaults to False.
        include_const_in_mpo (bool) : Include scalar constant value (such as refernce energy) in MPO. Defaults to False.
        ref_ene (float) : Opt coord energy in a.u., if you need \
            subtract from function or database energy. Defaults to `0.0`
        dipole (bool) : Get dipole moment from database. \
            Defaults to `False`
        efield (List[float]) : \
            Electronic field for inner products with dipole moment.\
            Defaults to [1.0e-02 , 1.0e-02, 1.0e-02]
        rate (float) : SVD contribution rate. Defaults to 1.0
        k (int) : SVD bond dimension. Defaults to 200

    Returns:
        List[np.ndarray] : nMR MPO

    Example:

    .. math::
       V &:= \sum_{\boldsymbol{x}\in \mathrm{coord.}} V(x_\alpha,x_\beta) \\
         &= V_0 + V_{\mathrm{1MR}} + V_{\mathrm{2MR}}\\

    where,

    .. math::
       V_0 &= V(0,0)\\
       V_{\mathrm{1MR}} &= \sum_{x_\alpha} \left(V(x_\alpha,0) - V_0\right)
                + \sum_{x_\beta} \left(V(0, x_\beta) - V_0\right)\\
       V_{\mathrm{2MR}} &= \sum_{x_\alpha}\sum_{x_\beta} \
        \left(V(x_\alpha,x_\beta) - V(x_\alpha,0)- V(0, x_\beta) - V_0\right)

    """
    if ndof is None:
        if zero_indices is not None:
            ndof = len(zero_indices)
        else:
            ndof = len(dvr_prims)
    elif ndof < len(dvr_prims):
        raise TypeError("ndof must be equal to or larger than dvr_prims length")
    if active_dofs is None:
        active_dofs = [idof for idof in range(ndof)]
    elif len(active_dofs) != len(dvr_prims):
        raise TypeError("active dofs length must be equal to dvr_prims length")
    elif max(active_dofs) > ndof - 1:
        raise TypeError("active_dofs `f` must be in 0 <= f < ndof")

    if site_order is None:
        site_order = {}
        for isite, dof in enumerate(active_dofs):
            site_order[dof] = isite
    elif len(site_order) != ndof:
        raise TypeError
    elif min(site_order.values()) != 0 or max(site_order.values()) != ndof - 1:
        raise TypeError
    dvr_prims_site_order = [dvr_prims[site_order[p]] for p in active_dofs]
    ngrids = [dvr_prims_site_order[p].ngrid for p in range(ndof)]

    nMR_operators: dict[tuple[int, ...], TensorOperator | float] = dict()
    if func is None and db is not None and df is None:
        if zero_indices is None:
            zero_indices = [None for _ in range(ndof)]  # type: ignore
            for i, prim in enumerate(dvr_prims):
                grids = prim.get_grids()
                for j, grid in enumerate(grids):
                    if abs(grid) < 1.0e-10:
                        zero_indices[i] = j
                        break
            if None in zero_indices:
                raise ValueError(
                    "No zero point grid in DVR grids."
                    + "You cannot use n-Mode Representation approx."
                    + "in this DVR primitives"
                )
        with connect(db) as _db:
            logger.info("connected database")
            if dipole:
                dipole_ref = np.inner(
                    next(_db.select(grids=to_dbkey(tuple(zero_indices)))).dipole
                    / debye_in_ase,
                    efield,
                )
                if include_const_in_mpo:
                    nMR_operators[()] = dipole_ref
                else:
                    nMR_operators[()] = 0.0
                logger.info(f"reference permanent dipole moment: {dipole_ref}")
            else:
                if ref_ene is None:
                    ref_ene = (
                        next(
                            _db.select(grids=to_dbkey(tuple(zero_indices)))
                        ).energy
                        / Hartree
                    )
                    nMR_operators[()] = 0.0
                else:
                    nMR_operators[()] = (
                        next(
                            _db.select(grids=to_dbkey(tuple(zero_indices)))
                        ).energy
                        / Hartree
                        - ref_ene
                    )
                logger.info(
                    f"reference energy: {ref_ene} [a.u.], scalar term: {nMR_operators[()]} [a.u.]"
                )

            for iMR in range(1, nMR + 1):
                logger.info(f"read database: {iMR}-mode representation")
                for mode_pair in combinations(active_dofs, iMR):
                    _site_pair = np.array([site_order[p] for p in mode_pair])
                    arg_sort = np.argsort(_site_pair)
                    site_pair: tuple[int, ...] = tuple(_site_pair[arg_sort])
                    shape = tuple(
                        [dvr_prims_site_order[p].ngrid for p in site_pair]
                    )
                    op = TensorOperator(
                        legs=site_pair, shape=shape, only_diag=True
                    )
                    for row in _db.select(dofs=to_dbkey(mode_pair)):
                        full_index = from_dbkey(row.key_value_pairs["grids"])
                        tensor_index: tuple[int, ...] = tuple(
                            np.array([full_index[p] for p in mode_pair])[
                                arg_sort
                            ]
                        )
                        if dipole:
                            op.tensor_orig[tensor_index] = np.inner(
                                row.dipole / debye_in_ase, efield
                            )
                            if not include_const_in_mpo:
                                op.tensor_orig[tensor_index] -= dipole_ref

                        else:
                            op.tensor_orig[tensor_index] = (
                                row.energy / Hartree - ref_ene
                            )
                    nMR_operators[site_pair] = op

        """separation"""
        logger.info("START: separate nMR tensor operators")
        nMR_operators = _separate_recursive(nMR, ngrids, nMR_operators)

    elif func is None and db is None and df is not None:
        if dipole:
            dipole_ref = np.inner(
                df.filter(pl.col("distance") == 0)["dipole"][0], efield
            )
            if include_const_in_mpo:
                nMR_operators[()] = dipole_ref
            else:
                nMR_operators[()] = 0.0
            logger.info(f"reference permanent dipole moment: {dipole_ref}")
        else:
            if ref_ene is None:
                if "energies" in df.schema:
                    ref_ene = df.filter(pl.col("distance") == 0)["energies"][0]
                else:
                    ref_ene = df.filter(pl.col("distance") == 0)["energy"][0]
                nMR_operators[()] = 0.0
            else:
                if "energies" in df.schema:
                    nMR_operators[()] = (
                        df.filter(pl.col("distance") == 0)["energies"][0]
                        - ref_ene
                    )
                else:
                    nMR_operators[()] = (
                        df.filter(pl.col("distance") == 0)["energy"][0]
                        - ref_ene
                    )
            logger.info(
                f"reference energy: {ref_ene} [a.u.], scalar term: {nMR_operators[()]} [a.u.]"
            )

        for iMR in range(1, nMR + 1):
            for mode_pair in combinations(active_dofs, iMR):
                _site_pair = np.array([site_order[p] for p in mode_pair])
                arg_sort = np.argsort(_site_pair)
                site_pair = tuple(_site_pair[arg_sort])
                shape = tuple(
                    [dvr_prims_site_order[p].ngrid for p in site_pair]
                )
                op = TensorOperator(legs=site_pair, shape=shape, only_diag=True)
                for row in df.filter(
                    pl.col("dofs_db") == to_dbkey(mode_pair)
                ).iter_rows(named=True):
                    full_index = tuple(row["grids"])
                    tensor_index = tuple(
                        np.array([full_index[p] for p in mode_pair])[arg_sort]
                    )
                    if dipole:
                        op.tensor_orig[tensor_index] = np.inner(
                            row["dipole"], efield
                        )
                        if not include_const_in_mpo:
                            op.tensor_orig[tensor_index] -= dipole_ref
                    else:
                        if "energy" in row:
                            op.tensor_orig[tensor_index] = (
                                row["energy"] - ref_ene
                            )
                        else:
                            op.tensor_orig[tensor_index] = (
                                row["energies"][0] - ref_ene
                            )
                nMR_operators[site_pair] = op
        """separation"""
        logger.info("START: separate nMR tensor operators")
        nMR_operators = _separate_recursive(nMR, ngrids, nMR_operators)

    elif func is not None and db is None and df is None:
        if tuple() in func:
            nMR_operators[()] = func[()]()
        else:
            nMR_operators[()] = 0.0
        for iMR in range(1, nMR + 1):
            for mode_pair in combinations(active_dofs, iMR):
                if mode_pair not in func:
                    continue
                site_pair = tuple(site_order[p] for p in mode_pair)
                grids = [enumerate(dvr_prims[p].get_grids()) for p in mode_pair]
                shape = tuple([dvr_prims[p].ngrid for p in mode_pair])
                op = TensorOperator(legs=mode_pair, shape=shape, only_diag=True)
                for q_pair in product(*grids):
                    arg_pes = [q[1] for q in q_pair]
                    tensor_index = tuple([q[0] for q in q_pair])
                    op.tensor_orig[tensor_index] = func[mode_pair](*arg_pes)
                nMR_operators[mode_pair] = op

    else:
        raise ValueError(
            "Required either Callable Function or Calculated Data Base"
        )

    """merge"""
    scalar_term, nMR_operators_merged = _merge_nMR_operator_subspace(
        nMR_operators
    )
    if return_tensor:
        return nMR_operators_merged

    if not include_const_in_mpo:
        logger.info(f"scalar term {scalar_term} is excluded from MPO.")
        scalar_term = 0.0

    assert isinstance(scalar_term, float)

    mpo = to_mpo(
        nMR_operators=nMR_operators_merged,  # type: ignore
        ngrids=ngrids,
        scalar_term=scalar_term,
        rate=rate,
        k=k,
        nsweep=nsweep,
    )

    return mpo


def tensor_dict_to_tensor_op(
    tensor_dict: dict[tuple[int, ...], np.ndarray],
) -> dict[tuple[int, ...], TensorOperator]:
    """Convert tensor dictionary to TensorOperator dictionary

    Args:
        tensor_dict (Dict[Tuple[int, ...], np.ndarray]) : Tensor dictionary

    Returns:
        Dict[Tuple[int, ...], TensorOperator] : TensorOperator dictionary
    """
    tensor_op = {}
    for dof_pair, tensor in tensor_dict.items():
        if dof_pair == ():
            continue
        tensor_op[dof_pair] = TensorOperator(
            legs=dof_pair, tensor=tensor, only_diag=True
        )
    return tensor_op


def tensor_dict_to_mpo(
    tensor_dict: dict[tuple[int, ...], np.ndarray],
    rate: float = 1.0,
    nsweep: int = 4,
) -> list[np.ndarray]:
    """Convert tensor dictionary to MPO

    Args:
        tensor_dict (Dict[Tuple[int, ...], np.ndarray]) : Tensor dictionary
        rate (float) : SVD contribution rate. Defaults to 1.0.\
            If ``rate < 1.0``, MPO bond dimension is reduced.\
            Typically, ``rate = 0.999999999`` is enough.
        nsweep (int) : Number of sweep in SVD compression. Defaults to 4

    Returns:
        List[np.ndarray] : MPO

    Notes:
        scalar term ``tensor_dict[()]`` is not included in MPO.

    """
    if not (0.0 < rate <= 1.0):
        raise ValueError("rate must be 0.0 < rate <= 1.0")
    ngrids = []
    dof = 0
    while (dof,) in tensor_dict:
        ngrids.append(tensor_dict[(dof,)].shape[0])
        dof += 1
    tensor_op = tensor_dict_to_tensor_op(tensor_dict)
    mpo = to_mpo(
        nMR_operators=tensor_op,  # type: ignore
        ngrids=ngrids,
        scalar_term=0.0,
        rate=rate,
        nsweep=nsweep,
    )
    return mpo


def _separate_recursive(
    nMR: int,
    ngrids: list[int],
    nMR_operators: dict[tuple[int, ...], float | TensorOperator],
) -> dict[tuple[int, ...], float | TensorOperator]:
    ndof = len(ngrids)
    for iMR in range(1, nMR + 1):
        for mode_pair in combinations(range(ndof), iMR):
            shape = tuple([range(ngrids[p]) for p in mode_pair])
            nMR_target = nMR_operators[mode_pair]
            assert isinstance(nMR_target, TensorOperator)
            for q_indices in product(*shape):
                if mode_pair not in nMR_operators:
                    continue
                nMR_target.tensor_orig[q_indices] -= nMR_operators[()]
                for jMR in range(1, iMR):
                    for mode_pair_sub, q_indices_sub in zip(
                        combinations(mode_pair, jMR),
                        combinations(q_indices, jMR),
                        strict=True,
                    ):
                        nMR_sub = nMR_operators[mode_pair_sub]
                        assert isinstance(nMR_sub, TensorOperator)
                        nMR_target.tensor_orig[q_indices] -= (
                            nMR_sub.tensor_orig[q_indices_sub]
                        )
    return nMR_operators


def construct_fulldimensional(
    dvr_prims: list[DVRPrimitivesMixin],
    func: Callable | None = None,
    db: str | None = None,
    ref_ene: float = 0.0,
    dipole: bool = False,
    efield: tuple[float, float, float] = (1.0e-02, 1.0e-02, 1.0e-02),
) -> dict[tuple[int, ...], TensorOperator]:
    """Construct full-dimensional Operator from DVR grid-based PES

    Args:
        dvr_prims (List[DVRPrimitivesMixin]) : DVR functions. Sorted in Database order.
        func (Optional,Callable) : Potential energy function
        db (Optional,str) : Electronic structure calculation database path, \
            such as 'foo/hoge.db'
        ref_ene (Optional, float) : Opt coord energy in a.u., if you need \
            subtract from function or database energy. \
            Defaults to `0.0`
        dipole (Optional, bool) : Get dipole moment from database. \
            Defaults to `False`
        efield (Optional, List[float, float, float]) : Electronic field for \
            inner products with dipole moment.\
            Defaults to [1.0,1.0,1.0]

    Returns:
        Dict[Tuple[int], TensorOperator] : full-dimensional tensor operator

    """
    ndof = len(dvr_prims)
    grids = [enumerate(dvr_prims[p].get_grids()) for p in range(ndof)]
    shape = tuple([dvr_prims[p].ngrid for p in range(ndof)])
    dofs = tuple([idof for idof in range(ndof)])
    op = TensorOperator(shape=shape, only_diag=True)
    if db is None:
        assert func is not None
        for q_pair in product(*grids):
            arg_pes = tuple([q[1] for q in q_pair])
            q_indices = tuple([q[0] for q in q_pair])
            op.tensor_orig[q_indices] = func(*arg_pes) - ref_ene
    elif func is None:
        with connect(db) as _db:
            for row in _db.select():
                q_indices = from_dbkey(row.key_value_pairs["grids"])
                if dipole:
                    op.tensor_orig[q_indices] = np.inner(
                        row.dipole / debye_in_ase, efield
                    )
                else:
                    op.tensor_orig[q_indices] = row.energy / Hartree - ref_ene
    else:
        raise TypeError
    dic = {dofs: op}
    return dic


def construct_kinetic_operator(
    dvr_prims: list[DVRPrimitivesMixin],
    coefs: list[float] | None = None,
    forms: str = "mpo",
) -> dict[tuple[tuple[int, int], ...], TensorOperator]:
    r"""
    .. note::
        The off-diagonal terms of the kinetic operator are not zero in DVR basis.

    Kinetic energy operator (KEO) in linear coordinate is represented by following matrix product operator (MPO):

    .. math::
       \begin{pmatrix}
       -\frac{\hat{P}_1^2}{2} & 1
       \end{pmatrix}
       \begin{pmatrix}
       1 & 0 \\
       -\frac{\hat{P}_2^2}{2} & 1
       \end{pmatrix}
       \begin{pmatrix}
       1 & 0 \\
       -\frac{\hat{P}_3^2}{2} & 1
       \end{pmatrix}
       \begin{pmatrix}
       1 \\
       -\frac{\hat{P}_4^2}{2}
       \end{pmatrix}

    Args:
        dvr_prims(List[DVRPrimitivesMixin]) : DVR functions
        coefs (List[float]) : Coefficients of kinetic operator. \
            Defaults to ``[1.0, 1.0, ...]``, i.e., :math:`\sum_i -\frac{1}{2}\frac{d^2}{dQ_i^2}` is given. \
            For example, \
            when one uses dimensionless coordinate and rotational coordinate, \
            the kinetic operator is given by \
            :math:`\hat{T} = -\frac{\omega}{2} \frac{d^2}{dx^2} - \frac{1}{2I} \frac{d^2}{d\theta^2}`,\
            then ``coefs`` should be given by :math:`[\omega, 1/I]`.
        forms (str) : Either 'sop' or 'mpo'. Defaults to 'mpo'.

    """
    ndof = len(dvr_prims)
    kinetic_operators = {}
    if coefs is None:
        coefs = [1.0 for _ in range(ndof)]
    if forms.lower() == "mpo":
        operator_key = tuple([(idof, idof) for idof in range(ndof)])
        mpo = []
        for i, (dvr_prim, coef) in enumerate(
            zip(dvr_prims, coefs, strict=True)
        ):
            if i == 0:
                # [[-1/2 * d^2/dQ^2, 1]]
                matrix = np.zeros(
                    (1, dvr_prim.ngrid, dvr_prim.ngrid, 2), dtype=np.complex128
                )
                matrix[0, :, :, 0] = (
                    -1 / 2 * dvr_prim.get_2nd_derivative_matrix_dvr() * coef
                )
                matrix[0, :, :, 1] = np.eye(dvr_prim.ngrid)
            elif i == ndof - 1:
                # [[1              ],
                #  [-1/2 * d^2/dQ^2]]
                matrix = np.zeros(
                    (2, dvr_prim.ngrid, dvr_prim.ngrid, 1), dtype=np.complex128
                )
                matrix[0, :, :, 0] = np.eye(dvr_prim.ngrid)
                matrix[1, :, :, 0] = (
                    -1 / 2 * dvr_prim.get_2nd_derivative_matrix_dvr() * coef
                )
            else:
                # [[1,               0],
                #  [-1/2 * d^2/dQ^2, 1]]
                matrix = np.zeros(
                    (2, dvr_prim.ngrid, dvr_prim.ngrid, 2), dtype=np.complex128
                )
                matrix[0, :, :, 0] = np.eye(dvr_prim.ngrid)
                matrix[1, :, :, 0] = (
                    -1 / 2 * dvr_prim.get_2nd_derivative_matrix_dvr() * coef
                )
                matrix[1, :, :, 1] = np.eye(dvr_prim.ngrid)
            mpo.append(matrix)
        kinetic_operators[operator_key] = TensorOperator(mpo=mpo)
    elif forms.lower() == "sop":
        for idof, (dvr_prim, coef) in enumerate(
            zip(dvr_prims, coefs, strict=True)
        ):
            kinetic_operator = TensorOperator(
                tensor=-1 / 2 * dvr_prim.get_2nd_derivative_matrix_dvr() * coef,
                only_diag=False,
                legs=(idof, idof),
            )
            kinetic_operators[((idof, idof),)] = kinetic_operator
    else:
        raise ValueError("forms must be 'sop' or 'mpo'")

    return kinetic_operators


def _merge_nMR_operator_subspace(
    nMR_operators: dict[tuple[int, ...], float | TensorOperator],
    only_diag: bool = True,
) -> tuple[float | None, dict[tuple[int, ...], TensorOperator]]:
    """

    Merge nMR operator which has same legs, \
        such as V_123 += V_1 + V_2 + V_3 + V_12 + V_13 + V_23

    Args:
        nMR_operators (Dict[Tuple[int], TensorOperator]) : \
            nMR operator dictionary
        only_diag (bool) : Defaults to ``True``

    Returns:
        Tuple[float or complex, Dict[Tuple[int], TensorOperator]] : \
            (scalar term, summed up nMR operator)

    """
    scalar_term = None
    nMR_items = list(nMR_operators.items())
    nMR_items.sort(key=lambda x: -len(x[0]))
    parent = dict()
    sum_op_dict = dict()
    different_only_diag_found = False

    for mode_pair, operator in nMR_items:
        if mode_pair == ():
            assert isinstance(operator, float)
            scalar_term = operator
            continue
        else:
            assert isinstance(operator, TensorOperator)
        if operator.only_diag == only_diag:
            my_parent = mode_pair
            for p_key in parent.keys():
                if set(mode_pair) < set(p_key):
                    my_parent = parent[p_key]
            if my_parent == mode_pair:
                sum_op_dict[mode_pair] = operator
            else:
                sum_op_dict[my_parent] += operator
            parent[mode_pair] = my_parent
        else:
            different_only_diag_found = True

    if different_only_diag_found:
        logger.error("different `only_diag` found.")

    return (scalar_term, sum_op_dict)


def database_to_dataframe(
    db: str,
    reference_id: int | None = None,
    reference_grids: tuple[int, ...] | None = None,
) -> pl.DataFrame:
    """ASE Database to Polars DataFrame

    DataFrame takes more memory than Database, \
        but it is faster than Database.

    Args:
        db (str) : ASE Database path such as "foo/hoge.db". Extension must be ".db"
        reference_id (int) : Database id of reference geometry
        reference_grids (Optional, List[int]) : Grid indices of reference geometry

    Returns:
        pl.DataFrame : Polars DataFrame

    - Dipole is saved in Debye.
    - Energy is saved in Hartree.

    """
    schema = [
        "id",
        "grids",
        "grids_db",
        "dofs",
        "dofs_db",
        "nMR",
        "energy",
        "dipole",
        "distance",
    ]

    if reference_grids is None and reference_id is None:
        raise ValueError(
            "Either `reference_id` or `reference_grid` must be specified."
        )

    data = []
    with connect(db) as _db:
        if reference_grids is None:
            reference_grids = from_dbkey(
                _db.get(id=reference_id).key_value_pairs["grids"]
            )
        if reference_id is None:
            if isinstance(reference_grids, str):
                reference_grids = from_dbkey(reference_grids)
            reference_id = next(_db.select(grids=to_dbkey(reference_grids))).id
        reference_atoms = _db.get(id=reference_id).toatoms()
        reference_energy = _db.get(id=reference_id).energy / Hartree
        logger.info(
            f"reference_id: {reference_id}"
            + f" reference_grids: {reference_grids}"
            + f" reference_atoms: {reference_atoms}"
            + f" reference_energy: {reference_energy}"
        )
        for row in _db.select():
            grids = from_dbkey(row.grids)
            dofs = from_dbkey(row.dofs)
            nMR = _get_mode_representation(grids, reference_grids)
            _data_row = [row.id, grids, row.grids, dofs, row.dofs, nMR]
            try:
                _data_row.append(row.energy / Hartree)
            except Exception:
                _data_row.append(None)
            try:
                _data_row.append((row.dipole / debye_in_ase).tolist())
            except Exception:
                _data_row.append(None)
            _data_row.append(_get_manhattan_distance(grids, reference_grids))

            data.append(_data_row)

    df = pl.DataFrame(data, schema, orient="row")

    """polars does not support pickle"""
    df.write_parquet(db.replace(".db", ".parquet"))
    """When you want to load parquet file, use
    df = pl.parquet(db.replace(".db", ".parquet"))
    """
    logger.info(f'DataFrame is saved as {db.replace(".db", ".parquet")}')

    return df


def _get_grids_df(df, grids, dofs):
    """This method is too lazy but I have no idea how to check filtering of list[int64]"""
    for dof in dofs:
        df = df.filter(pl.col("grids").apply(lambda x: x[dof] == grids[dof]))
    return df


def _get_manhattan_distance(array1, array2) -> int:
    """Get Manhattan distance between two arrays"""
    return int(np.sum(np.abs(np.array(array1) - np.array(array2))))


def _get_mode_representation(array, ref_array) -> int:
    """Get number ofmode representation

    Args:
        array (_ArrayLike) : Array to be compared
        ref_array (_ArrayLike) : Reference array

    Returns:
        int : Number of mode representation

    If you give (1,2,3) and (2,2,2) as array and ref_array, \
    this function returns 2 because (1,2,3) and (2,2,2) have two different modes.

    """
    return int(np.sum(np.array(array) != np.array(ref_array)))
