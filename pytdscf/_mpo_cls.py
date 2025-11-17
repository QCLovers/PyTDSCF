"""
Matrix Product Operator (MPO) class
"""

from __future__ import annotations

import itertools
from typing import Annotated

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
from loguru import logger as _logger
from numpy.typing import NDArray
from scipy.linalg import LinAlgError
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import svds

import pytdscf
from pytdscf._const_cls import const
from pytdscf._helper import get_tensornetwork_diagram_MPO

logger = _logger.bind(name="main")

CoreMxNxM = Annotated[NDArray[np.float64], "shape=(M_{p-1}, N_{p}, M_{p})"]
CoreMxNxNxM = Annotated[
    NDArray[np.float64], "shape=(M_{p-1}, N_{p}, N_{p+1}, M_{p+1})"
]
CoreMNxM = Annotated[NDArray[np.float64], "shape=(M_{p-1}*N_{p}, M_{p})"]
GridMPO = Annotated[list[CoreMxNxM], "MPO"]


def _dense_svd(mat: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    try:
        U, s, Vh = scipy.linalg.svd(mat, full_matrices=False, overwrite_a=True)
    except LinAlgError:
        U, s, Vh = scipy.linalg.svd(
            mat, full_matrices=False, lapack_driver="gesvd", overwrite_a=True
        )
    return U, s, Vh


class MatrixProductOperators:
    r""" MPO class

    Note : We distinguish SVD or QRD -based MPO from sum-of-priduct (SOP) operator.

    Args:
        nsite (int) : Site Length
        operators(Dict[Tuple[int], np.ndarray | jax.Array]) : MPOs


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
       j_2 = 0, 1, .. l-1

    ``tensor = np.array([W0, W1, W3])``.

    Attributes:
        nsite (int) : Site Length
        operators (Dict[Tuple[int], np.ndarray | jax.Array]) : MPOs
        calc_point (List[List[OperatorCore]]) : [p] is the list of calculation need tensor core.
    """

    calc_point: list[list[OperatorCore]]

    def __init__(
        self,
        nsite: int,
        operators: dict[
            tuple[int | tuple[int, int], ...],
            list[np.ndarray] | list[jax.Array],
        ],
        backend: str,
    ):
        self.nsite = nsite
        self.operators = operators
        self.calc_point = [[] for isite in range(self.nsite)]
        if backend.lower() == "numpy":
            self.backend = "numpy"
        elif backend.lower() == "jax":
            self.backend = "jax"
        else:
            raise ValueError(
                f"backend must be jax or numpy, but {backend} is given"
            )
        self._searach_calc_point()

    def __repr__(self) -> str:
        message = "MPO : \n"
        for key, mpo in self.operators.items():
            message += f"key : {key} \n"
            message += get_tensornetwork_diagram_MPO(mpo)
        return message

    def _searach_calc_point(self):
        for key, mpo in self.operators.items():
            original_key = key
            parent_key: list[int] = []
            for ind, core in zip(key, mpo, strict=True):
                if isinstance(ind, tuple):
                    if len(ind) != len(core.shape) - 2:
                        raise ValueError(
                            f"MPO core index {ind} is not consistent with MPO core shape {core.shape}"
                        )
                    if len(set(ind)) != 1:
                        raise ValueError(
                            f"MPO core index must be single DOFs, but assinged {ind}"
                        )
                    parent_key.append(ind[0])
                elif isinstance(ind, int):
                    if len(core.shape) - 2 != 1:
                        raise ValueError(
                            f"MPO core index {ind} in {key} is not consistent with MPO core shape {core.shape}"
                        )
                    parent_key.append(ind)
                else:
                    raise ValueError(
                        f"MPO core index type is wrong. Must be int or Tuple[int,int], but {ind} is given"
                    )

            for ind, core in zip(parent_key, mpo, strict=True):
                assert isinstance(core, np.ndarray | jax.Array | int)
                self.calc_point[ind].append(
                    OperatorCore(
                        parent_key=parent_key,
                        original_key=original_key,
                        psite=ind,
                        data=core,
                        backend=self.backend,
                    )
                )
            for ind in range(min(parent_key) + 1, max(parent_key)):
                if ind not in parent_key:
                    self.calc_point[ind].append(
                        OperatorCore(
                            parent_key=parent_key,
                            original_key=original_key,
                            psite=ind,
                            data=1,
                            backend=self.backend,
                        )
                    )


class OperatorCore:
    r"""Matrix Product Operator Core

    :math:``W\substack{j_p^\prime \\ \beta_{p-1}\beta_{p}\\ j_p}``

    Attributes:
        key (Tuple[int]) : operator group key
        psite (int) : p-site. p is included in key.
        data (jax.Array | np.ndarray | int) : 3 or 4 rank tensor. If only identity ovlp, this is integer 1.
        right_side (bool) : If this core is right side of group
        left_side (bool) : If this core is left side of group
        only_diag (bool) : only diagonal contraction.
        is_unitmat_op (bool) : operator is identity operator.

    """

    def __init__(
        self,
        parent_key: list[int],
        original_key: tuple[int | tuple[int, int], ...],
        psite: int,
        data: jax.Array | np.ndarray | int,
        backend,
    ):
        self.key = original_key
        self.psite = psite
        self.data = data
        self.is_right_side = max(parent_key) == psite
        self.is_left_side = min(parent_key) == psite
        self.is_unitmat_op = isinstance(data, int)
        self.only_diag = (
            True if isinstance(data, int) or len(data.shape) == 3 else False
        )
        self.backend = backend

        if isinstance(data, np.ndarray | jax.Array):
            self.shape = data.shape
            self.size = data.size
            if backend.lower() == "jax":
                self.data = jnp.array(data, dtype=jnp.complex128)

    def __repr__(self) -> str:
        message = f"key : {self.key} \n"
        message += f"site : {self.psite} \n"
        message += f"diag : {self.only_diag} \n"
        message += f"data : {self.data}"
        return message

    def is_hermitian(self) -> bool:
        if isinstance(self.data, int):
            return True
        if self.only_diag:
            return True
        else:
            triu_inds = np.triu_indices(self.data.shape[1], k=1)
            tril_inds = np.tril_indices(self.data.shape[1], k=-1)
            return np.allclose(
                self.data[:, triu_inds, :], self.data[:, tril_inds, :].conj()
            )


def decompose_and_fill_nMR(
    tensor: pytdscf.dvr_operator_cls.TensorOperator | float,
    dofs: tuple[int, ...],
    ngrids: list[int],
) -> GridMPO:
    """Decompose nMR tensor

    Args:
        tensor (np.ndarray) : n-dimensional tensor
        dofs (Tuple[int]) : which degree of freedoms. 1-index.
        ngrids (List[int]) : number of grids for each dofs

    Returns:
        List[np.ndarray] : Decomposed tensors

    """
    mpo = []

    if isinstance(tensor, float):
        """scalar scale"""
        for center in ngrids:
            tmp = np.array(
                [np.eye(N=1, M=1, dtype=np.float64) for _ in range(center)],
                dtype=np.float64,
            )
            mpo.append(np.swapaxes(tmp, 0, 1))
        mpo[-1] *= tensor

    else:
        dec_tensors = list(reversed(tensor.decompose(decompose_type="QR")))
        left = 1
        right = dec_tensors[-1].shape[0]
        for idof, center in enumerate(ngrids):
            if idof in dofs:
                mpo.append(dec_tensors.pop())
                left = mpo[-1].shape[2]
                if dec_tensors:
                    right = dec_tensors[-1].shape[0]
                else:
                    right = 1
            else:
                tmp = np.array(
                    [
                        np.eye(N=left, M=right, dtype=np.float64)
                        for _ in range(center)
                    ],
                    dtype=np.float64,
                )
                mpo.append(np.swapaxes(tmp, 0, 1))

    return mpo


def guess_bond_dimension(svals, rate=0.999999999999, trace=None) -> int:
    """svals=sval square"""
    if trace is None:
        trace = np.sum(svals)
    else:
        if np.sum(svals) / trace < rate:
            if const.verbose > 2:
                logger.warning(f"Contribution rate = {np.sum(svals) / trace}")
            return svals.size
    if rate == 1.0:
        return svals.size
    elif 0.0 < rate < 1.0:
        cumsum = 0.0
        rank = 0
        while cumsum / trace < rate and rank < svals.size:
            cumsum += svals[rank]
            rank += 1
        if const.verbose > 2:
            logger.debug(f"Contribution rate = {cumsum / trace}")
        return rank
    else:
        raise ValueError("contribution rate must be 0.0 < `rate` < 1.0")


def merge_mpos(
    mpos: list[list[np.ndarray]],
    sparse: bool = True,
    k: int = 50,
    rate: float = 0.999999999999,
) -> list[np.ndarray]:
    """We recomment use merge_mpos_twodom() instead of this function."""
    mpo_merged = []
    nsite = len(mpos[0])
    norm = 1.0  # normalize each site to reduce numerical error
    if sparse:
        center = nsite // 2
        for isite in range(nsite // 2):
            left_dim = [0]
            right_dim = [0]
            center_dim = mpos[0][isite].shape[1]
            for mpo in mpos:
                left_dim.append(left_dim[-1] + mpo[isite].shape[0])
                right_dim.append(right_dim[-1] + mpo[isite].shape[2])
            if const.verbose == 4:
                logger.debug(
                    f"{isite}-site : {mpo_merged.__sizeof__() / 10**6} MBytes"
                )
            if isite == 0:
                matrix_shape = (center_dim, right_dim[-1])
                lmat = lil_matrix(matrix_shape)
                for i, mpo in enumerate(mpos):
                    assert 1 == mpo[isite].shape[0]
                    lmat[:, right_dim[i] : right_dim[i + 1]] = mpo[isite][
                        0, :, :
                    ]
                cmat = lmat.tocsr()
                norm_site = scipy.sparse.linalg.norm(cmat)
                norm *= norm_site
                cmat /= norm_site
                trace = np.sum((cmat @ cmat.T).diagonal())
                if k < min(*cmat.shape):
                    # U, s, Vh = svds(cmat, k=max(min(k, min(*cmat.shape)), 1),
                    # solver='propack', which='LM')
                    logger.warning("matrix is too large to execute full-SVD")
                    U, s, Vh = svds(
                        cmat,
                        k=max(min(k, min(*cmat.shape) - 1), 1),
                        solver="arpack",
                        which="LM",
                    )
                    U = U[:, ::-1]
                    s = s[::-1]
                    Vh = Vh[::-1, :]
                else:
                    U, s, Vh = _dense_svd(cmat.todense())

                bd = guess_bond_dimension(s**2, rate=rate, trace=trace)
                mpo_merged.append(
                    (U[:, :bd] @ np.diag(s[:bd])).reshape(1, center_dim, bd)
                )
            else:
                matrix_shape = (left_dim[-1], center_dim * right_dim[-1])
                data = []
                row = []
                col = []
                for i, mpo in enumerate(mpos):
                    core = mpo[isite]
                    for a, b, c in itertools.product(
                        range(core.shape[0]),
                        range(core.shape[1]),
                        range(core.shape[2]),
                    ):
                        if abs(core[a, b, c]) > 1.0e-16:
                            data.append(core[a, b, c])
                            row.append(left_dim[i] + a)
                            col.append(right_dim[-1] * b + c + right_dim[i])
                cmat = csr_matrix(
                    (np.array(data), (np.array(row), np.array(col))),
                    shape=matrix_shape,
                )
                norm_site = scipy.sparse.linalg.norm(cmat)
                norm *= norm_site
                cmat /= norm_site
                bd_prev = bd
                mat: np.ndarray = Vh[:bd_prev, :] @ cmat
                if isite == nsite // 2 - 1:
                    mpo_merged.append(mat.reshape(bd_prev, center_dim, -1))
                else:
                    mat = mat.reshape(bd * center_dim, right_dim[-1])
                    trace = np.tensordot(mat, mat.T, axes=[[1, 0], [0, 1]])
                    if const.verbose == 4:
                        logger.debug(f"SVD shape = {mat.shape}")
                    """mat @ mat may be time consuming part"""
                    if k < min(*mat.shape):
                        # U, s, Vh = svds(
                        # mat, k=max(min(k, min(*mat.shape)), 1),
                        # solver='propack', which='LM')
                        # If you has LinAlgError, try arpack
                        logger.warning(
                            "matrix is too large to execute full-SVD"
                        )
                        U, s, Vh = svds(
                            mat,
                            k=max(min(k, min(*mat.shape) - 1), 1),
                            solver="arpack",
                            which="LM",
                        )
                        U = U[:, ::-1]
                        s = s[::-1]
                        Vh = Vh[::-1, :]
                    else:
                        U, s, Vh = _dense_svd(mat)

                    bd = guess_bond_dimension(s**2, rate=rate, trace=trace)
                    if const.verbose == 4:
                        logger.debug(f"{isite} site bond-dimension = {bd}")
                    if bd == k:
                        center = isite + 1
                        mpo_merged.append(mat.reshape(bd_prev, center_dim, -1))
                        break
                    else:
                        mpo_merged.append(
                            (U[:, :bd] @ np.diag(s[:bd])).reshape(
                                -1, center_dim, bd
                            )
                        )

        mpo_merged_from_left = []
        mpos = [[np.swapaxes(core, 0, 2) for core in mpo] for mpo in mpos]
        for isite in range(nsite - 1, center - 1, -1):
            left_dim = [0]
            right_dim = [0]
            center_dim = mpos[0][isite].shape[1]
            for mpo in mpos:
                left_dim.append(left_dim[-1] + mpo[isite].shape[0])
                right_dim.append(right_dim[-1] + mpo[isite].shape[2])
            if const.verbose == 4:
                logger.debug(
                    f"{isite}-site : {mpo_merged.__sizeof__() / 10**6} MBytes"
                )
            if isite == nsite - 1:
                matrix_shape = (center_dim, right_dim[-1])
                lmat = lil_matrix(matrix_shape)
                for i, mpo in enumerate(mpos):
                    assert 1 == mpo[isite].shape[0]
                    lmat[:, right_dim[i] : right_dim[i + 1]] = mpo[isite][
                        0, :, :
                    ]
                cmat = lmat.tocsr()
                norm_site = scipy.sparse.linalg.norm(cmat)
                norm *= norm_site
                cmat /= norm_site
                trace = np.sum((cmat @ cmat.T).diagonal())
                if k < min(*cmat.shape):
                    logger.warning("matrix is too large to execute full-SVD")
                    U, s, Vh = svds(
                        cmat,
                        k=max(min(k, min(*cmat.shape)), 1),
                        solver="propack",
                        which="LM",
                    )
                    U = U[:, ::-1]
                    s = s[::-1]
                    Vh = Vh[::-1, :]
                else:
                    try:
                        U, s, Vh = scipy.linalg.svd(
                            cmat.todense(),
                            full_matrices=False,
                            overwrite_a=True,
                        )
                    except LinAlgError:
                        U, s, Vh = scipy.linalg.svd(
                            cmat.todense(),
                            full_matrices=False,
                            overwrite_a=True,
                            lapack_driver="gesvd",
                        )
                bd = guess_bond_dimension(s**2, rate=rate, trace=trace)
                mpo_merged_from_left.append(
                    np.swapaxes(
                        (U[:, :bd] @ np.diag(s[:bd])).reshape(
                            1, center_dim, bd
                        ),
                        0,
                        2,
                    )
                )
            else:
                matrix_shape = (left_dim[-1], center_dim * right_dim[-1])
                data = []
                row = []
                col = []
                for i, mpo in enumerate(mpos):
                    core = mpo[isite]
                    for a, b, c in itertools.product(
                        range(core.shape[0]),
                        range(core.shape[1]),
                        range(core.shape[2]),
                    ):
                        if abs(core[a, b, c]) > 1.0e-16:
                            data.append(core[a, b, c])
                            row.append(left_dim[i] + a)
                            col.append(right_dim[-1] * b + c + right_dim[i])
                cmat = csr_matrix(
                    (np.array(data), (np.array(row), np.array(col))),
                    shape=matrix_shape,
                )
                norm_site = scipy.sparse.linalg.norm(cmat)
                norm *= norm_site
                cmat /= norm_site
                mat = Vh[:bd, :] @ cmat
                if isite == center:
                    mpo_merged_from_left.append(
                        np.swapaxes(mat.reshape(bd, center_dim, -1), 0, 2)
                    )
                else:
                    mat = mat.reshape(bd * center_dim, right_dim[-1])
                    trace = np.tensordot(mat, mat.T, axes=[[1, 0], [0, 1]])
                    if const.verbose == 4:
                        logger.debug(f"SVD shape = {mat.shape}")
                    """mat @ mat may be time consuming part"""
                    if k < min(*mat.shape):
                        # U, s, Vh = svds(
                        # mat, k=max(min(k, min(*mat.shape)), 1),
                        # solver='propack', which='LM')
                        # If you has LinAlgError, try arpack
                        logger.warning(
                            "matrix is too large to execute full-SVD"
                        )
                        U, s, Vh = svds(
                            mat,
                            k=max(min(k, min(*mat.shape) - 1), 1),
                            solver="arpack",
                            which="LM",
                        )
                        U = U[:, ::-1]
                        s = s[::-1]
                        Vh = Vh[::-1, :]
                    else:
                        U, s, Vh = _dense_svd(mat)

                    bd = guess_bond_dimension(s**2, rate=rate, trace=trace)
                    if const.verbose == 4:
                        logger.debug(f"{isite} site bond-dimension = {bd}")
                    mpo_merged_from_left.append(
                        np.swapaxes(
                            (U[:, :bd] @ np.diag(s[:bd])).reshape(
                                -1, center_dim, bd
                            ),
                            0,
                            2,
                        )
                    )
        mpo_merged.extend(mpo_merged_from_left[::-1])
        mpo_merged[-1] *= norm
    else:
        for isite in range(nsite):
            left_dim = [0]
            right_dim = [0]
            center_dim = mpos[0][isite].shape[1]
            for mpo in mpos:
                left_dim.append(left_dim[-1] + mpo[isite].shape[0])
                right_dim.append(right_dim[-1] + mpo[isite].shape[2])
            if isite == 0:
                core_merged = np.zeros((1, center_dim, right_dim[-1]))
                for i, mpo in enumerate(mpos):
                    core_merged[:, :, right_dim[i] : right_dim[i + 1]] = mpo[
                        isite
                    ]
            elif isite == nsite - 1:
                core_merged = np.zeros((left_dim[-1], center_dim, 1))
                for i, mpo in enumerate(mpos):
                    core_merged[left_dim[i] : left_dim[i + 1], :, :] = mpo[
                        isite
                    ]
            else:
                core_merged = np.zeros(
                    (left_dim[-1], center_dim, right_dim[-1])
                )
                for i, mpo in enumerate(mpos):
                    core_merged[
                        left_dim[i] : left_dim[i + 1],
                        :,
                        right_dim[i] : right_dim[i + 1],
                    ] = mpo[isite]

            mpo_merged.append(core_merged)
    return mpo_merged


def merge_mpos_twodot(
    mpos: list[list[np.ndarray]],
    k: int = 50,
    rate: float = 0.999999999999,
) -> list[np.ndarray]:
    mpo_merged: list[np.ndarray] = []
    nsite: int = len(mpos[0])
    init_core: NDArray = np.concatenate(
        tuple([mpo[0] for mpo in mpos]), axis=2, dtype=np.float64
    )
    assert init_core.shape[0] == 1
    mpo_merged.append(init_core)
    left_site: int = 0
    right_site: int = 1

    while right_site < nsite:
        """
          j     l
          |     |
        i-.-k k-.-m

        ==> SVD (ij, lm) matrix
        """

        k_dim: list[int] = [0]
        l_dim: int = mpos[0][right_site].shape[1]
        m_dim: list[int] = [0]
        for mpo in mpos:
            k_dim.append(k_dim[-1] + mpo[right_site].shape[0])
            m_dim.append(m_dim[-1] + mpo[right_site].shape[2])
        core_left: CoreMxNxM = mpo_merged[-1]
        mat_left: CoreMNxM = core_left.reshape(-1, core_left.shape[2])

        if right_site == nsite - 1:
            core_right: CoreMxNxM = np.concatenate(
                tuple([mpo[-1] for mpo in mpos]), axis=0
            )
            assert core_right.shape[-1] == 1
            mat_right: NDArray = core_right[:, :, 0]
        else:
            matrix_shape: tuple[int, int] = (k_dim[-1], l_dim * m_dim[-1])
            data: list[float] = []
            row: list[int] = []
            col: list[int] = []
            for i, mpo in enumerate(mpos):
                core: NDArray = mpo[right_site]
                for a, b, c in itertools.product(
                    range(core.shape[0]),
                    range(core.shape[1]),
                    range(core.shape[2]),
                ):
                    if abs(core[a, b, c]) > 1.0e-16:
                        data.append(core[a, b, c])
                        row.append(k_dim[i] + a)
                        col.append(m_dim[-1] * b + c + m_dim[i])
            mat_right = csr_matrix(
                (np.array(data), (np.array(row), np.array(col))),
                shape=matrix_shape,
                dtype=np.float64,
            )

        mat: NDArray = mat_left @ mat_right
        _norm = np.linalg.norm(mat)
        mat /= _norm
        if const.verbose > 2:
            logger.info(
                f"Execute {mat.shape} matrix SVD in "
                + f"{left_site}-{right_site} sites MPO optimization"
            )
        if min(mat.shape) > 10**5:
            logger.warning("matrix is too large to execute full-SVD")
            # Note that arpack cannot calculate all singular value.
            # Propack in scipy > 1.9.0 support full SVD in sparse matrix
            U, s, Vh = svds(
                mat,
                k=max(min(k, min(*mat.shape) - 1), 1),
                solver="arpack",
                which="LM",
            )
            U = U[:, ::-1]
            s = s[::-1]
            Vh = Vh[::-1, :]
            trace = np.sum(np.diag((mat @ mat.T)))
            bd = guess_bond_dimension(s**2, rate=rate, trace=trace)
        else:
            U, s, Vh = _dense_svd(mat)
            bd = guess_bond_dimension(s, rate=rate)
        core_left = U[:, :bd].reshape(
            core_left.shape[0], core_left.shape[1], bd
        )
        if right_site == nsite - 1:
            core_right = (np.diag(s[:bd]) @ Vh[:bd, :]).reshape(bd, l_dim, 1)
        else:
            core_right = (np.diag(s[:bd]) @ Vh[:bd, :]).reshape(
                bd, l_dim, m_dim[-1]
            )
        mpo_merged[-1] = core_left * np.sqrt(_norm)
        mpo_merged.append(core_right * np.sqrt(_norm))
        left_site += 1
        right_site += 1
    return mpo_merged


def sweep_compress(
    mpo: list[np.ndarray], rate=0.999999999, left_to_right=True
) -> list[np.ndarray]:
    raise NotImplementedError(
        "We recommend to use merge_mpos_twodot instead of this function"
    )
    nsite = len(mpo)
    bond_dimension = [1]
    for core in mpo:
        bond_dimension.append(core.shape[2])
    bond_dimension.append(1)
    if left_to_right:
        for isite in range(nsite - 1):
            core = mpo[isite]
            U, s, Vh = _dense_svd(core.reshape(-1, core.shape[2]))
            bd = guess_bond_dimension(s, rate)
            bond_dimension[isite] = bd
            mpo[isite] = U[:, :bd].reshape(-1, core.shape[1], bd)
            mpo[isite + 1] = np.einsum(
                "ij,jkl->ikl", np.diag(s[:bd]) @ Vh[:bd, :], mpo[isite + 1]
            )
    else:
        for isite in range(nsite - 1, 0, -1):
            core = mpo[isite]
            U, s, Vh = _dense_svd(core.reshape(core.shape[0], -1))
            bd = guess_bond_dimension(s, rate)
            bond_dimension[isite - 1] = bd
            mpo[isite] = (Vh[:bd, :]).reshape(bd, core.shape[1], -1)
            mpo[isite - 1] = np.einsum(
                "ijk,kl->ijl", mpo[isite - 1], U[:, :bd] @ np.diag(s[:bd])
            )
    return mpo


def sweep_compress_twodot(
    mpo: list[np.ndarray],
    rate=0.999999999,
    left_to_right=True,
) -> list[np.ndarray]:
    nsite = len(mpo)
    if left_to_right:
        start = 0
        end = nsite - 1
        step = +1
    else:
        start = nsite - 2
        end = -1
        step = -1
    for left_site in range(start, end, step):
        right_site = left_site + 1
        core_left = mpo[left_site]
        core_right = mpo[right_site]
        concat_core = np.einsum("ijk,klm->ijlm", core_left, core_right)
        concat_mat = concat_core.reshape(
            core_left.shape[0] * core_left.shape[1],
            core_right.shape[1] * core_right.shape[2],
        )
        _norm = np.linalg.norm(concat_mat)
        concat_mat /= _norm
        U, s, Vh = _dense_svd(concat_mat)
        bd = guess_bond_dimension(s, rate)
        if left_to_right:
            core_left = U[:, :bd].reshape(
                core_left.shape[0], core_left.shape[1], bd
            )
            core_right = (np.diag(s[:bd]) @ Vh[:bd, :]).reshape(
                bd, core_right.shape[1], core_right.shape[2]
            )
        else:
            core_left = (U[:, :bd] @ np.diag(s[:bd])).reshape(
                core_left.shape[0], core_left.shape[1], bd
            )
            core_right = (Vh[:bd, :]).reshape(
                bd, core_right.shape[1], core_right.shape[2]
            )
        mpo[left_site] = core_left * np.sqrt(_norm)
        mpo[right_site] = core_right * np.sqrt(_norm)
        if const.verbose > 2:
            logger.debug(f"MPO bond dimension {left_site}-{right_site} = {bd}")

    return mpo


def sweep_qr(mpo: list[np.ndarray]) -> list[np.ndarray]:
    nsite = len(mpo)
    for isite in range(0, nsite-1):
        core = mpo[isite]
        _norm = np.linalg.norm(core)
        core /= _norm
        Q, R = np.linalg.qr(
            core.reshape(core.shape[0] * core.shape[1], core.shape[2]),
            mode="reduced",
        )
        mpo[isite] = Q.reshape(core.shape[0], core.shape[1], -1) * np.sqrt(_norm)
        mpo[isite + 1] = np.einsum(
            "ij,jkl->ikl", R, mpo[isite + 1]
        ) * np.sqrt(_norm)

    return mpo

def sweep_lq(mpo: list[np.ndarray]) -> list[np.ndarray]:
    nsite = len(mpo)
    for isite in range(nsite-1, 0, -1):
        core = mpo[isite]
        _norm = np.linalg.norm(core)
        core /= _norm
        L, Q = np.linalg.qr(
            core.reshape(core.shape[0], core.shape[1] * core.shape[2]).T,
            mode="reduced",
        )
        np.testing.assert_allclose(core, (L.T @ Q.T).reshape(core.shape))
        mpo[isite] = Q.T.reshape(-1, core.shape[1], core.shape[2]) * np.sqrt(_norm)
        mpo[isite - 1] = np.einsum(
            "ijk,kl->ijl", mpo[isite - 1], L.T
        ) * np.sqrt(_norm)
    return mpo

def to_mpo(
    nMR_operators: dict[
        tuple[int, ...], float | pytdscf.dvr_operator_cls.TensorOperator
    ],
    ngrids: list[int],
    scalar_term: float = 0.0,
    rate: float = 0.999999999,
    k: int = 1000,
    nsweep: int = 1,
    mode: str = "block-by-block",
) -> GridMPO:
    """Convert nMR operators to single MPO

    Args:
        nMR_operators (Dict[Tuple[int, ...], NDArray]): nMR operators
        ngrids (List[int]): number of grids for each dimension
        scalar_term (Optional[float], optional): scalar term. Defaults to 0.0. \
            We recommend to set it to 0.0 and add it as a constant term ``coupleJ`` in the Hamiltonian.
        rate (float, optional): compression rate. Defaults to 0.999999999.
        k (int, optional): number of singular values to calculate in sparse mode. Defaults to 1000.
        nsweep (int, optional): number of sweep in compression. Defaults to 4.
        mode (str, optional): compression mode. Defaults to 'block-by-block'. \
            'block-by-block' is numerically stable but expensive. 'term-by-term' is cheap but numerically unstable.\
            otherwise, no compression is performed (very, very sparse MPO will be constructed)
    Returns:
        GridMPO: Matrix product operator

    """
    mpos: list[GridMPO] = []
    for key, val in nMR_operators.items():
        mpos.append(decompose_and_fill_nMR(tensor=val, dofs=key, ngrids=ngrids))
    if scalar_term != 0.0:
        mpos.append(
            decompose_and_fill_nMR(float(scalar_term), dofs=(), ngrids=ngrids)
        )
    if const.verbose > 2:
        logger.info(f"DONE: construct {len(mpos)} quasi full-dimensional MPOs")

    if mode == "block-by-block":
        mpo = _compress_block_by_block(mpos, rate, nsweep, k)
    elif mode == "term-by-term":
        raise NotImplementedError("term-by-term compression is not valid")
        # mpo = _compress_term_by_term(mpos, rate, nsweep)
    else:
        logger.warning(
            f"Unknown mode: {mode}. No compression is performed. (too silly)"
        )
        mpo = _sum_of_mpos_to_single_mpo(mpos)

    bond_dim = []
    for core in mpo[:-1]:
        bond_dim.append(core.shape[2])
    logger.info(f"final MPO bond-dimension {bond_dim}")
    return mpo


def _compress_block_by_block(mpos, rate, nsweep, k):
    sub_mpo = 50
    n_loop = len(mpos) // sub_mpo
    if len(mpos) % sub_mpo != 0:
        n_loop += 1
    _mpos = []
    for i in range(n_loop):
        low_index = i * sub_mpo
        high_index = min((i + 1) * sub_mpo, len(mpos) + 1)
        logger.info(
            f"{low_index}-{high_index - 1}: "
            + "part of full-dimensional MPOs optimization"
        )
        _mpo = merge_mpos_twodot(mpos[low_index:high_index], k=k, rate=rate)
        # Canonicalise at first
        _mpo = sweep_qr(_mpo)
        # Then, compress
        _mpo = sweep_compress_twodot(_mpo, rate=rate, left_to_right=False)
        _mpos.append(_mpo)
    logger.info(f"{0}-{len(_mpos)}: full-dimensional MPOs optimization")
    mpo = merge_mpos_twodot(_mpos, k=k, rate=rate)
    del mpos  # free RAM
    del _mpos
    for isweep in range(nsweep):
        logger.info(f"{isweep}-sweep: full-dimensional MPOs optimization")
        # Canonicalise at first
        mpo = sweep_qr(mpo)
        # Then, compress
        mpo = sweep_compress_twodot(mpo, rate=rate, left_to_right=False)
    return mpo


def _sum_of_mpos_to_single_mpo(mpos: list[GridMPO]) -> GridMPO:
    """Convert sum of MPOs to single MPO

    Args:
        mpos (List[MPO]): sum of MPOs

    Returns:
        MPO: single MPO

    Notes:
        Before using this function, you should compress each MPOs, otherwise it will consume too much RAM.
    """
    nsite = len(mpos[0])
    allocated_mpo: GridMPO = []
    core: CoreMxNxM
    for isite in range(nsite):
        if isite == 0:
            # W = [W_1, W_2, ..., W_n]
            core = np.concatenate(
                tuple([mpo[0] for mpo in mpos]),
                axis=2,
            )
            assert core.shape[0] == 1
        elif isite == nsite - 1:
            # W = [W_1, W_2, ..., W_n]^T
            core = np.concatenate(
                tuple([mpo[-1] for mpo in mpos]),
                axis=0,
            )
            assert core.shape[-1] == 1
        else:
            # W = [[W_1, 0, ..., 0],
            #      [0, W_2, ..., 0],
            #      ...
            #      [0, 0, ..., W_n]]
            left_indices: list[int] = np.cumsum(
                [0] + [mpo[isite].shape[0] for mpo in mpos]  # type: ignore
            )
            right_indices: list[int] = np.cumsum(
                [0] + [mpo[isite].shape[2] for mpo in mpos]  # type: ignore
            )
            center_index = mpos[0][isite].shape[1]
            core = np.zeros((left_indices[-1], center_index, right_indices[-1]))
            for i, mpo in enumerate(mpos):
                core[
                    left_indices[i] : left_indices[i + 1],
                    :,
                    right_indices[i] : right_indices[i + 1],
                ] = mpo[isite]
        allocated_mpo.append(core)
    return allocated_mpo


def _twodot_svd(core: CoreMxNxNxM) -> tuple[NDArray, NDArray, NDArray]:
    shape = core.shape
    assert len(shape) == 4
    mat = core.reshape(shape[0] * shape[1], shape[2] * shape[3])
    U, s, Vh = _dense_svd(mat)
    return (U, s, Vh)


def _sweep_compress_term_by_term(
    mpos: list[GridMPO], rate: float, left_to_right: bool
):
    """
    1. Two-dot SVD mpo by mpo
    2. Truncate bond dimension (using argsort)
    3. Allocate new cores into mpo
    """
    nsite: int = len(mpos[0])
    nmpo: int = len(mpos)
    assert all([len(mpo) == nsite for mpo in mpos])
    if left_to_right:
        # terminal site
        # W[0] = [W1[0], W2[0], ..., Wn[0]]
        # W[1] = diag([W1[1], W2[1], ..., Wn[1]])
        # W[0]W[1] = [W1[0]W1[1], W2[0]W2[1], ..., Wn[0]Wn[1]] cannot be calculated by term-by-term SVD.
        left_site: int = 1
        right_site: int = 2
    else:
        left_site = nsite - 3
        right_site = nsite - 2

    while right_site < nsite and left_site >= 0:
        """
           j       l
           |       |
        i-[W]-k k-[W]-m

        ==> SVD (ij, lm) matrix
        """
        left_cores: list[CoreMxNxM] = [mpo[left_site] for mpo in mpos]
        right_cores: list[CoreMxNxM] = [mpo[right_site] for mpo in mpos]
        merged_cores: list[CoreMxNxNxM] = [
            np.einsum("ijk,klm->ijlm", left_core, right_core)
            for left_core, right_core in zip(
                left_cores, right_cores, strict=True
            )
        ]
        norm: np.float64 = np.sqrt(
            np.sum([np.linalg.norm(core) ** 2 for core in merged_cores])
        )
        merged_cores = [core / norm for core in merged_cores]
        decomposed_cores: list[tuple[NDArray, NDArray, NDArray]] = [
            _twodot_svd(core) for core in merged_cores
        ]
        singular_values = np.concatenate([s for _, s, _ in decomposed_cores])
        # Descending order
        argsort = np.argsort(singular_values)[::-1]
        singular_values = singular_values[argsort]
        bond_dimension = guess_bond_dimension(singular_values, rate)
        if const.verbose > 2:
            logger.info(
                f"bond dimension {left_site}-{right_site} = {bond_dimension}"
            )
        adopted_indices = np.sort(argsort[:bond_dimension])
        indices_cumsum = np.cumsum(
            [0] + [s.size for _, s, _ in decomposed_cores]
        )
        adopted_indices_of_each_core: list[list[int]] = [
            [] for _ in range(nmpo)
        ]
        core_i = 0
        for index in adopted_indices:
            while core_i < nmpo:
                if indices_cumsum[core_i] <= index < indices_cumsum[core_i + 1]:
                    adopted_indices_of_each_core[core_i].append(
                        index - indices_cumsum[core_i]
                    )
                    break
                else:
                    core_i += 1
        bond_dimension_of_each_core: list[int] = [
            len(indices) for indices in adopted_indices_of_each_core
        ]
        for mpo, (U, s, Vh), bd in zip(
            mpos, decomposed_cores, bond_dimension_of_each_core, strict=True
        ):
            left_core = U[:, :bd] @ np.diag(np.sqrt(s)[:bd]) * np.sqrt(norm)
            right_core = np.diag(np.sqrt(s)[:bd]) @ Vh[:bd, :] * np.sqrt(norm)
            mpo[left_site] = left_core.reshape(*mpo[left_site].shape[:-1], bd)
            mpo[right_site] = right_core.reshape(bd, *mpo[right_site].shape[1:])
        if left_to_right:
            left_site += 1
            right_site += 1
        else:
            left_site -= 1
            right_site -= 1

    return mpos


def _compress_term_by_term(mpos: list[GridMPO], rate: float, nsweep: int = 1):
    for isweep in range(nsweep):
        if isweep % 2 == 0:
            mpos = _sweep_compress_term_by_term(mpos, rate, left_to_right=True)
        else:
            mpos = _sweep_compress_term_by_term(mpos, rate, left_to_right=False)

    mpo = _sum_of_mpos_to_single_mpo(mpos)
    # Execute SVD after allocating all cores
    mpo = sweep_compress_twodot(mpo, rate=rate, left_to_right=True)
    mpo = sweep_compress_twodot(mpo, rate=rate, left_to_right=False)
    # mpo = mpos[0]
    return mpo
