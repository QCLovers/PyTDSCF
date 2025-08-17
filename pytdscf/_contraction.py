"""
Tensor Contraction Module is separated from module to achieve acceleration
"""

from __future__ import annotations

import itertools
from functools import lru_cache
from time import time
from typing import Annotated, Any, Callable, overload

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as linalg
from opt_einsum import contract, contract_expression

import pytdscf
from pytdscf._const_cls import const
from pytdscf._mpo_cls import OperatorCore
from pytdscf._site_cls import SiteCoef

_op_keys = Annotated[
    str | tuple[int | tuple[int, int], ...],
    "str | tuple[int | tuple[int, int], ...]",
]

_block_type = Annotated[
    np.ndarray | jax.Array | int, "np.ndarray | jax.Array | int"
]

_core_type = Annotated[
    np.ndarray | jax.Array | int | OperatorCore,
    "np.ndarray | jax.Array | int | OperatorCore",
]


def get_expr(contraction: str, shapes: tuple[tuple[int, ...], ...]) -> Callable:
    if len(shapes) > 3:
        return contract_expression(contraction, *shapes)
    else:
        return _get_expr_cached(contraction, shapes)


@lru_cache(maxsize=256)  # up to 256 different contractions
def _get_expr_cached(
    contraction: str, shapes: tuple[tuple[int, ...], ...]
) -> Callable:
    return contract_expression(contraction, *shapes)


def is_unitmat_op(op_block_single: _block_type) -> bool:
    """Whether op_l or op_c or op_r is identity or not.
    Args:
        op_block_single (_block_type) : op_l, op_c, op_r
    Returns:
        bool : Whether op_l or op_c or op_r is identity or not
    """
    return isinstance(op_block_single, int)


def contract_with_site_concat(mat_bra, mat_ket, op_LorR_concat, op_site_concat):
    if mat_bra.gauge == "A":
        contraction = "xmn,mri,xrs,nsj->xij"
    elif mat_bra.gauge == "B":
        contraction = "xmn,irm,xrs,jsn->xij"
    else:
        raise AssertionError(
            f"mat_bra.gauge is neither A nor B, but {mat_bra.gauge}"
        )
    if const.use_jax:
        coef_bra = jnp.conj(mat_bra.data)
        coef_ket = mat_ket.data
        op_next_concat = jnp.einsum(
            contraction, op_LorR_concat, coef_bra, op_site_concat, coef_ket
        )
    else:
        coef_bra = np.conj(mat_bra)
        coef_ket = np.array(mat_ket)
        op_next_concat = contract(
            contraction, op_LorR_concat, coef_bra, op_site_concat, coef_ket
        )
    return op_next_concat


# @profile
def contract_with_site(
    mat_bra: SiteCoef,
    mat_ket: SiteCoef,
    op_LorR: _block_type,
    op_site: _block_type,
) -> np.ndarray | jax.Array:
    r"""Contraction between p-site bra, p-site ket, p-site operator and side-block

    Args:
        mat_bra (SiteCoef): :math:`\left(L^{\tau_{p-1}^\prime\tau_{p}^\prime}_{j_p^\prime}\right)^\ast` or similar R
        mat_ket (SiteCoef): :math:`L_{\tau_{p-1}\tau_{p}}^{j_p}` or similar R
        op_LorR (_block_type) : pre-calculated block operator :math:`E_{\tau_{p-1}}^{\tau_{p-1}^\prime}` \
            or similar p+1 index
        op_site (_block_type) : p-site operator :math:`O_{j_p}^{j_p^\prime}`

    Returns:
        np.ndarray | jax.Array : contracted system block :math:`E_{\tau_{p}}^{\tau_{p}^\prime}`
    """
    """
    op_next = np.einsum('nri,nrj->ij',\
              np.einsum('mn,mri->nri',op_LorR,coef_bra),\
              np.einsum('rs,nsj->nrj',op_site,coef_ket))
    """
    if mat_bra.gauge == "A":
        contraction = "mri,nsj,rs,mn->ij"
    elif mat_bra.gauge == "B":
        contraction = "irm,jsn,rs,mn->ij"
    else:
        raise AssertionError(
            f"mat_bra.gauge is neither A nor B, but {mat_bra.gauge}"
        )
    coef_bra: np.ndarray | jax.Array
    coef_ket: np.ndarray | jax.Array
    if const.use_jax:
        coef_bra = jnp.conj(mat_bra.data)
        coef_ket = mat_ket.data
    else:
        coef_bra = np.conj(mat_bra)
        coef_ket = np.array(mat_ket)
    operator = [coef_bra, coef_ket]
    if isinstance(op_site, int):
        contraction = contraction.replace(",rs", "")
        contraction = contraction.replace("r", "s")
    else:
        operator.append(op_site)
    if isinstance(op_LorR, int):
        contraction = contraction.replace(",mn", "")
        contraction = contraction.replace("m", "n")
    else:
        operator.append(op_LorR)

    if const.use_jax:
        op_next = jnp.einsum(contraction, *operator)
    else:
        expr = get_expr(contraction, tuple(op.shape for op in operator))
        op_next = expr(*operator)
        # op_next = contract(contraction, *operator)
    return op_next


def contract_with_site_mpo(
    mat_bra: SiteCoef,
    mat_ket: SiteCoef,
    op_LorR: _block_type,
    op_site: _core_type,
) -> np.ndarray | jax.Array:
    r"""Contraction between p-site bra, p-site ket, p-site operator and side-block

    Args:
        mat_bra (SiteCoef): :math:`\left(L^{\tau_{p-1}^\prime\tau_{p}^\prime}_{j_p^\prime}\right)^\ast` or similar R
        mat_ket (SiteCoef): :math:`L_{\tau_{p-1}\tau_{p}}^{j_p}` or similar R
        op_LorR (_block_type) : pre-calculated block operator \
            :math:`[O^{[:p-1]}_{\rm sys}]\substack{\tau_{p-1}^\prime \\ \beta_{p-1} \\ \tau_{p-1}}` \
            or similar p+1 index
        op_site (_block_type) : p-site operator
            :math:`W\substack{j_p^\prime \\ \beta_{p-1}\beta_{p}\\ j_p}`

    Returns:
        np.ndarray | jax.Array : contracted system block \
            :math:`[O^{[:p]}_{\rm sys}]\substack{\tau_{p}^\prime \\ \beta_{p} \\ \tau_{p}}`

    Ifs right canonical formulation,

    .. math::
       [O^{[p:]}_{\rm sys}]\substack{\tau_{p-1}^\prime \\ \beta_{p-1} \\ \tau_{p-1}} = \
       \sum_{j_p,j_p^\prime}\sum_{\tau_p,\tau_p^\prime}\sum_{\beta_p} \
       R^{\prime j_p^\prime}_{\tau_{p-1}^\prime\tau_p^\prime}R^{j_p}_{\tau_{p-1}\tau_{p}} \
       W\substack{j_p^\prime \\ \beta_{p-1}\beta_{p}\\ j_p} \
       [O^{[p+1:]}_{\rm sys}]\substack{\tau_{p}^\prime \\ \beta_{p} \\ \tau_{p}}

    """
    # Status of op_site
    # 1. op_site is int -> means identity
    # 2. op_site is np.ndarray | jax.Array -> means one-site operator
    # 3. op_site is OperatorCore -> means W
    if isinstance(op_site, int):
        op_site_mode = 1
    elif isinstance(op_site, np.ndarray | jax.Array):
        assert len(op_site.shape) == 2, f"op_site.shape = {op_site.shape}"
        op_site_mode = 2
    elif isinstance(op_site, OperatorCore):
        if isinstance(op_site.data, int):
            op_site_mode = 1
        else:
            op_site_mode = 3
    else:
        raise AssertionError(f"Invalid op_site type: {type(op_site)}")

    if isinstance(op_LorR, int):
        op_LorR_mode = 1
    elif isinstance(op_LorR, np.ndarray | jax.Array):
        assert len(op_LorR.shape) == 3, f"op_LorR.shape = {op_LorR.shape}"
        op_LorR_mode = 2
    else:
        raise AssertionError(f"Invalid op_LorR type: {type(op_LorR)}")

    if const.use_jax:
        coef_bra: jax.Array | np.ndarray = jnp.conj(mat_bra.data)
        coef_ket: jax.Array | np.ndarray = mat_ket.data
    else:
        coef_bra = np.conj(mat_bra)
        coef_ket = np.array(mat_ket)

    operator: list[jax.Array] | list[np.ndarray] = [coef_bra, coef_ket]  # type: ignore
    match (mat_bra.gauge, op_site_mode, op_LorR_mode):
        case ("A", 1, 1):
            # n-|-i
            # | s
            # | s
            # n-|-j
            contraction = "nsi,nsj->ij"
        case ("A", 1, 2):
            # m m-|-i
            # |   s
            # p-  |
            # |   s
            # n n-|-j
            contraction = "msi,nsj,mpn->ipj"
            assert isinstance(op_LorR, np.ndarray | jax.Array)
            operator.append(op_LorR)  # type: ignore
        case ("A", 2, 1):
            # m-|-i
            # | r
            # | r
            # | |
            # | s
            # | s
            # m-|-j
            contraction = "mri,msj,rs->ij"
            assert isinstance(op_site, np.ndarray | jax.Array)
            operator.append(op_site)  # type: ignore
        case ("A", 2, 2):
            # m m-|-i
            # |   r
            # |   r
            # |-p |
            # |   s
            # |   s
            # n n-|-j
            contraction = "mri,nsj,mpn,rs->ipj"
            assert isinstance(op_LorR, np.ndarray | jax.Array)
            assert isinstance(op_site, np.ndarray | jax.Array)
            operator.extend([op_LorR, op_site])  # type: ignore
        case ("A", 3, 1):
            assert isinstance(op_site, OperatorCore)
            data = op_site.data
            assert isinstance(data, np.ndarray | jax.Array)
            assert data.shape[0] == 1, f"op_site.data.shape = {data.shape}"
            if op_site.only_diag:
                # m-|-i
                # | r
                # | r
                # | | \
                # | r 1--q
                # m-|-j
                contraction = "mri,mrj,rq->iqj"
                operator.append(data[0, :, :])  # type: ignore
            else:
                # m--|-i
                # |  r
                # |  r
                # |1-|-q
                # |  s
                # |  s
                # m--|-j
                contraction = "mri,msj,rsq->iqj"
                operator.append(data[0, :, :, :])  # type: ignore
        case ("A", 3, 2):
            assert isinstance(op_site, OperatorCore)
            data = op_site.data
            assert isinstance(data, np.ndarray | jax.Array)
            if op_site.only_diag:
                # m m-|-i
                # |   r
                # |   r
                # |   | \
                # p-----p--q
                # |   r
                # n n-|-j
                contraction = "mri,nrj,mpn,prq->iqj"
            else:
                # m m-|-i
                # |   r
                # |   r
                # p p-|-q
                # |   s
                # |   s
                # n n-|-j
                contraction = "mri,nsj,mpn,prsq->iqj"
            assert isinstance(op_LorR, np.ndarray | jax.Array)
            operator.extend([op_LorR, data])  # type: ignore
        case ("B", 1, 1):
            # i-|-n
            #   s |
            #   s |
            # j-|-n
            contraction = "isn,jsn->ij"
        case ("B", 1, 2):
            # i-|-m m
            #   s   |
            #   |  -q
            #   s   |
            # j-|-n n
            contraction = "ism,jsn,mqn->iqj"
            assert isinstance(op_LorR, np.ndarray | jax.Array)
            operator.append(op_LorR)  # type: ignore
        case ("B", 2, 1):
            # i-|-m
            #   r |
            #   r |
            #   | |
            #   s |
            #   s |
            # j-|-m
            contraction = "irm,jsm,rs->ij"
            assert isinstance(op_site, np.ndarray | jax.Array)
            operator.append(op_site)  # type: ignore
        case ("B", 2, 2):
            # i-|-m m
            #   r   |
            #   r   |
            #   | q-|
            #   s   |
            #   s   |
            # j-|-n n
            contraction = "irm,jsn,mqn,rs->iqj"
            assert isinstance(op_LorR, np.ndarray | jax.Array)
            assert isinstance(op_site, np.ndarray | jax.Array)
            operator.extend([op_LorR, op_site])  # type: ignore
        case ("B", 3, 1):
            assert isinstance(op_site, OperatorCore)
            data = op_site.data
            assert isinstance(data, np.ndarray | jax.Array)
            assert data.shape[-1] == 1, f"op_site.data.shape = {data.shape}"
            if op_site.only_diag:
                #   i-|-m
                #     r |
                #     r |
                #   / | |
                # p--1r |
                #   j-|-m
                contraction = "irm,jrm,pr->ipj"
                operator.append(data[:, :, 0])  # type: ignore
            else:
                # i-|--m
                #   r  |
                #   r  |
                # p-|-1|
                #   s  |
                #   s  |
                # j-|--m
                contraction = "irm,jsm,prs->ipj"
                operator.append(data[:, :, :, 0])  # type: ignore
        case ("B", 3, 2):
            assert isinstance(op_site, OperatorCore)
            data = op_site.data
            assert isinstance(data, np.ndarray | jax.Array), type(data)
            if op_site.only_diag:
                #   i-|-m m
                #     r   |
                #     r   |
                #    /|   |
                # p--q----q
                #     r   |
                #   j-|-n n
                contraction = "irm,jrn,mqn,prq->ipj"
            else:
                # i-|-m m
                #   r   |
                #   r   |
                # p-|-q q
                #   s   |
                #   s   |
                # j-|-n n
                contraction = "irm,jsn,mqn,prsq->ipj"
            assert isinstance(op_LorR, np.ndarray | jax.Array)
            operator.extend([op_LorR, data])  # type: ignore
        case _:
            raise ValueError(
                f"{mat_bra.gauge=}, {op_site_mode=}, {op_LorR_mode=}"
            )
    if const.use_jax:
        op_next = jnp.einsum(contraction, *operator)
    else:
        expr = get_expr(contraction, tuple(op.shape for op in operator))
        op_next = expr(*operator)
        # op_next = contract(contraction, *operator)
    if len(op_next.shape) == 2:
        op_next = op_next[:, None, :]
    return op_next


def mfop_site(matPsi_bra, matPsi_ket, op_left, op_right):
    """
    mfop_site = np.einsum('imr,inr->mn',np.conj(matPsi_bra),\
                np.einsum('ij,jnr->inr',op_left,\
                np.einsum('rs,jns->jnr',op_right,matPsi_ket)))
    """
    assert matPsi_bra.gauge == "Psi"
    assert matPsi_ket.gauge == "Psi"

    if isinstance(op_right, int):
        dum_jnr = matPsi_ket.data
    else:
        if const.use_jax:
            dum_jnr = jnp.einsum("rs,jns->jnr", op_right, matPsi_ket.data)
        else:
            dum_jnr = np.einsum("rs,jns->jnr", op_right, matPsi_ket)

    if isinstance(op_left, int):
        dum_inr = dum_jnr
    else:
        if const.use_jax:
            dum_inr = jnp.einsum("ij,jnr->inr", op_left, dum_jnr)
        else:
            dum_inr = np.einsum("ij,jnr->inr", op_left, dum_jnr)

    if const.use_jax:
        mfop_site = jnp.einsum(
            "imr,inr->mn", jnp.conj(matPsi_bra.data), dum_inr
        )
    else:
        mfop_site = np.einsum("imr,inr->mn", np.conj(matPsi_bra), dum_inr)
    return mfop_site


def mfop_site_concat(matPsi_bra, matPsi_ket, op_left_concat, op_right_concat):
    assert matPsi_bra.gauge == "Psi"
    assert matPsi_ket.gauge == "Psi"
    subscripts = "imr,kij,krs,jns->mn"
    if const.use_jax:
        mfop_site_concat = jnp.einsum(
            subscripts,
            jnp.conj(matPsi_bra.data),
            op_left_concat,
            op_right_concat,
            matPsi_ket.data,
        )
    else:
        mfop_site_concat = contract(
            subscripts,
            np.conj(matPsi_bra),
            op_left_concat,
            op_right_concat,
            matPsi_ket,
        )
    return mfop_site_concat


class SplitStack:
    def __init__(
        self,
        psi_state_shape_in: tuple[int, ...],
        psi_state_shape_out: tuple[int, ...] | None = None,
        nstate: int = 1,
    ):
        self.nstate: int = nstate
        self.tensor_shapes_in: tuple[int, ...] = psi_state_shape_in
        self._split_idx_in: list[int] = np.cumsum(
            [np.prod(self.tensor_shapes_in)] * nstate
        ).tolist()[:-1]
        if psi_state_shape_out is None:
            self.tensor_shapes_out: tuple[int, ...] = psi_state_shape_in
            self._split_idx_out = self._split_idx_in
        else:
            self.tensor_shapes_out = psi_state_shape_out
            self._split_idx_out = np.cumsum(
                [np.prod(self.tensor_shapes_out)] * nstate
            ).tolist()[:-1]
        self.in_same_as_out = self.tensor_shapes_in == self.tensor_shapes_out

    def stack(
        self,
        psi_states: list[np.ndarray] | list[jax.Array],
        extend: bool = False,
    ) -> np.ndarray | jax.Array:
        """stack MPS p-site coef for each electronic states 1 dimensional

        [(n,m,n) shape array,(n,m,n) shape array] --> (1,2n^2m) shape array

        Args:
            psi_states (list[np.ndarray]): psi_states[i] denotes i-electronic states p-site coefficient (3-rank tensor)

        Returns:
            np.ndarray: flattened list. psi[i] denotes i-electronic states

        """
        if const.use_jax:
            if extend and not self.in_same_as_out:
                raise NotImplementedError("extend is not implemented")
                # TODO: implement extend with JIT
            if len(psi_states) > 1:
                return _stack(psi_states)
            else:
                return jnp.reshape(psi_states[0], -1)
        else:
            if extend and not self.in_same_as_out:
                if len(psi_states[0].shape) == 3:
                    l, c, r = psi_states[0].shape  # noqa: E741
                    L, C, R = self.tensor_shapes_out
                    assert L >= l and C >= c and R >= r, (
                        f"{L=}, {C=}, {R=}, {l=}, {c=}, {r=}"
                    )
                    psi_states = [
                        np.pad(
                            x,
                            (
                                (0, L - l),
                                (0, C - c),
                                (0, R - r),
                            ),
                        )
                        for i, x in enumerate(psi_states)
                    ]
                elif len(psi_states[0].shape) == 2:
                    l, r = psi_states[0].shape  # noqa: E741
                    L, R = self.tensor_shapes_out
                    assert L >= l and R >= r, f"{L=}, {R=}, {l=}, {r=}"
                    psi_states = [
                        np.pad(x, ((0, L - l), (0, R - r))) for x in psi_states
                    ]
            # psi = np.hstack([x.flatten() for x in psi_states])
            psi = np.hstack([x.ravel() for x in psi_states])
        return psi

    def split(
        self, psi: np.ndarray | jax.Array, truncate: bool = False
    ) -> list[np.ndarray] | list[jax.Array]:
        """split MPS p-site coef for each electronic states from 1 dimensional to 3-rank tensor

        Correspond to reverse stack

        (1,2n^2m) shape array --> [(n,m,n) shape array,(n,m,n) shape array]

        Args:
            psi (np.ndarray | jax.Array): flattened psi_states

        Returns:
            list[np.ndarray] | list[jax.Array] : 3-rank tensor list. psi_states[i] denotes i-electronic states

        """

        psi_states: list[np.ndarray] | list[jax.Array]
        if const.use_jax:
            # Splitting is difficult to be jit-compiled because of the variable length
            if len(self._split_idx_out) == 0:
                psi_states = [jnp.reshape(psi, self.tensor_shapes_out)]
            else:
                psi_states = [
                    jnp.reshape(x, self.tensor_shapes_out)
                    for x in jnp.split(psi, self._split_idx_out)
                ]
        else:
            if len(self._split_idx_out) == 0:
                psi_states = [psi.reshape(self.tensor_shapes_out)]  # type: ignore
            else:
                psi_states = [
                    x.reshape(self.tensor_shapes_out)
                    for x in np.split(psi, self._split_idx_out)
                ]
        if truncate and not self.in_same_as_out:
            if len(self.tensor_shapes_in) == 3:
                psi_states = [
                    x[
                        : self.tensor_shapes_in[0],
                        : self.tensor_shapes_in[1],
                        : self.tensor_shapes_in[2],
                    ]
                    for x in psi_states
                ]  # type: ignore
            else:
                psi_states = [
                    x[: self.tensor_shapes_in[0], : self.tensor_shapes_in[1]]
                    for x in psi_states
                ]  # type: ignore
        return psi_states


class multiplyH_MPS_direct(SplitStack):
    r"""Operator and LCR MPS Multiplication Class

    Args:
        op_lcr_states (list[list[dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]]) : \
            [state_bra][state_ket][op_name] = (op_l, op_c, op_r)
        psi_states (np.ndarray) : __description__
    """

    def __init__(
        self,
        op_lcr_states: list[
            list[
                dict[
                    _op_keys,
                    tuple[
                        _block_type,
                        _block_type,
                        _block_type,
                    ],
                ]
            ]
        ],
        psi_states: list[np.ndarray] | list[jax.Array],
        hamiltonian: pytdscf.hamiltonian_cls.HamiltonianMixin | None = None,
        tensor_shapes_out: tuple[int, ...] | None = None,
    ):
        self.matH_cas = hamiltonian
        self.op_lcr_states = op_lcr_states
        super().__init__(
            psi_states[0].shape, tensor_shapes_out, len(psi_states)
        )

    # @profile
    def _op_lcr_dot(self, _op_l, _op_c, _op_r, _trial):
        """
        return np.einsum('rs,ais->air',_op_r,\
               np.einsum('ab,bis->ais',_op_l,\
               np.einsum('ij,bjs->bis',_op_c, _trial)))
        """
        """
        -a       r-
        |    i    |
        |    |    |
        |    j    |
        |    j    |
        -b b-|-s s-
        """

        if const.use_jax:
            contraction = "bjs,ab,ij,rs->air"
            operator = [_trial]
            if isinstance(_op_l, int):
                contraction = contraction.replace(",ab", "")
                contraction = contraction.replace("a", "b")
            else:
                operator.append(_op_l)
            if isinstance(_op_c, int):
                contraction = contraction.replace(",ij", "")
                contraction = contraction.replace("i", "j")
            else:
                operator.append(_op_c)
            if isinstance(_op_r, int):
                contraction = contraction.replace(",rs", "")
                contraction = contraction.replace("r", "s")
            else:
                operator.append(_op_r)
            return jnp.einsum(contraction, *operator)
        else:
            # Numpy can be accelerated by BLAS ZGEMM due to the hermitian property of the operator
            if isinstance(_op_l, int):
                dum_l_cr = _trial.reshape(_trial.shape[0], -1)
                sig_shape_0 = _trial.shape[0]
            else:
                dum_l_cr = linalg.blas.zgemm(
                    alpha=1.0, a=_op_l, b=_trial.reshape(_op_l.shape[1], -1)
                )
                sig_shape_0 = _op_l.shape[0]

            if isinstance(_op_c, int):
                dum_c_rl = dum_l_cr.T.reshape(_trial.shape[1], -1)
                sig_shape_1 = _trial.shape[1]
            else:
                dum_c_rl = linalg.blas.zgemm(
                    alpha=1.0, a=_op_c, b=dum_l_cr.T.reshape(_op_c.shape[1], -1)
                )
                sig_shape_1 = _op_c.shape[0]

            if isinstance(_op_r, int):
                dum_r_lc = dum_c_rl.T.reshape(_trial.shape[2], -1)
                sig_shape_2 = _trial.shape[2]
            else:
                dum_r_lc = linalg.blas.zgemm(
                    alpha=1.0, a=_op_r, b=dum_c_rl.T.reshape(_op_r.shape[1], -1)
                )
                sig_shape_2 = _op_r.shape[0]
            return dum_r_lc.T.reshape(sig_shape_0, sig_shape_1, sig_shape_2)

    # @profile
    def dot(self, trial_states):
        """Only supported sum op products form"""
        sigvec_states = [None for _ in trial_states]
        assert isinstance(
            self.matH_cas, pytdscf.hamiltonian_cls.PolynomialHamiltonian
        )
        for i, j in itertools.product(
            range(len(self.matH_cas.coupleJ)), repeat=2
        ):
            if const.use_jax:
                sigvecs_add = []
            op_lcr = self.op_lcr_states[i][j]
            if op_lcr is None:
                continue
            op_l_ovlp, op_c_ovlp, op_r_ovlp = op_lcr["ovlp"]

            if (coupleJ := self.matH_cas.coupleJ[i][j]) != 0.0:
                sigvec_add = (
                    self._op_lcr_dot(
                        op_l_ovlp, op_c_ovlp, op_r_ovlp, trial_states[j]
                    )
                    * coupleJ
                )
                if const.use_jax:
                    sigvecs_add.append(sigvec_add)
                else:
                    if sigvec_states[i] is None:
                        sigvec_states[i] = sigvec_add
                    else:
                        sigvec_states[i] += sigvec_add
            if self.matH_cas.onesite:
                op_l_onesite, op_c_onesite, op_r_onesite = op_lcr["onesite"]
                sigvec_add1 = self._op_lcr_dot(
                    op_l_onesite, op_c_ovlp, op_r_ovlp, trial_states[j]
                )
                sigvec_add2 = self._op_lcr_dot(
                    op_l_ovlp, op_c_onesite, op_r_ovlp, trial_states[j]
                )
                sigvec_add3 = self._op_lcr_dot(
                    op_l_ovlp, op_c_ovlp, op_r_onesite, trial_states[j]
                )
                if const.use_jax:
                    sigvecs_add.extend([sigvec_add1, sigvec_add2, sigvec_add3])
                else:
                    if sigvec_states[i] is None:
                        sigvec_states[i] = (
                            sigvec_add1 + sigvec_add2 + sigvec_add3
                        )
                    else:
                        sigvec_states[i] += (
                            sigvec_add1 + sigvec_add2 + sigvec_add3
                        )
            if i == j and self.matH_cas.general:
                if "enable_summed_op" in const.keys:
                    op_l_general_sum, op_c_ovlp, op_r_ovlp = op_lcr[
                        "general_summ_l"
                    ]
                    op_l_ovlp, op_c_ovlp, op_r_general_sum = op_lcr[
                        "general_summ_r"
                    ]
                    sigvec_add1 = self._op_lcr_dot(
                        op_l_general_sum, op_c_ovlp, op_r_ovlp, trial_states[j]
                    )
                    sigvec_add2 = self._op_lcr_dot(
                        op_l_ovlp, op_c_ovlp, op_r_general_sum, trial_states[j]
                    )
                    if const.use_jax:
                        sigvecs_add.extend([sigvec_add1, sigvec_add2])
                    else:
                        if sigvec_states[i] is None:
                            sigvec_states[i] = sigvec_add1 + sigvec_add2
                        else:
                            sigvec_states[i] += sigvec_add1 + sigvec_add2
                (
                    op_l_general_concat,
                    op_c_general_concat,
                    op_r_general_concat,
                ) = op_lcr["general_concat"]
                subscript = "xrs,xij,xab,bjs-> air"
                if const.use_jax:
                    sigvec_add = jnp.einsum(
                        subscript,
                        op_r_general_concat,
                        op_c_general_concat,
                        op_l_general_concat,
                        trial_states[j],
                    )
                else:
                    sigvec_add = contract(
                        subscript,
                        op_r_general_concat,
                        op_c_general_concat,
                        op_l_general_concat,
                        trial_states[j],
                    )
                if const.use_jax:
                    sigvecs_add.append(sigvec_add)
                else:
                    if sigvec_states[i] is None:
                        sigvec_states[i] = sigvec_add
                    else:
                        sigvec_states[i] += sigvec_add
            if const.use_jax:
                sigvec_states[i] = add_to_sigvec_states(
                    sigvec_states[i], sigvecs_add
                )
        return sigvec_states

    def dot_autocorr(self, trial_states):
        if const.use_jax:
            sigvec_states = get_zeros_sigvec_states(trial_states)
        else:
            sigvec_states = [
                np.zeros_like(trial, dtype=np.complex128)
                for trial in trial_states
            ]
        for bra_states, ket_states in itertools.product(
            range(len(self.op_lcr_states)), repeat=2
        ):
            op_lcr = self.op_lcr_states[bra_states][ket_states]
            if op_lcr is not None:
                assert bra_states == ket_states
                op_l_auto, op_c_auto, op_r_auto = op_lcr["auto"]
                sigvec_states[bra_states] += self._op_lcr_dot(
                    op_l_auto, op_c_auto, op_r_auto, trial_states[ket_states]
                )
        return sigvec_states


class multiplyK_MPS_direct(SplitStack):
    r"""Operator and LsR MPS Multiplication Class

    Args:
        op_lr_states (list[list[dict[str, Tuple[np.ndarray, np.ndarray]]]]) : \
            [state_bra][state_ket][op_name] = (op_l, op_c, op_r)
        matH_cas (HamiltonianMixin) : Hamiltonian
        psi_states (np.ndarray) : __description__
    """

    def __init__(
        self,
        op_lr_states: list[
            list[
                dict[
                    _op_keys,
                    tuple[
                        _block_type,
                        _block_type,
                    ],
                ]
            ]
        ],
        psi_states: list[np.ndarray] | list[jax.Array],
        hamiltonian: pytdscf.hamiltonian_cls.HamiltonianMixin,
        tensor_shapes_out: tuple[int, ...] | None = None,
    ):
        self.matH_cas = hamiltonian
        self.op_lr_states = op_lr_states
        super().__init__(
            psi_states[0].shape, tensor_shapes_out, len(psi_states)
        )

    def _op_lr_dot(self, _op_l, _op_r, _trial):
        match (isinstance(_op_l, int), isinstance(_op_r, int)):
            case (True, True):
                return _trial
            case (True, False):
                contraction = "rs,as->ar"
                operator = [_op_r, _trial]
            case (False, True):
                contraction = "ab,bs->as"
                operator = [_op_l, _trial]
            case (False, False):
                contraction = "ab,bs,rs->ar"
                operator = [_op_l, _trial, _op_r]
        if const.use_jax:
            return jnp.einsum(contraction, *operator)
        else:
            return contract(contraction, *operator)

    # @profile
    def dot(self, trial_states):
        """Only supported sum op products form"""
        sigvec_states = [None for _ in trial_states]
        for i, j in itertools.product(
            range(len(self.matH_cas.coupleJ)), repeat=2
        ):
            if const.use_jax:
                sigvecs_add = []
            op_lr = self.op_lr_states[i][j]
            if op_lr is None:
                continue
            coupleJ = self.matH_cas.coupleJ[i][j]
            op_l_ovlp, op_r_ovlp = op_lr["ovlp"]

            if coupleJ != 0.0:
                sigvec_add = (
                    self._op_lr_dot(op_l_ovlp, op_r_ovlp, trial_states[j])
                    * coupleJ
                )
                if const.use_jax:
                    sigvecs_add.append(sigvec_add)
                else:
                    if sigvec_states[i] is None:
                        sigvec_states[i] = sigvec_add
                    else:
                        sigvec_states[i] += sigvec_add

            if self.matH_cas.onesite:
                op_l_onesite, op_r_onesite = op_lr["onesite"]
                sigvec_add1 = self._op_lr_dot(
                    op_l_onesite, op_r_ovlp, trial_states[j]
                )
                sigvec_add2 = self._op_lr_dot(
                    op_l_ovlp, op_r_onesite, trial_states[j]
                )
                if const.use_jax:
                    sigvecs_add.extend([sigvec_add1, sigvec_add2])
                else:
                    if sigvec_states[i] is None:
                        sigvec_states[i] = sigvec_add1 + sigvec_add2
                    else:
                        sigvec_states[i] += sigvec_add1 + sigvec_add2

            if i == j and self.matH_cas.general:
                if "enable_summed_op" in const.keys:
                    op_l_general_sum, op_r_ovlp = op_lr["general_summ_l"]
                    op_l_ovlp, op_r_general_sum = op_lr["general_summ_r"]
                    sigvec_add1 = self._op_lr_dot(
                        op_l_general_sum, op_r_ovlp, trial_states[j]
                    )
                    sigvec_add2 = self._op_lr_dot(
                        op_l_ovlp, op_r_general_sum, trial_states[j]
                    )
                    if const.use_jax:
                        sigvecs_add.extend([sigvec_add1, sigvec_add2])
                    else:
                        if sigvec_states[i] is None:
                            sigvec_states[i] = sigvec_add1 + sigvec_add2
                        else:
                            sigvec_states[i] += sigvec_add1 + sigvec_add2

                op_l_general_concat, op_r_general_concat = op_lr[
                    "general_concat"
                ]
                subscript = "xrs,xab,bs->ar"
                if const.use_jax:
                    sigvec_add = jnp.einsum(
                        subscript,
                        op_r_general_concat,
                        op_l_general_concat,
                        trial_states[j],
                    )
                else:
                    sigvec_add = contract(
                        subscript,
                        op_r_general_concat,
                        op_l_general_concat,
                        trial_states[j],
                    )
                if const.use_jax:
                    sigvecs_add.append(sigvec_add)
                else:
                    if sigvec_states[i] is None:
                        sigvec_states[i] = sigvec_add
                    else:
                        sigvec_states[i] += sigvec_add
            if const.use_jax:
                sigvec_states[i] = add_to_sigvec_states(
                    sigvec_states[i], sigvecs_add
                )
        return sigvec_states


class multiplyH_MPS_direct_MPO(multiplyH_MPS_direct):
    r"""Operator and LCR MPS Multiplication Class

    Args:
        op_lcr_states (list[list[dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]]) : \
            [state_bra][state_ket][op_name] = (op_l, op_c, op_r)
        psi_states (np.ndarray) : __description__
    """

    def __init__(
        self,
        op_lcr_states: list[
            list[
                dict[
                    _op_keys,
                    tuple[
                        _block_type,
                        _block_type,
                        _block_type,
                    ],
                ]
            ]
        ],
        psi_states: list[np.ndarray] | list[jax.Array],
        hamiltonian: pytdscf.hamiltonian_cls.TensorHamiltonian,
        tensor_shapes_out: tuple[int, ...] | None = None,
    ):
        super().__init__(
            op_lcr_states,
            psi_states,
            hamiltonian=hamiltonian,
            tensor_shapes_out=tensor_shapes_out,
        )
        self._op_lcr_dot_cached = {}

    # @profile
    @overload
    def _op_lcr_dot(
        self,
        _op_l: int | np.ndarray,
        _op_c: int | np.ndarray,
        _op_r: int | np.ndarray,
        _trial: np.ndarray,
        key: Any | None = None,
    ) -> np.ndarray: ...
    @overload
    def _op_lcr_dot(
        self,
        _op_l: int | jax.Array,
        _op_c: int | jax.Array,
        _op_r: int | jax.Array,
        _trial: jax.Array,
        key: Any | None = None,
    ) -> jax.Array: ...
    def _op_lcr_dot(self, _op_l, _op_c, _op_r, _trial, key: Any | None = None):
        """Operator and LCR MPS Multiplication"""
        """
        Tensor contraction diagram:
        | |-a     r-| |
        | |    i    | |
        | |    |    | |
        |L|-c-|C|-t-|R|
        | |    |    | |
        | |    j    | |
        | |-b-|T|-s-| |
        """
        if (
            not const.use_jax
            and key is not None
            and key in self._op_lcr_dot_cached
        ):
            return self._op_lcr_dot_cached[key](_trial)
        operator = [_trial]
        if isinstance(_op_l, int):
            op_l_mode = 1
        elif len(_op_l.shape) == 2:
            op_l_mode = 2
        elif len(_op_l.shape) == 3:
            op_l_mode = 3
        else:
            raise ValueError(f"Invalid operator shape: {_op_l.shape}")
        if isinstance(_op_c, int):
            op_c_mode = 1
        elif len(_op_c.shape) == 2:
            op_c_mode = 2
        elif isinstance(_op_c.data, int):
            op_c_mode = 1
        elif _op_c.only_diag:
            op_c_mode = 3
        else:
            op_c_mode = 4
        if isinstance(_op_r, int):
            op_r_mode = 1
        elif len(_op_r.shape) == 2:
            op_r_mode = 2
        elif len(_op_r.shape) == 3:
            op_r_mode = 3
        else:
            raise ValueError(f"Invalid operator shape: {_op_r.shape}")
        if op_l_mode > 1:
            operator.append(_op_l)
        if op_c_mode > 1:
            operator.append(_op_c.data)
        if op_r_mode > 1:
            operator.append(_op_r)
        match (op_l_mode, op_c_mode, op_r_mode):
            case (1, 1, 1):
                return _trial
            case (1, 1, 2):
                contraction = "bjs,rs->bjr"
            case (1, 1, 3):
                contraction = "bjs,rts->bjr"
            case (1, 2, 1):
                contraction = "bjs,ij->bis"
            case (1, 2, 2):
                contraction = "bjs,ij,rs->bir"
            case (1, 2, 3):
                contraction = "bjs,ij,rts->bir"
            case (1, 3, 1):
                contraction = "bjs,cjt->bjs"
            case (1, 3, 2):
                contraction = "bjs,cjt,rs->bjr"
            case (1, 3, 3):
                contraction = "bjs,cjt,rts->bjr"
            case (1, 4, 1):
                contraction = "bjs,cijt->bis"
            case (1, 4, 2):
                contraction = "bjs,cijt,rs->bir"
            case (1, 4, 3):
                contraction = "bjs,cijt,rts->bir"
            case (2, 1, 1):
                contraction = "bjs,ab->ajs"
            case (2, 1, 2):
                contraction = "bjs,ab,rs->ajr"
            case (2, 1, 3):
                contraction = "bjs,ab,rts->ajr"
            case (2, 2, 1):
                contraction = "bjs,ab,ij->ais"
            case (2, 2, 2):
                contraction = "bjs,ab,ij,rs->air"
            case (2, 2, 3):
                contraction = "bjs,ab,ij,rts->air"
            case (2, 3, 1):
                contraction = "bjs,ab,cjt->ajs"
            case (2, 3, 2):
                contraction = "bjs,ab,cjt,rs->ajr"
            case (2, 3, 3):
                contraction = "bjs,ab,cjt,rts->ajr"
            case (2, 4, 1):
                contraction = "bjs,ab,cijt->ais"
            case (2, 4, 2):
                contraction = "bjs,ab,cijt,rs->air"
            case (2, 4, 3):
                contraction = "bjs,ab,cijt,rts->air"
            case (3, 1, 1):
                contraction = "bjs,acb->ajs"
            case (3, 1, 2):
                contraction = "bjs,acb,rs->ajr"
            case (3, 1, 3):
                contraction = "bjs,acb,rts->ajr"
            case (3, 2, 1):
                contraction = "bjs,acb,ij->ais"
            case (3, 2, 2):
                contraction = "bjs,acb,ij,rs->air"
            case (3, 2, 3):
                contraction = "bjs,acb,ij,rts->air"
            case (3, 3, 1):
                contraction = "bjs,acb,cjt->ajs"
            case (3, 3, 2):
                contraction = "bjs,acb,cjt,rs->ajr"
            case (3, 3, 3):
                contraction = "bjs,acb,cjt,rts->ajr"
            case (3, 4, 1):
                contraction = "bjs,acb,cijt->ais"
            case (3, 4, 2):
                contraction = "bjs,acb,cijt,rs->air"
            case (3, 4, 3):
                contraction = "bjs,acb,cijt,rts->air"
        if const.use_jax:
            sig_lcr = jnp.einsum(contraction, *operator)
        else:
            if key is not None:
                expr = contract_expression(
                    contraction,
                    _trial.shape,
                    *operator[1:],
                    constants=list(range(1, len(operator))),
                )
                self._op_lcr_dot_cached[key] = expr
                return expr(_trial)
            sig_lcr = contract(contraction, *operator)
            # Numpy might be accelerated by BLAS ZGEMM due to the hermitian property of the operator
        return sig_lcr

    @overload
    def dot(self, trial_states: list[np.ndarray]) -> list[np.ndarray]: ...
    @overload
    def dot(self, trial_states: list[jax.Array]) -> list[jax.Array]: ...
    def dot(
        self, trial_states: list[np.ndarray] | list[jax.Array]
    ) -> list[np.ndarray] | list[jax.Array]:
        """Only supported MPO"""
        sigvec_states: list[np.ndarray] | list[jax.Array]
        sigvec_states = [None for _ in trial_states]  # type: ignore
        assert isinstance(
            self.matH_cas, pytdscf.hamiltonian_cls.TensorHamiltonian
        )
        for i, j in itertools.product(
            range(len(self.matH_cas.coupleJ)), repeat=2
        ):
            if const.use_jax:
                sigvecs_add = []
            op_lcr = self.op_lcr_states[i][j]
            if op_lcr is None:
                continue

            if (coupleJ := self.matH_cas.coupleJ[i][j]) != 0.0:
                sigvec_add = (
                    self._op_lcr_dot(
                        *op_lcr["ovlp"],  # type: ignore
                        trial_states[j],  # type: ignore
                        key="ovlp",
                    )
                    * coupleJ
                )
                if const.use_jax:
                    sigvecs_add.append(sigvec_add)
                else:
                    if (sigvec_states_i := sigvec_states[i]) is None:
                        sigvec_states[i] = sigvec_add
                    else:
                        assert isinstance(sigvec_states_i, np.ndarray)
                        sigvec_states_i += sigvec_add

            for key, (op_l, op_c, op_r) in op_lcr.items():
                if key == "ovlp":
                    # Already calculated in the above with coupleJ.
                    continue
                sigvec_add = self._op_lcr_dot(
                    op_l,  # type: ignore
                    op_c,  # type: ignore
                    op_r,  # type: ignore
                    trial_states[j],  # type: ignore
                    key=key,
                )
                if const.use_jax:
                    sigvecs_add.append(sigvec_add)
                else:
                    if (sigvec_states_i := sigvec_states[i]) is None:
                        sigvec_states[i] = sigvec_add
                    else:
                        assert isinstance(sigvec_states_i, np.ndarray)
                        sigvec_states_i += sigvec_add
            if const.use_jax:
                sigvec_states[i] = add_to_sigvec_states(
                    sigvec_states[i], sigvecs_add
                )
        if const.verbose == 4:
            pytdscf._helper._ElpTime.dot += time()
        return sigvec_states


class multiplyK_MPS_direct_MPO(multiplyK_MPS_direct):
    r"""Operator and LsR MPS Multiplication Class

    Args:
        op_lr_states (list[list[dict[str, Tuple[np.ndarray, np.ndarray]]]]) : \
            [state_bra][state_ket][op_name] = (op_l, op_c, op_r)
        matH_cas : Hamiltonian
        psi_states (np.ndarray) : __description__
    """

    def __init__(
        self,
        op_lr_states: list[
            list[
                dict[
                    _op_keys,
                    tuple[
                        _block_type,
                        _block_type,
                    ],
                ]
            ]
        ],
        psi_states: list[np.ndarray] | list[jax.Array],
        hamiltonian: pytdscf.hamiltonian_cls.TensorHamiltonian,
        tensor_shapes_out: tuple[int, ...] | None = None,
    ):
        super().__init__(
            op_lr_states=op_lr_states,
            psi_states=psi_states,
            hamiltonian=hamiltonian,
            tensor_shapes_out=tensor_shapes_out,
        )
        self._op_lr_dot_cached = {}

    @overload
    def _op_lr_dot(
        self,
        _op_l: int | np.ndarray,
        _op_r: int | np.ndarray,
        _trial: np.ndarray,
        key: Any | None = None,
    ) -> np.ndarray: ...
    @overload
    def _op_lr_dot(
        self,
        _op_l: int | jax.Array,
        _op_r: int | jax.Array,
        _trial: jax.Array,
        key: Any | None = None,
    ) -> jax.Array: ...
    def _op_lr_dot(self, _op_l, _op_r, _trial, key: Any | None = None):
        """

        -a     r-
        |       |
        -c --- c-
        |       |
        -b b-s s-

        """
        if (
            not const.use_jax
            and key is not None
            and key in self._op_lr_dot_cached
        ):
            return self._op_lr_dot_cached[key](_trial)

        match (isinstance(_op_l, int), isinstance(_op_r, int)):
            case (True, True):
                return _trial
            case (True, False):
                if len(_op_r.shape) == 2:
                    contraction = "as,rs->ar"
                else:
                    contraction = "as,rcs->ar"
                operator = [_trial, _op_r]
            case (False, True):
                if len(_op_l.shape) == 2:
                    contraction = "br,ab->ar"
                else:
                    contraction = "br,acb->ar"
                operator = [_trial, _op_l]
            case (False, False):
                operator = [_trial, _op_l, _op_r]
                match (len(_op_l.shape) == 2, len(_op_r.shape) == 2):
                    case (True, True):
                        contraction = "bs,ab,rs->ar"
                    case (True, False):
                        contraction = "bs,ab,rcs->ar"
                    case (False, True):
                        contraction = "bs,acb,rs->ar"
                    case (False, False):
                        contraction = "bs,acb,rcs->ar"
        if const.use_jax:
            return jnp.einsum(contraction, *operator)
        else:
            if key is not None:
                expr = contract_expression(
                    contraction,
                    _trial.shape,
                    *operator[1:],
                    constants=list(range(1, len(operator))),
                )
                self._op_lr_dot_cached[key] = expr
                return expr(_trial)
            return contract(contraction, *operator)

    @overload
    def dot(self, trial_states: list[np.ndarray]) -> list[np.ndarray]: ...
    @overload
    def dot(self, trial_states: list[jax.Array]) -> list[jax.Array]: ...
    def dot(
        self, trial_states: list[np.ndarray] | list[jax.Array]
    ) -> list[np.ndarray] | list[jax.Array]:
        sigvec_states: list[np.ndarray] | list[jax.Array]
        sigvec_states = [None for _ in trial_states]  # type: ignore
        for i, j in itertools.product(
            range(len(self.matH_cas.coupleJ)), repeat=2
        ):
            if const.use_jax:
                sigvecs_add = []
            op_lr = self.op_lr_states[i][j]
            if op_lr is None:
                continue
            if (coupleJ := self.matH_cas.coupleJ[i][j]) != 0.0:
                sigvec_add = (
                    self._op_lr_dot(*op_lr["ovlp"], trial_states[j], key="ovlp")  # type: ignore
                    * coupleJ
                )
                if const.use_jax:
                    sigvecs_add.append(sigvec_add)
                else:
                    if (sigvec_states_i := sigvec_states[i]) is None:
                        sigvec_states[i] = sigvec_add
                    else:
                        assert isinstance(sigvec_states_i, np.ndarray)
                        sigvec_states_i += sigvec_add
            for key, (op_l, op_r) in op_lr.items():
                if key == "ovlp":
                    # Already calculated in the above with coupleJ.
                    continue
                sigvec_add = self._op_lr_dot(
                    op_l,  # type: ignore
                    op_r,  # type: ignore
                    trial_states[j],  # type: ignore
                    key=key,
                )
                if const.use_jax:
                    sigvecs_add.append(sigvec_add)
                else:
                    if (sigvec_states_i := sigvec_states[i]) is None:
                        sigvec_states[i] = sigvec_add
                    else:
                        assert isinstance(sigvec_states_i, np.ndarray)
                        sigvec_states_i += sigvec_add
            if const.use_jax:
                sigvec_states[i] = add_to_sigvec_states(
                    sigvec_states[i], sigvecs_add
                )

        return sigvec_states


@jax.jit
def get_zeros_sigvec_states(
    trial_states: list[jax.Array],
) -> list[jax.Array]:
    return [
        jnp.zeros_like(trial, dtype=jnp.complex128) for trial in trial_states
    ]


@jax.jit
def _stack(psi_states: list[jax.Array]) -> jax.Array:
    return jnp.reshape(jnp.stack([jnp.reshape(x, -1) for x in psi_states]), -1)


@jax.jit
def add_to_sigvec_states(
    sigvec_istate: jax.Array, sigvecs_add: list[jax.Array]
) -> jax.Array:
    if sigvec_istate is None:
        sigvec_istate = sigvecs_add[0]
    else:
        sigvec_istate += sigvecs_add[0]
    for sigvec_add in sigvecs_add[1:]:
        sigvec_istate += sigvec_add
    return sigvec_istate
