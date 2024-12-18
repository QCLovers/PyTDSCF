"""
Tensor Contraction Module is separated from module to achieve acceleration
"""

from __future__ import annotations

import itertools
from time import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as linalg
from opt_einsum import contract

import pytdscf
from pytdscf._const_cls import const
from pytdscf._mpo_cls import OperatorCore
from pytdscf._site_cls import SiteCoef


def is_unitmat_op(op_block_single: int | np.ndarray | jax.Array) -> bool:
    """Whether op_l or op_c or op_r is identity or not.
    Args:
        op_block_single (int or numpy.ndarray) : op_l, op_c, op_r
    Returns:
        bool : Whether op_l or op_c or op_r is identity or not
    """
    return isinstance(op_block_single, int)


def contract_with_site_concat(mat_bra, mat_ket, op_LorR_concat, op_site_concat):
    if mat_bra.gauge == "L":
        contraction = "xmn,mri,xrs,nsj->xij"
    elif mat_bra.gauge == "R":
        contraction = "xmn,irm,xrs,jsn->xij"
    else:
        raise AssertionError(
            f"mat_bra.gauge is neither L nor R, but {mat_bra.gauge}"
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
    op_LorR: int | np.ndarray | jax.Array,
    op_site: int | np.ndarray | jax.Array,
) -> np.ndarray | jax.Array:
    r"""Contraction between p-site bra, p-site ket, p-site operator and side-block

    Args:
        mat_bra (SiteCoef): :math:`\left(L^{\tau_{p-1}^\prime\tau_{p}^\prime}_{j_p^\prime}\right)^\ast` or similar R
        mat_ket (SiteCoef): :math:`L_{\tau_{p-1}\tau_{p}}^{j_p}` or similar R
        op_LorR (int | np.ndarray | jax.Array) : pre-calculated block operator :math:`E_{\tau_{p-1}}^{\tau_{p-1}^\prime}` \
            or similar p+1 index
        op_site (int | np.ndarray | jax.Array) : p-site operator :math:`O_{j_p}^{j_p^\prime}`

    Returns:
        np.ndarray | jax.Array : contracted system block :math:`E_{\tau_{p}}^{\tau_{p}^\prime}`
    """
    """
    op_next = np.einsum('nri,nrj->ij',\
              np.einsum('mn,mri->nri',op_LorR,coef_bra),\
              np.einsum('rs,nsj->nrj',op_site,coef_ket))
    """
    if mat_bra.gauge == "L":
        contraction = "mri,nsj,rs,mn->ij"
    elif mat_bra.gauge == "R":
        contraction = "irm,jsn,rs,mn->ij"
    else:
        raise AssertionError(
            f"mat_bra.gauge is neither L nor R, but {mat_bra.gauge}"
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
    # if is_unitmat_op(op_site):
    if isinstance(op_site, int):
        contraction = contraction.replace(",rs", "")
        contraction = contraction.replace("r", "s")
    else:
        operator.append(op_site)
    # if is_unitmat_op(op_LorR):
    if isinstance(op_LorR, int):
        contraction = contraction.replace(",mn", "")
        contraction = contraction.replace("m", "n")
    else:
        operator.append(op_LorR)

    if const.use_jax:
        op_next = jnp.einsum(contraction, *operator)
    else:
        op_next = contract(contraction, *operator)
    return op_next


def contract_with_site_mpo(
    mat_bra: SiteCoef,
    mat_ket: SiteCoef,
    op_LorR: int | np.ndarray | jax.Array,
    op_site: OperatorCore | int,
) -> np.ndarray | jax.Array:
    r"""Contraction between p-site bra, p-site ket, p-site operator and side-block

    Args:
        mat_bra (SiteCoef): :math:`\left(L^{\tau_{p-1}^\prime\tau_{p}^\prime}_{j_p^\prime}\right)^\ast` or similar R
        mat_ket (SiteCoef): :math:`L_{\tau_{p-1}\tau_{p}}^{j_p}` or similar R
        op_LorR (int | np.ndarray | jax.Array) : pre-calculated block operator \
            :math:`[O^{[:p-1]}_{\rm sys}]\substack{\tau_{p-1}^\prime \\ \beta_{p-1} \\ \tau_{p-1}}` \
            or similar p+1 index
        op_site (int | np.ndarray | jax.Array) : p-site operator
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
    """
    op_next = np.einsum('nri,nrj->ij',\
              np.einsum('mn,mri->nri',op_LorR,coef_bra),\
              np.einsum('rs,nsj->nrj',op_site,coef_ket))

    Tensor network contraction diagram

    When sweep ===>,

     -m m-|-i
     |    r
     |    r
     -p p-|-q
     |    s
     |    s
     -n n-|-j

    When sweep <===,

    i-|-m m-
      r    |
      r    |
    p-|-q q-
      s    |
      s    |
    j-|-n n-

    """
    if mat_bra.gauge == "L":
        contraction = "mri,nsj,prsq,mpn->iqj"
    elif mat_bra.gauge == "R":
        contraction = "irm,jsn,prsq,mqn->ipj"
    else:
        raise AssertionError(
            f"mat_bra.gauge is neither L nor R, but {mat_bra.gauge}"
        )

    coef_bra: np.ndarray | jax.Array
    coef_ket: np.ndarray | jax.Array
    if const.use_jax:
        coef_bra = jnp.conj(mat_bra.data)
        coef_ket = mat_ket.data
    else:
        coef_bra = np.conj(mat_bra)
        coef_ket = np.array(mat_ket)

    operator: list[jax.Array] | list[np.ndarray] = [coef_bra, coef_ket]  # type: ignore
    if isinstance(op_site, int):
        contraction = (
            contraction.replace(",prsq", "").replace("r", "s").replace("p", "q")
        )
    else:
        assert isinstance(op_site, OperatorCore), f"op_site = {op_site}"
        data = op_site.data
        assert isinstance(data, np.ndarray | jax.Array)
        if op_site.only_diag:
            contraction = contraction.replace("prsq", "psq").replace("r", "s")
        else:
            assert len(data.shape) == 4, f"op_site.data.shape = {data.shape}"
        operator.append(data)  # type: ignore
    # if is_unitmat_op(op_LorR):
    if isinstance(op_LorR, int):
        if mat_bra.gauge == "L":
            contraction = contraction.replace(",mpn", "").replace("m", "n")
        else:
            contraction = contraction.replace(",mqn", "").replace("m", "n")
    else:
        assert isinstance(op_LorR, np.ndarray | jax.Array)
        operator.append(op_LorR)  # type: ignore
    if const.use_jax:
        op_next = jnp.einsum(contraction, *operator)
    else:
        op_next = contract(contraction, *operator)
    return op_next


def mfop_site(matC_bra, matC_ket, op_left, op_right):
    """
    mfop_site = np.einsum('imr,inr->mn',np.conj(matC_bra),\
                np.einsum('ij,jnr->inr',op_left,\
                np.einsum('rs,jns->jnr',op_right,matC_ket)))
    """
    assert matC_bra.gauge == "C"
    assert matC_ket.gauge == "C"

    # if is_unitmat_op(op_right):
    if isinstance(op_right, int):
        dum_jnr = matC_ket.data
    else:
        if const.use_jax:
            dum_jnr = jnp.einsum("rs,jns->jnr", op_right, matC_ket.data)
        else:
            dum_jnr = np.einsum("rs,jns->jnr", op_right, matC_ket)

    # if is_unitmat_op(op_left):
    if isinstance(op_left, int):
        dum_inr = dum_jnr
    else:
        if const.use_jax:
            dum_inr = jnp.einsum("ij,jnr->inr", op_left, dum_jnr)
        else:
            dum_inr = np.einsum("ij,jnr->inr", op_left, dum_jnr)

    if const.use_jax:
        mfop_site = jnp.einsum("imr,inr->mn", jnp.conj(matC_bra.data), dum_inr)
    else:
        mfop_site = np.einsum("imr,inr->mn", np.conj(matC_bra), dum_inr)
    return mfop_site


def mfop_site_concat(matC_bra, matC_ket, op_left_concat, op_right_concat):
    assert matC_bra.gauge == "C"
    assert matC_ket.gauge == "C"
    subscripts = "imr,kij,krs,jns->mn"
    if const.use_jax:
        mfop_site_concat = jnp.einsum(
            subscripts,
            jnp.conj(matC_bra.data),
            op_left_concat,
            op_right_concat,
            matC_ket.data,
        )
    else:
        mfop_site_concat = contract(
            subscripts,
            np.conj(matC_bra),
            op_left_concat,
            op_right_concat,
            matC_ket,
        )
    return mfop_site_concat


class SplitStack:
    def __init__(self, psi_states: list[np.ndarray] | list[jax.Array]):
        self._split_idx: list[int] = np.cumsum(
            [x.size for x in psi_states]
        ).tolist()[:-1]  # type: ignore
        # if const.use_jax:
        #     self._split_idx = jnp.array(self._split_idx)
        self.matC_sval_shapes: list[tuple[int, ...]] = [
            x.shape for x in psi_states
        ]

    def stack(
        self, psi_states: list[np.ndarray] | list[jax.Array]
    ) -> np.ndarray | jax.Array:
        """stack MPS p-site coef for each electronic states 1 dimensional

        [(n,m,n) shape array,(n,m,n) shape array] --> (1,2n^2m) shape array

        Args:
            psi_states (list[np.ndarray]): psi_states[i] denotes i-electronic states p-site coefficient (3-rank tensor)

        Returns:
            np.ndarray: flattened list. psi[i] denotes i-electronic states

        """
        if const.use_jax:
            # psi = jnp.hstack([x.flatten() for x in psi_states])
            return _stack(psi_states)
        else:
            psi = np.hstack([x.flatten() for x in psi_states])
        return psi

    def split(
        self, psi: np.ndarray | jax.Array
    ) -> list[np.ndarray] | list[jax.Array]:
        """split MPS p-site coef for each electronic states from 1 dimensional to 3-rank tensor

        Correspond to reverse stack

        (1,2n^2m) shape array --> [(n,m,n) shape array,(n,m,n) shape array]

        Args:
            psi (list[np.ndarray]): psi[i] denotes i-electronic states p-site coefficient (3-rank tensor)

        Returns:
            np.ndarray : 3-rank tensor list. psi_states[i] denotes i-electronic states

        """

        psi_states: list[np.ndarray] | list[jax.Array]
        if const.use_jax:
            # Splitting is difficult to be jit-compiled because of the variable length
            psi_states = [
                jnp.reshape(x, self.matC_sval_shapes[i])
                for i, x in enumerate(jnp.split(psi, self._split_idx))
            ]
        else:
            psi_states = [
                x.reshape(self.matC_sval_shapes[i])
                for i, x in enumerate(np.split(psi, self._split_idx))
            ]
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
                    str,
                    tuple[
                        np.ndarray | jax.Array,
                        np.ndarray | jax.Array,
                        np.ndarray | jax.Array,
                    ],
                ]
            ]
        ],
        psi_states: list[np.ndarray] | list[jax.Array],
        matH_cas=None,
    ):
        self.matH_cas = matH_cas
        self.op_lcr_states = op_lcr_states
        super().__init__(psi_states)

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
            # if is_unitmat_op(_op_l):
            if isinstance(_op_l, int):
                contraction = contraction.replace(",ab", "")
                contraction = contraction.replace("a", "b")
            else:
                operator.append(_op_l)
            # if is_unitmat_op(_op_c):
            if isinstance(_op_c, int):
                contraction = contraction.replace(",ij", "")
                contraction = contraction.replace("i", "j")
            else:
                operator.append(_op_c)
            # if is_unitmat_op(_op_r):
            if isinstance(_op_r, int):
                contraction = contraction.replace(",rs", "")
                contraction = contraction.replace("r", "s")
            else:
                operator.append(_op_r)
            return jnp.einsum(contraction, *operator)
        else:
            # Numpy can be accelerated by BLAS ZGEMM due to the hermitian property of the operator
            # if is_unitmat_op(_op_l):
            if isinstance(_op_l, int):
                dum_l_cr = _trial.reshape(_trial.shape[0], -1)
                sig_shape_0 = _trial.shape[0]
            else:
                dum_l_cr = linalg.blas.zgemm(
                    alpha=1.0, a=_op_l, b=_trial.reshape(_op_l.shape[1], -1)
                )
                sig_shape_0 = _op_l.shape[0]

            # if is_unitmat_op(_op_c):
            if isinstance(_op_c, int):
                dum_c_rl = dum_l_cr.T.reshape(_trial.shape[1], -1)
                sig_shape_1 = _trial.shape[1]
            else:
                dum_c_rl = linalg.blas.zgemm(
                    alpha=1.0, a=_op_c, b=dum_l_cr.T.reshape(_op_c.shape[1], -1)
                )
                sig_shape_1 = _op_c.shape[0]

            # if is_unitmat_op(_op_r):
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
        if const.use_jax:
            sigvec_states = get_zeros_sigvec_states(trial_states)
        else:
            sigvec_states = [
                np.zeros_like(trial, dtype=np.complex128)
                for trial in trial_states
            ]
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
                    sigvec_states[i] += sigvec_add1 + sigvec_add2 + sigvec_add3
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
                    str,
                    tuple[
                        int | np.ndarray | jax.Array,
                        int | np.ndarray | jax.Array,
                    ],
                ]
            ]
        ],
        matH_cas: pytdscf.hamiltonian_cls.HamiltonianMixin,
        psi_states,
    ):
        self.matH_cas = matH_cas
        self.op_lr_states = op_lr_states
        super().__init__(psi_states)

    def _op_lr_dot(self, _op_l, _op_r, _trial):
        # match (is_unitmat_op(_op_l), is_unitmat_op(_op_r)):
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
        if const.use_jax:
            sigvec_states = get_zeros_sigvec_states(trial_states)
        else:
            sigvec_states = [
                np.zeros_like(trial, dtype=np.complex128)
                for trial in trial_states
            ]
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
                    str,
                    tuple[
                        int | np.ndarray | jax.Array,
                        int | np.ndarray | jax.Array,
                        int | np.ndarray | jax.Array,
                    ],
                ]
            ]
        ],
        psi_states: list[np.ndarray] | list[jax.Array],
        hamiltonian=None,
    ):
        super().__init__(op_lcr_states, psi_states, matH_cas=hamiltonian)  # type: ignore

    # @profile
    def _op_lcr_dot(self, _op_l, _op_c, _op_r, _trial):
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
        contraction = "bjs,acb,cijt,rts->air"
        operator = [_trial]
        # if is_unitmat_op(_op_l):
        if isinstance(_op_l, int):
            contraction = contraction.replace(",acb", "").replace("a", "b")
        else:
            operator.append(_op_l)
            if len(_op_l.shape) == 2:
                """one-site operator such as ovlp, q^2"""
                contraction = contraction.replace("acb", "ab")
        # if is_unitmat_op(_op_c):
        if isinstance(_op_c, int):
            # When _op_c is diagonal 1-rank tensor, such as identity matrix.
            contraction = (
                contraction.replace(",cijt", "")
                .replace("i", "j")
                .replace("c", "t")
            )
        elif len(_op_c.shape) == 2:
            # When _op_c is non-diagonal 2-rank tensor, such as ovlp, q^2
            contraction = contraction.replace("cijt", "ij")
            operator.append(_op_c)
        elif _op_c.only_diag:
            # When _op_c is diagonal 3-rank tensor.
            contraction = contraction.replace("cijt", "cjt").replace("i", "j")
            operator.append(_op_c.data)
        else:
            # When _op_c is non-diagonal 4-rank tensor.
            operator.append(_op_c.data)
        # if is_unitmat_op(_op_r):
        if isinstance(_op_r, int):
            contraction = contraction.replace(",rts", "").replace("r", "s")
        else:
            if len(_op_r.shape) == 2:
                """one-site operator such as ovlp, q^2"""
                contraction = contraction.replace("rts", "rs")
            operator.append(_op_r)
        if const.use_jax:
            sig_lcr = jnp.einsum(contraction, *operator)
        else:
            sig_lcr = contract(contraction, *operator)
            # Numpy might be accelerated by BLAS ZGEMM due to the hermitian property of the operator
        return sig_lcr

    # @profile
    def dot(self, trial_states) -> list[np.ndarray] | list[jax.Array]:
        """Only supported MPO"""
        if const.use_jax:
            sigvec_states = get_zeros_sigvec_states(trial_states)
        else:
            sigvec_states = [
                np.zeros_like(trial, dtype=np.complex128)
                for trial in trial_states
            ]
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
                    self._op_lcr_dot(*op_lcr["ovlp"], trial_states[j]) * coupleJ
                )
                if const.use_jax:
                    sigvecs_add.append(sigvec_add)
                else:
                    sigvec_states[i] += sigvec_add

            for key, (op_l, op_c, op_r) in op_lcr.items():
                if key == "ovlp":
                    # Already calculated in the above with coupleJ.
                    continue
                sigvec_add = self._op_lcr_dot(op_l, op_c, op_r, trial_states[j])
                if const.use_jax:
                    sigvecs_add.append(sigvec_add)
                else:
                    sigvec_states[i] += sigvec_add
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
                    str,
                    tuple[
                        int | np.ndarray | jax.Array,
                        int | np.ndarray | jax.Array,
                    ],
                ]
            ]
        ],
        hamiltonian: pytdscf.hamiltonian_cls.TensorHamiltonian,
        psi_states: list[np.ndarray] | list[jax.Array],
    ):
        super().__init__(
            op_lr_states=op_lr_states,
            psi_states=psi_states,
            matH_cas=hamiltonian,
        )

    def _op_lr_dot(self, _op_l, _op_r, _trial):
        """

        -a     r-
        |       |
        -c --- c-
        |       |
        -b b-s s-

        """

        # match (
        #     is_unitmat_op(_op_l),
        #     is_unitmat_op(_op_r),
        # ):
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
            return contract(contraction, *operator)

    def dot(self, trial_states):
        if const.use_jax:
            sigvec_states = get_zeros_sigvec_states(trial_states)
        else:
            sigvec_states = [
                np.zeros_like(trial, dtype=np.complex128)
                for trial in trial_states
            ]
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
                    self._op_lr_dot(*op_lr["ovlp"], trial_states[j]) * coupleJ
                )
                if const.use_jax:
                    sigvecs_add.append(sigvec_add)
                else:
                    sigvec_states[i] += sigvec_add
            for key, (op_l, op_r) in op_lr.items():
                if key == "ovlp":
                    # Already calculated in the above with coupleJ.
                    continue
                sigvec_add = self._op_lr_dot(op_l, op_r, trial_states[j])
                if const.use_jax:
                    sigvecs_add.append(sigvec_add)
                else:
                    sigvec_states[i] += sigvec_add
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
    for sigvec_add in sigvecs_add:
        sigvec_istate += sigvec_add
    return sigvec_istate
