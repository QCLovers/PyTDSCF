"""
Conventional MCTDH (so-called A-vector) module
"""

import itertools
import math

import numpy as np
import scipy.linalg as linalg

import pytdscf._helper as helper
from pytdscf import _integrator
from pytdscf._const_cls import const


class multiplyH_CI(object):
    """A class provide various methods for Hamiltonian multiplication in the CI space

    Explanation

    Attributes:
    """

    def __init__(self, ints_spf, matH, ci_coef):
        self.nstate = ci_coef.nstate
        self.ints_spf = ints_spf
        self.matH = matH
        self.nspf_list_states = [
            ci_coef[istate].shape for istate in range(self.nstate)
        ]
        self.split_idx = np.cumsum(
            [
                np.prod(self.nspf_list_states[istate])
                for istate in range(self.nstate)
            ]
        ).tolist()[:-1]

    def dot(self, ci_coef):
        """i dot{A} += H_ciop A"""

        sigvec = multiply_ciop_prod(
            ci_coef, self.ints_spf["ovlp"], self.matH.coupleJ
        )

        if const.oldcode:
            sigvec += multiply_ciop_diag(ci_coef, self.ints_spf["ham1"])
        else:
            if self.matH.general:
                sigvec += multiply_ciop_same_spfs(
                    ci_coef, self.ints_spf, self.matH, opform="general"
                )
            if self.matH.onesite:
                sigvec += multiply_ciop_diag(ci_coef, self.ints_spf["onesite"])
                sigvec += multiply_ciop_nondiag(
                    ci_coef, self.ints_spf["onesite"], self.ints_spf["ovlp"]
                )
        # any problem with this line? (instead of the two lines above)
        #                sigvec += multiply_ciop(ci_coef, self.ints_spf, self.matH, opform = 'onesite')
        return sigvec

    def dot_TEMP4DIPOLE(self, ci_coef_ket):
        """i dot{A} += H_ciop A"""
        if ci_coef_ket.ndof != 1:
            raise NotImplementedError

        sigvec = multiply_ciop_prod(
            ci_coef_ket, self.ints_spf["ovlp"], self.matH.coupleJ
        )  # for scalar term
        if self.matH.general:
            sigvec += multiply_ciop_diff_spfset(
                ci_coef_ket, self.ints_spf, self.matH, opform="general"
            )
        if self.matH.onesite:
            sigvec += multiply_ciop_diff_spfset(
                ci_coef_ket, self.ints_spf, self.matH, opform="onesite"
            )
        return sigvec

    def stack(self, ci_coef):
        ci_flat_states = [
            ci_coef[istate].flatten() for istate in range(self.nstate)
        ]
        ci_flat = np.hstack([x for x in ci_flat_states])
        return ci_flat

    def split(self, ci_flat):
        data = []
        ci_flat_states = np.split(ci_flat, self.split_idx)  # type: ignore
        for istate in range(self.nstate):
            nspf_list = self.nspf_list_states[istate]
            data.append(ci_flat_states[istate].reshape(nspf_list))
        return CICoef(data)


def apply_dipole_ci_coef(ci_coef, ints_spf, ci_coef_init, matOp):
    # equation number (55)
    return (
        multiplyH_CI(ints_spf, matOp, ci_coef_init)
        .dot_TEMP4DIPOLE(ci_coef_init)
        .data
    )


def propagate_CI_SIL(stepsize, ci_coef, ints_spf, matH):
    multiplyH = multiplyH_CI(ints_spf, matH, ci_coef)
    if not const.doRelax:
        ci_coef.data = _integrator.short_iterative_arnoldi(
            -1.0j * stepsize, multiplyH, ci_coef, const.thresh_exp
        ).data
    else:
        ci_coef.data = _integrator.short_iterative_arnoldi(
            -1.0 * stepsize, multiplyH, ci_coef, const.thresh_exp
        ).data


def derivatives_CI(ci_coef, ints_spf, matH_coupleJ):
    ints_spf_ovi = ints_spf["ovlp"]
    ints_spf_ham1 = ints_spf["ham1"]

    """ i dot{A} += H_ciop A"""
    ci_deri = multiply_ciop_diag(ci_coef, ints_spf_ham1)
    ci_deri += multiply_ciop_prod(ci_coef, ints_spf_ovi, matH_coupleJ)

    ci_deri *= -1.0j if not const.doRelax else -1.0

    return ci_deri


class CICoef(object):
    """ """

    def __init__(self, data):
        self.data = data
        self.nstate = len(data)
        self.ndof = len(data[0].shape)

    def __iadd__(self, other):
        for self_array, other_array in zip(self.data, other.data, strict=True):
            self_array += other_array
        return self

    def __imul__(self, scale):
        for self_array in self.data:
            self_array *= scale
        return self

    def __add__(self, other):
        return CICoef(
            [x + y for x, y in zip(self.data, other.data, strict=True)]
        )

    def __sub__(self, other):
        return CICoef(
            [x - y for x, y in zip(self.data, other.data, strict=True)]
        )

    def __mul__(self, scale):
        return CICoef([scale * x for x in self.data])

    def __rmul__(self, scale):
        return CICoef([scale * x for x in self.data])

    def __getitem__(self, istate):
        return self.data[istate]

    def distance(self, other):
        return (self - other).norm()

    def size(self, istate):
        return self.data[istate].size

    def sum_dot_conj(self, other):
        return self.dot_conj(other)

    def dot_conj(self, other):
        retval = 0.0 + 0.0j
        for istate in range(len(self.data)):
            retval += np.sum(np.conj(other[istate]) * self[istate])
        return retval

    def norm(self):
        retval = 0.0
        for istate in range(len(self.data)):
            retval += linalg.norm(self[istate]) ** 2
        retval = math.sqrt(retval)
        return retval

    def autocorr(self, ints_spf):
        r"""evaluate autocorrelation function
        Args:
             ints_spf_autocorr: <i_spf*|j_spf> = \sum_{k} spf_coef(i_spf,k_prim) * spf_coef(j_spf,k_prim)
        """
        ints_spf_autocorr = ints_spf["auto"]
        nstate = self.nstate
        ndof = self.ndof
        sigmavec = self.alloc_zeros_like()
        for istate in range(nstate):
            ci_ket_state = self[istate]
            sigma_state = sigmavec[istate]
            multiply_ciop_prod_state(
                ci_ket_state,
                sigma_state,
                ints_spf_autocorr[(istate, istate)],
                ndof,
            )

        retval = 0.0 + 0.0j
        for istate in range(nstate):
            retval += np.sum(self[istate] * sigmavec[istate])
        return retval

    def pop_states(self):
        return [np.sum(np.conj(coef) * coef).real for coef in self.data]

    def fill(self, val):
        for array in self.data:
            array.fill(val)
        return

    def alloc_zeros_like(self):
        data = []
        for istate in range(self.nstate):
            nspf_list = self[istate].shape
            data.append(np.zeros(nspf_list, dtype=complex))
        return CICoef(data)

    def derivatives(self, ints_spf, matH_coupleJ):
        return derivatives_CI(self, ints_spf, matH_coupleJ)

    # exp(H)->    def propagate(self, stepsize, ints_spf, matH_coupleJ):
    # exp(H)->        self = propagate_CI_SIL(stepsize, self, ints_spf, matH_coupleJ)
    # exp(H)->        self *= 1.0 / self.norm()

    def apply_dipole(self, ints_spf, ci_coef_init, matOp):
        self.data = apply_dipole_ci_coef(self, ints_spf, ci_coef_init, matOp)
        self *= 1.0 / self.norm()
        return

    def ovlp_operator(self, ints_spf, ci_coef_B, matOp):
        # calcurate <A|Op|B>.
        data1 = (
            multiplyH_CI(ints_spf, matOp, ci_coef_B)
            .dot_TEMP4DIPOLE(ci_coef_B)
            .data
        )
        data2 = self.data
        ovlp = np.vdot(data2[0].flatten(), data1[0].flatten())
        return ovlp

    # 2step
    #    def delete_gs(self, ints_spf_dic, ci_coef_dic, matOp, ovlp):
    #        self *= 1.0/self.norm()
    #        return None

    def propagate(self, stepsize, ints_spf, matH):
        # OLD        propagate_CI_SIL(stepsize, self, ints_spf, matH)
        # OLD        self *= 1.0 / self.norm()
        if not const.doRelax:
            propagate_CI_SIL(stepsize, self, ints_spf, matH)
            self *= 1.0 / self.norm()
        else:
            helper.diagonalize_CI(self, ints_spf, matH)

    @staticmethod
    def alloc(basinfo):
        data = []
        for istate in range(basinfo.get_nstate()):
            nspf_list = basinfo.get_nspf_list(istate)
            data.append(np.zeros(nspf_list, dtype=complex))
        return CICoef(data)

    def expectation(self, ints_spf, matH):
        # A* H_{ciop} A
        if const.oldcode:
            energy = (
                multiply_ciop_diag(self, ints_spf["ham1"])
                + multiply_ciop_prod(self, ints_spf["ovlp"], matH.coupleJ)
            ).dot_conj(self) / self.dot_conj(self)
        else:
            sigvec = multiply_ciop_prod(self, ints_spf["ovlp"], matH.coupleJ)
            if matH.onesite:
                sigvec += multiply_ciop_diag(self, ints_spf["onesite"])
                sigvec += multiply_ciop_nondiag(
                    self, ints_spf["onesite"], ints_spf["ovlp"]
                )
            if matH.general:
                sigvec += multiply_ciop_same_spfs(
                    self, ints_spf, matH, opform="general"
                )

            energy = sigvec.dot_conj(self) / self.dot_conj(self)
        return energy

    def construct_mfop(self, ints_spf, matH):
        mfop = {}
        mfop["rho"] = calc_mfop_prod(self, ints_spf["ovlp"])
        if const.oldcode:
            mfop["ham1"] = calc_mfop_onesite_same_spfset(self, ints_spf["ham1"])
        if matH.onesite:
            mfop["onesite"] = calc_mfop_onesite_same_spfset(
                self, ints_spf["onesite"]
            )
            mfop["onesite"].update(
                calc_mfop_onesite_diff_spfset(
                    self, ints_spf["onesite"], ints_spf["ovlp"]
                )
            )
        if matH.general:
            mfop["general"] = calc_mfop_general(self, ints_spf, matH.general)

        mfop["rhoinv"] = {}
        for key_ij, mfop_rho_ij in mfop["rho"].items():
            if key_ij[0] == key_ij[1]:
                mfop["rhoinv"][key_ij] = [
                    helper.matrix_regularized_inverse(x, const.epsrho)
                    for x in mfop_rho_ij
                ]

        return mfop

    def construct_mfop_TEMP4DIPOLE(self, ints_spf, matH, ci_coef_init):
        mfop = {}
        mfop["rho"] = calc_mfop_prod(
            self, ints_spf["ovlp"], ci_coef_ket=ci_coef_init
        )

        if matH.onesite:  # (bra, ket)
            mfop["onesite"] = calc_mfop_onesite_diff_spfset_TEMP4DIPOLE(
                self, ci_coef_init, ints_spf["onesite"], ints_spf["ovlp"]
            )
        if matH.general:
            mfop["general"] = calc_mfop_general_TEMP4DIPOLE(
                self, ci_coef_init, ints_spf, matH.general
            )

        return mfop


def multiply_ciop_prod_state(
    ci_ket_state, sigmavec_state, ints_spf_op, ndof, scale=1.0
):
    dum = np.array(ci_ket_state)
    for kdof in range(ndof):
        indices = tuple(
            [
                i if (i < kdof) else -1 if (i == kdof) else i - 1
                for i in range(ndof)
            ]
        )
        dum = np.tensordot(dum, ints_spf_op[kdof], axes=(kdof, 1)).transpose(
            indices
        )
    sigmavec_state += scale * dum


def multiply_ciop_prod(ci_coef, ints_spf_op, matH_coupleJ):
    r"""sigmavec(I_0,I_1,...) = \prod_{kdof} <I_kdof|Op(kdof)|J_kdof> ci_coef(J_0,J_1,...)

    Args:
         ci_coef:
        ints_spf_op:
            matH_coupleJ:

    Returns:
        sigmavec:
    """
    ndof = ci_coef.ndof
    sigmavec = ci_coef.alloc_zeros_like()
    for istate in range(ci_coef.nstate):
        for jstate in range(ci_coef.nstate):
            ci_ket_state = ci_coef[jstate]
            sigma_state = sigmavec[istate]
            coupleJ = matH_coupleJ[istate][jstate]
            if coupleJ != 0.0:
                multiply_ciop_prod_state(
                    ci_ket_state,
                    sigma_state,
                    ints_spf_op[(istate, jstate)],
                    ndof,
                    coupleJ,
                )
    return sigmavec


def multiply_ciop_nondiag(ci_coef, ints_spf_onesite, ints_spf_ovlp):
    def multiply_ciop_nondiag_state(
        ci_ket_state,
        sigmavec_state,
        ints_spf_onesite_ij,
        ints_spf_ovlp_ij,
        ndof,
    ):
        for op_dof in range(ndof):
            if ints_spf_onesite_ij[op_dof] is not None:
                dum = np.array(ci_ket_state)
                for kdof in range(ndof):
                    indices = tuple(
                        [
                            i if (i < kdof) else -1 if (i == kdof) else i - 1
                            for i in range(ndof)
                        ]
                    )
                    ints_spf_op = (
                        ints_spf_onesite_ij[kdof]
                        if kdof == op_dof
                        else ints_spf_ovlp_ij[kdof]
                    )
                    dum = np.tensordot(
                        dum, ints_spf_op, axes=(kdof, 1)
                    ).transpose(indices)
                sigmavec_state += dum

    nstate = ci_coef.nstate
    ndof = ci_coef.ndof
    sigmavec = ci_coef.alloc_zeros_like()
    for istate in range(nstate):
        for jstate in range(nstate):
            if istate != jstate:
                ci_ket_state = ci_coef[jstate]
                sigma_state = sigmavec[istate]
                multiply_ciop_nondiag_state(
                    ci_ket_state,
                    sigma_state,
                    ints_spf_onesite[(istate, jstate)],
                    ints_spf_ovlp[(istate, jstate)],
                    ndof,
                )
    return sigmavec


def multiply_ciop_diag(ci_coef, ints_spf_op):
    def multiply_ciop_diag_state(
        ci_ket_state, sigmavec_state, ints_spf_op, ndof
    ):  # , scale = 1.0):
        for kdof in range(ndof):
            indices = tuple(
                [
                    i if (i < kdof) else -1 if (i == kdof) else i - 1
                    for i in range(ndof)
                ]
            )
            sigmavec_state += np.tensordot(
                ci_ket_state, ints_spf_op[kdof], axes=(kdof, 1)
            ).transpose(indices)

    ndof = ci_coef.ndof
    sigmavec = ci_coef.alloc_zeros_like()
    for istate in range(ci_coef.nstate):
        for jstate in range(ci_coef.nstate):
            ci_ket_state = ci_coef[jstate]
            sigma_state = sigmavec[istate]
            if istate == jstate:
                multiply_ciop_diag_state(
                    ci_ket_state,
                    sigma_state,
                    ints_spf_op[(istate, jstate)],
                    ndof,
                )
    return sigmavec


def multiply_ciop_same_spfs(ci_coef, ints_spf, matH, *, opform):
    def multiply_ciop_onesite_state(
        ci_ket_state, sigmavec_state, ints_spf_state, ndof, terms
    ):
        for term_onesite in terms:
            op_dof = term_onesite.op_dof
            op_key = term_onesite.op_key
            indices = tuple(
                [
                    i if (i < op_dof) else -1 if (i == op_dof) else i - 1
                    for i in range(ndof)
                ]
            )
            sigmavec_state += term_onesite.coef * np.tensordot(
                ci_ket_state, ints_spf_state[op_key][op_dof], axes=(op_dof, 1)
            ).transpose(indices)

    def multiply_ciop_general_state(
        ci_ket_state, sigmavec_state, ints_spf_state, ndof, terms
    ):
        for term_prodform in terms:
            dum = np.array(ci_ket_state)
            for op_dof, op_key in zip(
                term_prodform.op_dofs, term_prodform.op_keys, strict=True
            ):
                indices = tuple(
                    [
                        i if (i < op_dof) else -1 if (i == op_dof) else i - 1
                        for i in range(ndof)
                    ]
                )
                ints_spf_op = ints_spf_state[op_key][op_dof]
                dum = np.tensordot(
                    dum, ints_spf_op, axes=(op_dof, 1)
                ).transpose(indices)
            sigmavec_state += term_prodform.coef * dum

    nstate = ci_coef.nstate
    ndof = ci_coef.ndof
    sigmavec = ci_coef.alloc_zeros_like()
    for istate_bra in range(nstate):
        for istate_ket in range(nstate):
            ci_ket_state = ci_coef[istate_ket]
            sigma_state = sigmavec[istate_bra]
            if istate_bra == istate_ket:
                ints_spf_state = ints_spf[(istate_bra, istate_ket)]
                if opform == "general":
                    multiply_ciop_general_state(
                        ci_ket_state,
                        sigma_state,
                        ints_spf_state,
                        ndof,
                        matH.general[istate_bra][istate_ket],
                    )
                elif opform == "onesite":
                    multiply_ciop_onesite_state(
                        ci_ket_state,
                        sigma_state,
                        ints_spf_state,
                        ndof,
                        matH.onesite[istate_bra][istate_ket],
                    )
                else:
                    raise ValueError(
                        f"opform must be 'general' or 'onesite', but got {opform}"
                    )
    return sigmavec


def multiply_ciop_diff_spfset(
    ci_coef_ket, ints_spf, matH, *, opform, braket_have_same_shape=True
):
    def multiply_ciop_onesite_state(
        ci_ket_state, sigmavec_state, ints_spf_state, ndof, terms
    ):
        for op_dof in range(ndof):
            ints_spf_onesite = ints_spf_state["onesite"][op_dof]
            if ints_spf_onesite is not None:
                dum = np.array(ci_ket_state)
                for kdof in range(ndof):
                    indices = tuple(
                        [
                            i if (i < kdof) else -1 if (i == kdof) else i - 1
                            for i in range(ndof)
                        ]
                    )
                    ints_spf_op = (
                        ints_spf_onesite
                        if kdof == op_dof
                        else ints_spf_state["ovlp"][kdof]
                    )
                    dum = np.tensordot(
                        dum, ints_spf_op, axes=(kdof, 1)
                    ).transpose(indices)
                sigmavec_state += dum

    def multiply_ciop_general_state(
        ci_ket_state, sigmavec_state, ints_spf_state, ndof, terms
    ):
        for term_prodform in terms:
            dum = np.array(ci_ket_state)
            op_dofs = term_prodform.op_dofs
            op_keys = term_prodform.op_keys
            for kdof in range(ndof):
                indices = tuple(
                    [
                        i if (i < kdof) else -1 if (i == kdof) else i - 1
                        for i in range(ndof)
                    ]
                )
                ints_spf_op = (
                    ints_spf_state[op_keys[op_dofs.index(kdof)]][kdof]
                    if kdof in op_dofs
                    else ints_spf_state["ovlp"][kdof]
                )
                dum = np.tensordot(dum, ints_spf_op, axes=(kdof, 1)).transpose(
                    indices
                )
            sigmavec_state += term_prodform.coef * dum

    nstate = ci_coef_ket.nstate
    ndof = ci_coef_ket.ndof
    if not braket_have_same_shape:
        raise NotImplementedError
    sigmavec = ci_coef_ket.alloc_zeros_like()
    for istate_bra in range(nstate):
        for istate_ket in range(nstate):
            ci_ket_state = ci_coef_ket[istate_ket]
            sigma_state = sigmavec[istate_bra]
            if istate_bra == istate_ket:
                ints_spf_state = ints_spf[(istate_bra, istate_ket)]
                if opform == "general":
                    multiply_ciop_general_state(
                        ci_ket_state,
                        sigma_state,
                        ints_spf_state,
                        ndof,
                        matH.general[istate_bra][istate_ket],
                    )
                elif opform == "onesite":
                    multiply_ciop_onesite_state(
                        ci_ket_state,
                        sigma_state,
                        ints_spf_state,
                        ndof,
                        matH.onesite[istate_bra][istate_ket],
                    )
                else:
                    raise ValueError(
                        f"opform must be 'general' or 'onesite', but got {opform}"
                    )
    return sigmavec


def calc_mfop_prod(ci_coef_bra, ints_spf_op, ci_coef_ket=None):
    if ci_coef_ket is None:
        ci_coef_ket = ci_coef_bra
    ndof = ci_coef_bra.ndof
    mfop_op = {}
    for key in ints_spf_op.keys():
        istate_bra = key[0]
        istate_ket = key[1]
        ci_bra = ci_coef_bra[istate_bra]
        ci_ket = ci_coef_ket[istate_ket]
        spfints = ints_spf_op[(istate_bra, istate_ket)]
        mfop_op[(istate_bra, istate_ket)] = calc_mfop_prod_state(
            ci_bra, ci_ket, spfints, ndof
        )
    return mfop_op


def calc_mfop_prod_diff_spf(ci_coef_bra, ci_coef_ket, ints_spf_op):
    ndof = ci_coef_bra.ndof
    mfop_op = {}
    for key in ints_spf_op.keys():
        istate_bra = key[0]
        istate_ket = key[1]
        ci_bra = ci_coef_bra[istate_bra]
        ci_ket = ci_coef_ket[istate_ket]
        spfints = ints_spf_op[(istate_bra, istate_ket)]
        mfop_op[(istate_bra, istate_ket)] = calc_mfop_prod_state(
            ci_bra, ci_ket, spfints, ndof
        )
    return mfop_op


def calc_mfop_prod_state(ci_bra, ci_ket, spfints, ndof):
    mfop_state = []
    for idof in range(ndof):
        dofs_wo_idof = [i for i in range(ndof) if i != idof]
        dum = np.array(ci_ket)
        for kdof in dofs_wo_idof:
            indices = tuple(
                [
                    i if (i < kdof) else -1 if (i == kdof) else i - 1
                    for i in range(ndof)
                ]
            )
            """dum = np.tensordot(dum, spfints[kdof], axes=(kdof,1))#<<= 1 means |ket> in ints[<bra|, |ket>]"""
            dum = np.tensordot(dum, spfints[kdof], axes=(kdof, 1)).transpose(
                indices
            )
        mfop_state.append(
            np.tensordot(
                np.conj(ci_bra), dum, axes=(dofs_wo_idof, dofs_wo_idof)
            )
        )
    return mfop_state


def calc_mfop_onesite_diff_spfset_TEMP4DIPOLE(
    ci_coef_bra, ci_coef_ket, ints_spf_onesite, ints_spf_ovlp
):
    nstate = ci_coef_bra.nstate
    ndof = ci_coef_bra.ndof
    mfop_op = {}
    for istate_bra, istate_ket in itertools.product(range(nstate), repeat=2):
        ci_bra = ci_coef_bra[istate_bra]
        ci_ket = ci_coef_ket[istate_ket]

        mfop_op[(istate_bra, istate_ket)] = calc_mfop_onesite_diff_spfset_state(
            ci_bra,
            ci_ket,
            ints_spf_onesite[(istate_bra, istate_ket)],
            ints_spf_ovlp[(istate_bra, istate_ket)],
            ndof,
        )

    return mfop_op


def calc_mfop_onesite_diff_spfset(ci_coef, ints_spf_onesite, ints_spf_ovlp):
    nstate = ci_coef.nstate
    ndof = ci_coef.ndof
    mfop_op = {}
    for istate_bra, istate_ket in itertools.product(range(nstate), repeat=2):
        if istate_bra != istate_ket:
            ci_bra = ci_coef[istate_bra]
            ci_ket = ci_coef[istate_ket]
            mfop_op[(istate_bra, istate_ket)] = (
                calc_mfop_onesite_diff_spfset_state(
                    ci_bra,
                    ci_ket,
                    ints_spf_onesite[(istate_bra, istate_ket)],
                    ints_spf_ovlp[(istate_bra, istate_ket)],
                    ndof,
                )
            )
    return mfop_op


def calc_mfop_onesite_diff_spfset_state(
    ci_bra, ci_ket, ints_spf_onesite_ij, ints_spf_ovlp_ij, ndof
):
    mfop_state = []
    for idof in range(ndof):
        mfop_state.append(
            np.zeros((ci_bra.shape[idof], ci_ket.shape[idof]), dtype=complex)
        )

    for idof in range(ndof):
        dofs_wo_idof = [i for i in range(ndof) if i != idof]
        for op_dof in range(ndof):
            if op_dof == idof:
                continue
            if ints_spf_onesite_ij[op_dof] is not None:
                dum = np.array(ci_ket)
                for kdof in dofs_wo_idof:
                    indices = tuple(
                        [
                            i if (i < kdof) else -1 if (i == kdof) else i - 1
                            for i in range(ndof)
                        ]
                    )
                    ints_spf_op = (
                        ints_spf_onesite_ij[kdof]
                        if kdof == op_dof
                        else ints_spf_ovlp_ij[kdof]
                    )
                    dum = np.tensordot(
                        dum, ints_spf_op, axes=(kdof, 1)
                    ).transpose(indices)
                mfop_state[idof] += np.tensordot(
                    np.conj(ci_bra), dum, axes=(dofs_wo_idof, dofs_wo_idof)
                )
    return mfop_state


def calc_mfop_onesite_same_spfset(ci_coef, ints_spf_op):
    nstate = ci_coef.nstate
    ndof = ci_coef.ndof
    mfop_op = {}
    for istate_bra, istate_ket in itertools.product(range(nstate), repeat=2):
        if istate_bra == istate_ket:
            ci_bra = ci_coef[istate_bra]
            ci_ket = ci_coef[istate_ket]
            spfints = ints_spf_op[(istate_bra, istate_ket)]
            mfop_op[(istate_bra, istate_ket)] = (
                calc_mfop_onesite_same_spfset_state(
                    ci_bra, ci_ket, spfints, ndof
                )
            )
    return mfop_op


def calc_mfop_onesite_same_spfset_state(ci_bra, ci_ket, spfints, ndof):
    mfop_state = []
    for idof in range(ndof):
        dofs_wo_idof = [i for i in range(ndof) if i != idof]
        sigmavec_ket = np.zeros_like(ci_ket)
        for kdof in dofs_wo_idof:
            indices = tuple(
                [
                    i if (i < kdof) else -1 if (i == kdof) else i - 1
                    for i in range(ndof)
                ]
            )
            sigmavec_ket += np.tensordot(
                ci_ket, spfints[kdof], axes=(kdof, 1)
            ).transpose(indices)
        mfop_state.append(
            np.tensordot(
                np.conj(ci_bra), sigmavec_ket, axes=(dofs_wo_idof, dofs_wo_idof)
            )
        )
    return mfop_state


def calc_mfop_general_TEMP4DIPOLE(
    ci_coef_bra, ci_coef_ket, ints_spf, matH_general
):
    nstate = ci_coef_bra.nstate
    ndof = ci_coef_bra.ndof
    mfop_general = {}
    for istate_bra in range(nstate):
        for istate_ket in range(nstate):
            ci_bra = ci_coef_bra[istate_bra]
            ci_ket = ci_coef_ket[istate_ket]
            if matH_general[istate_bra][istate_ket] != []:
                # istate_bra == istate_ket: (has not been checked but should be applicable for the case istate_bra != istate_ket as well!)
                ints_spf_state = ints_spf[(istate_bra, istate_ket)]
                terms = matH_general[istate_bra][istate_ket]
                mfop_general[(istate_bra, istate_ket)] = (
                    calc_mfop_general_state(
                        ci_bra, ci_ket, ints_spf_state, ndof, terms
                    )
                )
    return mfop_general


def calc_mfop_general(ci_coef, ints_spf, matH_general):
    nstate = ci_coef.nstate
    ndof = ci_coef.ndof
    mfop_general = {}
    for istate_bra in range(nstate):
        for istate_ket in range(nstate):
            ci_bra = ci_coef[istate_bra]
            ci_ket = ci_coef[istate_ket]
            if matH_general[istate_bra][istate_ket] != []:
                # istate_bra == istate_ket: (has not been checked but should be applicable for the case istate_bra != istate_ket as well!)
                ints_spf_state = ints_spf[(istate_bra, istate_ket)]
                terms = matH_general[istate_bra][istate_ket]
                mfop_general[(istate_bra, istate_ket)] = (
                    calc_mfop_general_state(
                        ci_bra, ci_ket, ints_spf_state, ndof, terms
                    )
                )
    return mfop_general


def calc_mfop_general_state(ci_bra, ci_ket, ints_spf_state, ndof, terms):
    mfops_state = {}
    for op_key in ["d^2", "q^1", "q^2", "q^3", "q^4", "ovlp"]:
        if op_key not in mfops_state.keys():
            mfops_state[op_key] = {}

    for term_prodform in terms:
        mode_ops = term_prodform.mode_ops
        for idof in range(ndof):
            dofs_wo_idof = [i for i in range(ndof) if i != idof]
            dum = np.array(ci_ket)
            for kdof in dofs_wo_idof:
                indices = tuple(
                    [
                        i if (i < kdof) else -1 if (i == kdof) else i - 1
                        for i in range(ndof)
                    ]
                )
                if kdof in mode_ops.keys():
                    ints_spf_op = ints_spf_state[mode_ops[kdof]][kdof]
                    dum = np.tensordot(
                        dum, ints_spf_op, axes=(kdof, 1)
                    ).transpose(indices)
                else:
                    ints_spf_op = ints_spf_state["ovlp"][kdof]
                    dum = np.tensordot(
                        dum, ints_spf_op, axes=(kdof, 1)
                    ).transpose(indices)

            if idof in mode_ops.keys():
                if idof not in mfops_state[mode_ops[idof]]:
                    mfops_state[mode_ops[idof]][idof] = (
                        term_prodform.coef
                        * np.tensordot(
                            np.conj(ci_bra),
                            dum,
                            axes=(dofs_wo_idof, dofs_wo_idof),
                        )
                    )
                else:
                    mfops_state[mode_ops[idof]][idof] += (
                        term_prodform.coef
                        * np.tensordot(
                            np.conj(ci_bra),
                            dum,
                            axes=(dofs_wo_idof, dofs_wo_idof),
                        )
                    )
            else:
                if idof not in mfops_state["ovlp"]:
                    mfops_state["ovlp"][idof] = (
                        term_prodform.coef
                        * np.tensordot(
                            np.conj(ci_bra),
                            dum,
                            axes=(dofs_wo_idof, dofs_wo_idof),
                        )
                    )
                else:
                    mfops_state["ovlp"][idof] += (
                        term_prodform.coef
                        * np.tensordot(
                            np.conj(ci_bra),
                            dum,
                            axes=(dofs_wo_idof, dofs_wo_idof),
                        )
                    )
    return mfops_state
