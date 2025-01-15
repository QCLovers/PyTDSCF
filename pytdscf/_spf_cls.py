"""
Single Particle Function (SPF) class
(currentry, deprecated)
"""

from __future__ import annotations

import copy
import itertools
import math
from logging import getLogger

import numpy as np
import scipy.linalg

import pytdscf._ode_cls as ode_cls
from pytdscf._const_cls import const
from pytdscf.basis._primints_cls import PrimInts, ovi_HO_FBR_matrix
from pytdscf.model_cls import Model

logger = getLogger("main").getChild(__name__)


def check_orthogonal_spf(spf_coef):
    istate = 0
    ndof = spf_coef.ndof
    logger.debug("-" * 40)
    for idof in range(ndof):
        logger.debug("idof:", idof)
        nspf = spf_coef[istate][idof].shape[0]
        for spf_pair in itertools.product(range(nspf), repeat=2):
            dum = np.vdot(
                spf_coef[istate][idof][spf_pair[0]],
                spf_coef[istate][idof][spf_pair[1]],
            )
            logger.debug("spf_pair:", spf_pair, "norm:", dum)
    return None


def check_rank(spf_matA, spf_matB):
    # if rank[a_1,...a_n,b1,...b_n] = n, A and B spans the same space.
    istate = 0
    ndof = spf_matA.ndof
    logger.debug("-" * 40)
    for idof in range(ndof):
        nspf = spf_matA[istate][idof].shape[0]
        logger.debug("idof:", idof, "nspf:", nspf)
        A = spf_matA[istate][idof]
        B = spf_matB[istate][idof]
        mat = np.concatenate([A, B])
        logger.debug(mat)
        logger.debug("rank:", np.linalg.matrix_rank(mat, 1.0e-12))
    return None


def gram_schmidt_spf(spf_coef_next, spf_coef_normal=None):
    """orthonormarization spf(i+1)"""
    ndof = spf_coef_next.ndof
    spf_coef_on = spf_coef_next.alloc_zeros_like()  # 'on' means ortho normal
    istate = 0
    if spf_coef_normal:
        spf_coef_proj = (
            spf_coef_next.alloc_zeros_like()
        )  # 'on' means ortho normal
        logger.debug("projected")
        # project normal to next
        for idof in range(ndof):
            nspf = spf_coef_proj[istate][idof].shape[0]
            nprim = spf_coef_proj[istate][idof].shape[1]
            for ispf in range(nspf):
                for jspf in range(nspf):
                    spf_coef_proj[istate][idof][ispf] += (
                        np.vdot(
                            spf_coef_next[istate][idof][jspf],
                            spf_coef_normal[istate][idof][ispf],
                        )
                        * spf_coef_next[istate][idof][jspf]
                    )
    else:
        spf_coef_proj = spf_coef_next
    for idof in range(ndof):
        nspf = spf_coef_on[istate][idof].shape[0]
        nprim = spf_coef_on[istate][idof].shape[1]
        for ispf in range(nspf):
            for iprim in range(nprim):
                dum = spf_coef_proj[istate][idof][(ispf, iprim)]
                for jspf in range(ispf):
                    dum -= (
                        np.vdot(
                            spf_coef_on[istate][idof][jspf],
                            spf_coef_proj[istate][idof][ispf],
                        )
                        * spf_coef_on[istate][idof][(jspf, iprim)]
                    )
                spf_coef_on[istate][idof][(ispf, iprim)] = dum

            spf_coef_on[istate][idof][ispf] /= np.linalg.norm(
                spf_coef_on[istate][idof][ispf], ord=2
            )

    return spf_coef_on.data


def gram_schmidt_spf_2(spf_coef_next, spf_coef_normal=None):
    """orthonormarization spf(i+1)"""

    ndof = spf_coef_next.ndof
    spf_coef_on = spf_coef_next.alloc_zeros_like()  # 'on' means ortho normal
    # for istate in range(nstate):
    istate = 0
    for idof in range(ndof):
        nspf = spf_coef_on[istate][idof].shape[0]
        for ispf in range(nspf):
            dum = spf_coef_next[istate][idof][ispf]
            for jspf in range(nspf):
                dum = (
                    dum
                    - np.vdot(spf_coef_on[istate][idof][jspf], dum)
                    * spf_coef_on[istate][idof][jspf]
                )
            spf_coef_on[istate][idof][ispf] += dum
            if (
                norm := np.linalg.norm(spf_coef_on[istate][idof][ispf])
            ) < 1.0e-10:
                pass
            else:
                spf_coef_on[istate][idof][ispf] /= norm
    return spf_coef_on.data


def apply_dipole_spf_coef(spf_coef, spf_coef_init, ints_prim, mfop, matOp):
    ndof = spf_coef.ndof
    spf_coef_next = spf_coef.alloc_zeros_like()

    istate = 0
    mfop_onesite_state = mfop["onesite"][(istate, istate)]
    mfop_general_state = mfop["general"][(istate, istate)]
    mfop_rho_state = mfop["rho"][(istate, istate)]
    ints_prim_onesite = ints_prim["onesite"][(istate, istate)]
    ints_prim_ovlp = ints_prim["ovlp"][(istate, istate)]
    scalar_coef = matOp.coupleJ[istate][istate]
    for idof in range(ndof):
        nspf = spf_coef_next[istate][idof].shape[0]
        nprim = spf_coef_next[istate][idof].shape[1]
        for ispf in range(nspf):
            for iprim in range(nprim):
                dum_1 = 0.0
                for jspf in range(nspf):
                    for jprim in range(nprim):
                        dum_2 = 0.0
                        # scalar
                        dum_2 += (
                            mfop_rho_state[idof][ispf, jspf]
                            * ints_prim_ovlp[idof][iprim][jprim]
                            * scalar_coef
                        )
                        # onsite
                        # O_{onesite} = \sum{kdof} O[kdof]
                        # = \sum_{kdof!=idof} O[kdof]
                        dum_2 += (
                            mfop_onesite_state[idof][ispf, jspf]
                            * ints_prim_ovlp[idof][iprim][jprim]
                        )
                        # + O[idof]
                        dum_2 += (
                            mfop_rho_state[idof][ispf, jspf]
                            * ints_prim_onesite[idof][iprim][jprim]
                        )

                        # general
                        for op_key in mfop_general_state.keys():
                            if op_key == "d^2":
                                continue
                            if mfop_general_state[op_key] == {}:
                                continue
                            if idof not in mfop_general_state[op_key]:
                                continue
                            dum_2 += (
                                mfop_general_state[op_key][idof][ispf, jspf]
                                * ints_prim[op_key][(istate, istate)][idof][
                                    iprim
                                ][jprim]
                            )

                        dum_2 *= spf_coef_init[istate][idof][(jspf, jprim)]
                        dum_1 += dum_2
                spf_coef_next[istate][idof][(ispf, iprim)] += dum_1
            # spf_coef_next[istate][idof][ispf] /= np.linalg.norm(spf_coef_next[istate][idof][ispf], ord=2)
    return spf_coef_next.data


def derivatives_SPF(spf_coef, ints_prim, mfop, matH):
    mfop_rho = mfop["rho"]
    mfop_rhoinv = mfop["rhoinv"]
    ints_spf_ovlp = itrf_prim2spf(ints_prim["ovlp"], spf_coef, onlyDiag=True)
    nstate = spf_coef.nstate
    ndof = spf_coef.ndof
    spf_deri = spf_coef.alloc_zeros_like()

    for istate_bra in range(nstate):
        spf_deri_istate_bra = spf_deri[istate_bra]
        spf_coef_istate_bra = spf_coef[istate_bra]
        spfints_ovlpinv = [
            scipy.linalg.inv(x) for x in ints_spf_ovlp[(istate_bra, istate_bra)]
        ]
        projector = [
            (spf_coef_istate_bra[idof].T).dot(
                x.dot(np.conj(spf_coef_istate_bra[idof]))
            )
            for idof, x in enumerate(spfints_ovlpinv)
        ]
        for istate_ket in range(nstate):
            ints_prim_ovlp = ints_prim["ovlp"][(istate_bra, istate_ket)]
            spf_coef_istate_ket = spf_coef[istate_ket]
            isDiag = istate_bra == istate_ket
            coupleJ = matH.coupleJ[istate_bra][istate_ket]

            if not isDiag and coupleJ == 0.0:
                continue

            mfop_rho_ij = mfop_rho[(istate_bra, istate_ket)]
            mfop_rhoinv_ii = mfop_rhoinv[(istate_bra, istate_bra)]

            if coupleJ != 0.0:
                for idof in range(ndof):
                    spf_deri_istate_bra[idof] += (
                        _derivatives(
                            ints_prim_ovlp[idof],
                            mfop_rho_ij[idof],
                            mfop_rhoinv_ii[idof],
                            projector[idof],
                            spf_coef_istate_bra[idof],
                            spf_coef_istate_ket[idof],
                            True,
                        )
                        * coupleJ
                    )
            # OLD                      _derivatives_old(ints_prim_ovlp[idof], mfop_rho_ij[idof], mfop_rhoinv_ii[idof],\
            # OLD                                       spfints_ovlpinv[idof], spf_coef_istate_bra[idof], spf_coef_istate_ket[idof], True) * coupleJ

            if "onesite" in mfop:
                if (istate_bra, istate_ket) in mfop["onesite"]:
                    mfop_onesite_state = mfop["onesite"][
                        (istate_bra, istate_ket)
                    ]
                    ints_prim_onesite = ints_prim["onesite"][
                        (istate_bra, istate_ket)
                    ]
                    for idof in range(ndof):
                        spf_deri_istate_bra[idof] += _derivatives(
                            ints_prim_onesite[idof],
                            mfop_rho_ij[idof],
                            mfop_rhoinv_ii[idof],
                            projector[idof],
                            spf_coef_istate_bra[idof],
                            spf_coef_istate_ket[idof],
                        )
                        spf_deri_istate_bra[idof] += _derivatives(
                            ints_prim_ovlp[idof],
                            mfop_onesite_state[idof],
                            mfop_rhoinv_ii[idof],
                            projector[idof],
                            spf_coef_istate_bra[idof],
                            spf_coef_istate_ket[idof],
                        )

            if "general" in mfop and isDiag:
                # assert not isDiag, 'NIY: general type operators are not implemented yet for non-diagonal |al><be| terms'
                mfops_general_state = mfop["general"].get(
                    (istate_bra, istate_ket), {}
                )

                for op_key, mfops_op in mfops_general_state.items():
                    ints_prim_op = ints_prim[op_key][(istate_bra, istate_ket)]
                    # OLD                    for idof in range(ndof):
                    for idof in mfops_op.keys():
                        spf_deri_istate_bra[idof] += _derivatives(
                            ints_prim_op[idof],
                            mfops_op[idof],
                            mfop_rhoinv_ii[idof],
                            projector[idof],
                            spf_coef_istate_bra[idof],
                            spf_coef_istate_ket[idof],
                        )
            if isDiag:
                if const.oldcode:
                    mfop_ham1_ij = mfop["ham1"][(istate_bra, istate_ket)]
                    ints_prim_ham1 = ints_prim["ham1"][(istate_bra, istate_ket)]
                    for idof in range(ndof):
                        spf_deri_istate_bra[idof] += _derivatives(
                            ints_prim_ham1[idof],
                            mfop_rho_ij[idof],
                            mfop_rhoinv_ii[idof],
                            projector[idof],
                            spf_coef_istate_bra[idof],
                            spf_coef_istate_ket[idof],
                        )
                        spf_deri_istate_bra[idof] += _derivatives(
                            ints_prim_ovlp[idof],
                            mfop_ham1_ij[idof],
                            mfop_rhoinv_ii[idof],
                            projector[idof],
                            spf_coef_istate_bra[idof],
                            spf_coef_istate_ket[idof],
                        )

    spf_deri *= -1.0j if not const.doRelax else -1.0

    return spf_deri


def _derivatives(
    ints_prim_op,
    mfop_ham_ij,
    mfop_rhoinv_ii,
    projector,
    spf_coef_bra,
    spf_coef_ket,
    DEBUG=False,
):
    if ints_prim_op is None:
        return 0.0
    """better projector (1-\\sum_{mn}|\vphi_m>(S^{-1})_{mn}<\vphi_n|)"""
    nprim_bra = spf_coef_bra.shape[1]

    """dum(k,r) <= <Ham>^{al,be,mu}(k,l) x c^{be:jstate,mu:idof}(l:spf,q:prim) x O^{al,be,mu}(r:prim,q:prim) """
    dum = mfop_ham_ij.dot(spf_coef_ket.dot(ints_prim_op.T))
    """dum(j,r) <=  <invRho>^{al,al,mu}(j,k) x dum(k,r)"""
    dum = mfop_rhoinv_ii.dot(dum)
    """retval(j,p) = dum(j,r) x (1 - proj(p,r))"""
    retval = dum.dot(np.identity(nprim_bra, dtype=complex) - projector.T)

    return retval


def _derivatives_old(
    ints_prim_op,
    mfop_ham_ij,
    mfop_rhoinv_ii,
    spfints_ovlpinv_ii,
    spf_coef_bra,
    spf_coef_ket,
    DEBUG=False,
):
    """better projector (1-\\sum_{mn}|\vphi_m>(S^{-1})_{mn}<\vphi_n|)"""
    nprim_bra = spf_coef_bra.shape[1]

    """dum(n,p) <= [inv<vphi|vphi>]^{al:istate,mu:idof}(m:spf,n:spf) x c^{al:istate,mu:idof}(m:spf,p:prim)"""
    dum = (spfints_ovlpinv_ii.T).dot(spf_coef_bra)
    """dum(p,r) <= dum(n,p) x \bar{c}^{al:istate,mu:idof}(n:spf,r:prim)"""
    dum = dum.T.dot(np.conj(spf_coef_bra))
    """dum(l,p) <= c^{be:jstate,mu:idof}(l:spf,q:prim) x O^{al,be,mu}(r:prim,q:prim) x (1 - dum(p,r))"""
    dum = spf_coef_ket.dot(ints_prim_op.T).dot(
        np.identity(nprim_bra, dtype=complex) - dum.T
    )
    """dum(j,p) =  <invRho>^{al,al,mu}(j,k) x <Ham>^{al,be,mu}(k,l) dum(l,p)"""
    retval = mfop_rhoinv_ii.dot(mfop_ham_ij.dot(dum))
    """CAUTION!!! const.epsrho makes the dynamics slower"""

    return retval


def itrf_prim2spf(
    ints_prim_op, spf_coef_bra, *, spf_coef_ket=None, onlyDiag=False
):
    ints_spf = {}
    if spf_coef_ket is None:
        spf_coef_ket = spf_coef_bra

    for key, val in ints_prim_op.items():
        istate_bra = key[0]
        istate_ket = key[1]
        if not onlyDiag or istate_bra == istate_ket:
            ints_spf_dof: list[np.ndarray]
            ints_spf_dof = [None for idof in range(len(val))]  # type: ignore
            for idof, ints in enumerate(val):
                if ints is not None:
                    coef_bra = spf_coef_bra[istate_bra][idof]
                    coef_ket = spf_coef_ket[istate_ket][idof]
                    dum = ((np.conj(coef_bra)).dot(ints)).dot(coef_ket.T)
                    # bug->dum = (coef_ket.dot(ints)).dot(np.conj(coef_bra.T))
                    ints_spf_dof[idof] = dum.copy()
            ints_spf[key] = copy.deepcopy(ints_spf_dof)

    return ints_spf


class SPFInts:
    def __init__(
        self,
        ints_prim: PrimInts,
        spf_coef: SPFCoef,
        *,
        spf_coef_ket: SPFCoef | None = None,
        op_keys: list[str] | None = None,
    ) -> None:
        if op_keys is None:
            op_keys = ints_prim.op_keys()
        for op_key in ints_prim.op_keys():
            self[op_key] = itrf_prim2spf(
                ints_prim[op_key], spf_coef, spf_coef_ket=spf_coef_ket
            )

        for statepair_key in itertools.product(
            range(spf_coef.nstate), repeat=2
        ):
            ints_spf_statepair = {}
            for op_key in self.op_keys():
                if statepair_key in self[op_key]:
                    ints_spf_statepair[op_key] = self[op_key][statepair_key]
            self[statepair_key] = ints_spf_statepair

    def __len__(self):
        return len(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)

    def __str__(self):
        return str(self.__dict__)

    def __iter__(self):
        return self.__dict__.iteritems()

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def op_keys(self):
        #        if not hasattr(self, '_op_keys'):
        _op_keys = []
        for key in self.keys():
            if isinstance(key, str):
                _op_keys.append(key)
        return _op_keys

    def statepair_keys(self):
        #        if not hasattr(self, '_statepair_keys'):
        _statepair_keys = []
        for key in self.keys():
            if isinstance(key, tuple):
                assert len(key) == 2
                _statepair_keys.append(key)
        return _statepair_keys


class SPFCoef:
    def __init__(self, data):
        self.data = data
        self.nstate = len(self.data)
        self.ndof = len(self.data[0])

    def __iadd__(self, other):
        for istate in range(len(self.data)):
            for self_array, other_array in zip(
                self.data[istate], other.data[istate], strict=True
            ):
                self_array += other_array
        return self

    def __imul__(self, scale):
        for istate in range(len(self.data)):
            for idof in range(self.ndof):
                self.data[istate][idof] *= scale
        return self

    def __add__(self, other):
        retval = []
        for istate in range(len(self.data)):
            retval.append(
                [
                    x + y
                    for x, y in zip(
                        self.data[istate], other.data[istate], strict=True
                    )
                ]
            )
        return SPFCoef(retval)

    def __sub__(self, other):
        retval = []
        for istate in range(len(self.data)):
            retval.append(
                [
                    x - y
                    for x, y in zip(
                        self.data[istate], other.data[istate], strict=True
                    )
                ]
            )
        return SPFCoef(retval)

    def __mul__(self, scale):
        retval = []
        for istate in range(len(self.data)):
            retval.append([scale * x for x in self.data[istate]])
        return SPFCoef(retval)

    def __rmul__(self, scale):
        retval = []
        for istate in range(len(self.data)):
            retval.append([scale * x for x in self.data[istate]])
        return SPFCoef(retval)

    def __getitem__(self, istate):
        return self.data[istate]

    # OBSOLUTE    def sum_dot_conj(self, other):
    # OBSOLUTE        g = 0.0
    # OBSOLUTE        for istate in range(self.nstate):
    # OBSOLUTE            for idof in range(self.ndof):
    # OBSOLUTE                g += np.sum(np.absolute(np.conj(other[istate][idof]).dot(self[istate][idof].T)))
    # OBSOLUTE        return g

    def overlap(self, other):
        retval = []
        for istate in range(self.nstate):
            dum = []
            for idof in range(self.ndof):
                dum.append(
                    np.conj(other[istate][idof]) @ (self[istate][idof].T)
                )
            retval.append(dum)
        return retval

    def autocorr_ints(self):
        ints_spf = {}
        for istate in range(self.nstate):
            ints_spf_dof = []
            for idof in range(self.ndof):
                r"""<\psi*| = np.conj(np.conj(spf_coef[istate][idof])) = spf_coef[istate][idof]"""
                ints_spf_dof.append(self[istate][idof] @ (self[istate][idof].T))
            ints_spf[(istate, istate)] = ints_spf_dof
        return ints_spf

    def overlap_absmax(self, other):
        absmax = -1e9
        dum = self.overlap(other)
        for istate in range(self.nstate):
            for idof in range(self.ndof):
                absmax = max(absmax, np.max(np.absolute(dum[istate][idof])))
        return absmax

    def norm_absmax_rk5(self):
        absmax = -1e9
        dum = self.overlap(self)
        for istate in range(self.nstate):
            for idof in range(self.ndof):
                absmax = max(
                    absmax, np.max(np.absolute(np.diag(dum[istate][idof])))
                )
        return math.sqrt(absmax)

    def propagate_RK45(self, stepsize, ints_prim, mfop, matH):
        dotPsi = ode_cls.algorithm_RK4(stepsize, self, (ints_prim, mfop, matH))
        g = self.overlap_absmax(dotPsi)
        self += dotPsi
        return g

    #    def propagate_DPRK45(self, stepsize, ints_prim, mfop, matJ):
    def propagate(self, stepsize, ints_prim, mfop, matH):
        """"""
        dotPsi, stepsize_recommend = ode_cls.algorithm_DPRK45(
            stepsize, self, (ints_prim, mfop, matH)
        )

        if stepsize < stepsize_recommend:  # accept
            g = self.overlap_absmax(dotPsi)
            self += dotPsi
        else:
            stepsize_recommend = (
                stepsize_recommend
                if stepsize / 5 < stepsize_recommend
                else stepsize / 5
            )
            nstep = math.ceil(stepsize / stepsize_recommend)
            steps = [stepsize_recommend for i in range(nstep - 1)] + [
                stepsize - stepsize_recommend * (nstep - 1)
            ]
            for h in steps:
                dotPsi, h_recommend = ode_cls.algorithm_DPRK45(
                    h, self, (ints_prim, mfop, matH)
                )
                g = self.overlap_absmax(dotPsi)
                self += dotPsi
                # assert h < h_recommend
        return g

    def apply_dipole(self, spf_coef_init, ints_prim, mfop, matOp):
        self.data = apply_dipole_spf_coef(
            self, spf_coef_init, ints_prim, mfop, matOp
        )

    def gram_schmidt(self, spf_coef_norm=None):
        # self.data = gram_schmidt_spf(self,spf_coef_norm)
        self.data = gram_schmidt_spf_2(self, spf_coef_norm)

    def check_orthogonal(self):
        check_orthogonal_spf(self)
        self.data = gram_schmidt_spf(self)
        check_orthogonal_spf(self)

    def derivatives(self, ints_prim, mfop, matH):
        return derivatives_SPF(self, ints_prim, mfop, matH)

    def alloc_zeros_like(self):
        data = []
        for istate in range(self.nstate):
            spf_coef_dof = []
            for idof in range(self.ndof):
                nspf = self[istate][idof].shape[0]
                nprim = self[istate][idof].shape[1]
                spf_coef_dof.append(np.zeros((nspf, nprim), dtype=complex))
            data.append(spf_coef_dof)
        return SPFCoef(data)

    def alloc_eye_like(self):
        data = []
        for istate in range(self.nstate):
            spf_coef_dof = []
            for idof in range(self.ndof):
                nspf = self[istate][idof].shape(0)
                nprim = self[istate][idof].shape(1)

                spf_coef_dof.append(np.eye(nspf, nprim, dtype=complex))
            data.append(spf_coef_dof)
        return SPFCoef(data)

    @classmethod
    def alloc_random(cls, basinfo):
        pass

    @classmethod
    def alloc_eye(cls, model: Model):
        data = []
        for istate in range(model.get_nstate()):
            spf_coef_dof = []
            for idof in range(model.get_ndof()):
                nspf = model.get_nspf(istate, idof)
                nprim = model.get_nprim(istate, idof)
                spf_coef_dof.append(np.eye(nspf, nprim, dtype=complex))
            data.append(spf_coef_dof)
        return SPFCoef(data)

    @classmethod
    def alloc(cls, basinfo):
        data = []
        for istate in range(basinfo.get_nstate()):
            spf_coef_dof = []
            for idof in range(basinfo.get_ndof()):
                nspf = basinfo.get_nspf(istate, idof)
                nprim = basinfo.get_nprim(istate, idof)
                spf_coef_dof.append(np.zeros((nspf, nprim), dtype=complex))
            data.append(spf_coef_dof)
        return SPFCoef(data)

    @classmethod
    def alloc_proj_gs(cls, basinfo):
        data = []
        for istate in range(basinfo.get_nstate()):
            spf_coef_dof = []
            for idof in range(basinfo.get_ndof()):
                nspf = basinfo.get_nspf(istate, idof)
                pbas_bra = basinfo.get_primbas(istate, idof)
                nprim = pbas_bra.nprim
                coef_ex = np.eye(nspf, nprim, dtype=np.complex128)

                """define GS phonon w.f."""
                assert hasattr(
                    basinfo, "primbas_gs"
                ), 'you need to set "primbas_gs" to prepare projected initial wavefunction as a attribute of Model'
                nspf_gs = 1
                pbas_gs = basinfo.primbas_gs[idof]
                nprim_gs = pbas_gs.nprim
                coef_gs = np.eye(nspf_gs, nprim_gs, dtype=np.complex128)

                """project GS phonon w.f. to the EX phonon primbas space"""
                ovi = ovi_HO_FBR_matrix(pbas_bra, pbas_gs)
                coef_ex_proj = coef_gs @ ovi.T

                assert nspf_gs <= nspf
                """Löwdin orthogonalization for the projected basis"""
                ovlp_spf_proj = np.conj(coef_ex_proj) @ coef_ex_proj.T
                eigvals, eigvecs = scipy.linalg.eig(ovlp_spf_proj)
                ovlp_inv_half = eigvecs @ np.einsum(
                    " j,jk->jk", np.power(eigvals, -0.5), np.conj(eigvecs).T
                )
                coef_ex_proj = ovlp_inv_half @ coef_ex_proj
                assert np.allclose(
                    coef_ex_proj.dot(np.conj(coef_ex_proj).T),
                    np.identity(nspf_gs),
                ), coef_ex_proj.dot(np.conj(coef_ex_proj).T)

                """Schmidt-Orthogonalization for the extended basis"""
                coef_ex[:nspf_gs, :] = coef_ex_proj[:, :]
                for ispf in range(nspf_gs, nspf):
                    while True:
                        v = coef_ex[ispf, :]
                        u = v.copy()
                        for i in range(ispf):
                            ui = coef_ex[i, :]
                            u -= (
                                np.inner(np.conj(ui), v)
                                / np.inner(np.conj(ui), ui)
                            ) * ui
                        if np.linalg.norm(u) > 1.0e-14:
                            coef_ex[ispf, :] = u / scipy.linalg.norm(u)
                            break
                        else:
                            # Rank were dropped.
                            # Set new random vector.
                            logger.debug(
                                "SPF rank dropped in Schmidt orthogonalization. Change to new basis"
                            )
                            logger.debug(f"{istate=}, {idof=}")
                            logger.debug(f"{coef_ex=}")
                            new_vec = np.random.rand(
                                nprim
                            ) + 1j * np.random.rand(nprim)
                            coef_ex[ispf, :] = new_vec / np.linalg.norm(new_vec)

                spf_coef_dof.append(coef_ex)

            data.append(spf_coef_dof)
        return SPFCoef(data)
