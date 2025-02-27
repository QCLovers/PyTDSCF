"""Wave function handling module"""

from copy import deepcopy
from itertools import chain
from time import time

import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger

import pytdscf._helper as helper
from pytdscf._const_cls import const
from pytdscf._mps_cls import MPSCoef, _ovlp_single_state_jax
from pytdscf._mps_mpo import MPSCoefMPO
from pytdscf._mps_parallel import MPSCoefParallel
from pytdscf._mps_sop import MPSCoefSoP
from pytdscf._spf_cls import SPFCoef, SPFInts
from pytdscf.basis._primints_cls import PrimInts
from pytdscf.hamiltonian_cls import (
    HamiltonianMixin,
    PolynomialHamiltonian,
    TensorHamiltonian,
)

try:
    from mpi4py import MPI
except Exception:
    MPI = None  # type: ignore

logger = logger.bind(name="main")


class WFunc:
    """Wave Function (WF) class

    This class holds WF and some operation to WF.

    Attributes:
        ci_coef (mps_cls.MPSCoef) : Decomposed A-vector object. \
            (In MCTDH level, this object class is `ci_cls.CICoef`,\
                 not `mps_cls.MPSCoef`.)
        spf_coef (spf_cls.SPFCoef) : primitive coefficient of SPFs.
        ints_prim (primints_cls.PrimInts) : \
            the integral matrix element of primitives. \
            <χ_i|V|χ_j> is time-independent.
        ints_spf (spf_cls.SPFInts) : \
            the integral matrix element of SPFs. \
            <φ_i|V|φ_j> = Σ_k Σ_l c^*_k c_l <χ_k|V|χ_l> is time-dependent.

    """

    ints_spf: SPFInts | None

    def __init__(
        self, ci_coef: MPSCoef, spf_coef: SPFCoef, ints_prim: PrimInts
    ):
        self.ci_coef = ci_coef
        self.spf_coef = spf_coef
        self.ints_prim = ints_prim
        self.ints_spf = None
        if hasattr(self.ci_coef, "ints_site"):
            self.ci_coef.ints_site = None
        if hasattr(self.ci_coef, "op_sys_sites"):
            self.ci_coef.op_sys_sites = None

    def get_reduced_densities(
        self, remain_nleg: tuple[int, ...]
    ) -> list[np.ndarray]:
        """
        Calculate reduced density matrix of given degree of freedom pair.
        If (0,1) is given, calculate reduced density matrix of 0th and 1st degree of freedom.

        Args:
            remain_nleg (Tuple[int, ...]) : degree of freedom pair

        Returns:
            List[np.ndarray] : reduced density matrix of given degree of freedom pair for each state.
        """
        if not isinstance(self.ci_coef, MPSCoef):
            raise NotImplementedError(
                "Cannot calculate reduced density for MCTDH"
            )
        if not const.standard_method:
            raise NotImplementedError(
                "Cannot calculate reduced density for MPS-MCTDH"
            )
        return self.ci_coef.get_reduced_densities(remain_nleg)

    def expectation(self, matOp: HamiltonianMixin):
        """

        get expectation value of operator

        Args:
            matOp (hamiltonian_cls.HamiltonianMixin) : operator

        Returns:
            float : expectation value (real number)

        """
        ints_spf = SPFInts(self.ints_prim, self.spf_coef)
        expectation = self.ci_coef.expectation(ints_spf, matOp)
        if const.mpi_rank == 0:
            if abs(expectation.imag) > 1e-12:
                logger.warning(
                    f"expectation value {expectation} is not real, maybe operator is not Hermite"
                )
            return expectation.real
        else:
            return None

    def norm(self):
        """

        get WF norm

        Returns:
            float: norm of WF

        """
        return self.ci_coef.norm()

    def autocorr(self):
        """

        get auto-correlation <Psi(t/2)*|Psi(t/2)> of WF

        Returns:
            complex : auto-correlation value

        """
        ints_spf = SPFInts(self.ints_prim, self.spf_coef, op_keys=[])
        ints_spf["auto"] = self.spf_coef.autocorr_ints()
        return self.ci_coef.autocorr(ints_spf)

    def pop_states(self):
        """

        get population of electronic states

        Returns:
            List[float] : [norm of each electronic states]

        """
        return self.ci_coef.pop_states()

    def bonddim(self) -> list[int] | None:
        """

        get bond dimension of WF

        Returns:
            List[int] : bond dimension of WF
        """
        assert isinstance(self.ci_coef, MPSCoef)
        mps = self.ci_coef
        if const.mpi_size == 1:
            bonddim = [site.shape[2] for site in mps.superblock_states[0][:-1]]
        else:
            bonddim_rank = [site.shape[2] for site in mps.superblock_states[0]]
            comm = MPI.COMM_WORLD
            bonddim_all: list[list[int]] = comm.gather(bonddim_rank, root=0)  # type: ignore
            if comm.rank == 0:
                bonddim = []
                for rank in chain(*bonddim_all):
                    bonddim.append(rank)
                bonddim.pop()
            else:
                bonddim = None
        return bonddim

    def propagate(self, matH: HamiltonianMixin, stepsize: float):
        """

        propagate WF with VMF

        Args:
            matH (hamiltonian_cls.HamiltonianMixin) : Hamiltonian
            stepsize (float) : the width of time step [a.u.]

        Returns:
            Tuple[float, List]: (g value, spf_occ)\n
            where spf_occ = [mean field operator(mfop) overlap of each states]

        """

        if const.verbose == 4:
            helper._ElpTime.itrf -= time()
        ints_spf = SPFInts(self.ints_prim, self.spf_coef)
        if const.verbose == 4:
            helper._ElpTime.itrf += time()

        if const.verbose == 4:
            helper._ElpTime.ci -= time()
        _ = self.ci_coef.propagate(stepsize, ints_spf, matH)
        if const.verbose == 4:
            helper._ElpTime.ci += time()

        if const.verbose == 4:
            helper._ElpTime.mfop -= time()
        mfop_spf = self.ci_coef.construct_mfop(ints_spf, matH)
        if const.verbose == 4:
            helper._ElpTime.mfop += time()

        if const.verbose == 4:
            helper._ElpTime.spf -= time()
        g = self.spf_coef.propagate(
            stepsize / 2, self.ints_prim, mfop_spf, matH
        )
        g = self.spf_coef.propagate(
            stepsize / 2, self.ints_prim, mfop_spf, matH
        )
        if const.verbose == 4:
            helper._ElpTime.spf += time()

        spf_occ = [
            mfop_spf["ovlp"][(istate, istate)] for istate in range(matH.nstate)
        ]

        return (g, spf_occ)

    def _ints_wf_ovlp_mpssm(
        self, ci_coef_ket: MPSCoef, conj: bool = True
    ) -> complex:
        assert isinstance(self.ci_coef, MPSCoef)
        assert const.standard_method
        nstate = self.spf_coef.nstate
        ovlp: complex = 0.0
        for istate in range(nstate):
            istate_bra = istate_ket = istate
            ci_bra = self.ci_coef.superblock_states[istate_bra]
            ci_ket = ci_coef_ket.superblock_states[istate_ket]
            block: np.ndarray | jax.Array
            ket: np.ndarray | jax.Array
            bra: np.ndarray | jax.Array
            if const.use_jax:
                bra_data = [
                    jnp.conj(site_bra.data) if conj else site_bra.data
                    for site_bra in ci_bra
                ]
                ket_data = [site_ket.data for site_ket in ci_ket]
                ovlp += complex(_ovlp_single_state_jax(bra_data, ket_data))
            else:
                block = np.ones((1, 1), dtype=complex)
                for site_bra, site_ket in zip(ci_bra, ci_ket, strict=True):
                    bra = np.conjugate(site_bra) if conj else site_bra.data
                    ket = site_ket.data
                    block = np.einsum(
                        "abc,abk->ck", bra, np.einsum("ibk,ai->abk", ket, block)
                    )
                ovlp += complex(block[0, 0])

        return ovlp

    def _ints_wf_ovlp_sop(self, ci_coef_ket, spf_coef_ket, matOp, idof=0):
        assert isinstance(matOp, PolynomialHamiltonian)
        assert isinstance(self.ci_coef, MPSCoefSoP)

        nstate = self.spf_coef.nstate
        identityOp = deepcopy(matOp)
        for istate_bra in range(nstate):
            for istate_ket in range(nstate):
                identityOp.onesite[istate_bra][istate_ket] = {}
                identityOp.general[istate_bra][istate_ket] = {}
        for istate in range(nstate):
            identityOp.coupleJ[istate][istate] = 1
        ints_spf = SPFInts(
            self.ints_prim, self.spf_coef, spf_coef_ket=spf_coef_ket
        )
        mfop = self.ci_coef.construct_mfop_TEMP4DIPOLE(
            ints_spf, identityOp, ci_coef_ket
        )
        ovlp = 0.0
        for istate in range(nstate):
            istate_bra = istate_ket = istate
            mfop_rho = mfop["rho"][(istate_bra, istate_ket)][idof]
            ints_spf_ovlp = ints_spf["ovlp"][(istate_bra, istate_ket)][idof]
            ovlp += np.einsum("ij,ij", mfop_rho, ints_spf_ovlp)
        return ovlp

    def _is_converged(
        self, ci_coef_prev, spf_coef_prev, matOp, conv_tol=1.0e-08
    ):  # 1.e-08 is mctdh default
        """1 - <Phi(i+1)|Phi(i)> < threshold"""
        if isinstance(self.ci_coef, MPSCoef) and const.standard_method:
            ovlp = self._ints_wf_ovlp_mpssm(ci_coef_prev)
        elif type(self.ci_coef) is MPSCoefSoP:
            ovlp = self._ints_wf_ovlp_sop(ci_coef_prev, spf_coef_prev, matOp)
        else:
            raise NotImplementedError

        if const.verbose > 1:
            logger.debug(f"convergence : {abs(ovlp)}")
        if abs(1 - abs(ovlp)) < conv_tol:
            return True
        else:
            return False

    def apply_dipole(self, matOp) -> float:
        """

        Get excited states by operating dipole operator to WF.

        Args:
            matOp (hamiltonian_cls.HamiltonianMixin) : Dipole operator

        """
        ci_coef_init = deepcopy(self.ci_coef)  # save φ_0
        spf_coef_init = deepcopy(self.spf_coef)  # save A_0
        ints_prim = deepcopy(self.ints_prim)

        ints_spf = SPFInts(ints_prim, self.spf_coef)

        for _iter in range(const.maxstep):
            spf_coef_prev = deepcopy(self.spf_coef)  # save φ_i
            ci_coef_prev = deepcopy(self.ci_coef)  # save A_i

            if not const.standard_method:
                """(1) φ_i -> φ_(i+1) (not orthogonal) -> φ_(i+1) (orthogonal)"""
                mfop_spf = self.ci_coef.construct_mfop_TEMP4DIPOLE(
                    ints_spf, matOp, ci_coef_init
                )
                _ = self.spf_coef.apply_dipole(
                    spf_coef_init, ints_prim, mfop_spf, matOp
                )
                _ = self.spf_coef.gram_schmidt()

                """(2) get A_(i+1) by using φ_(i+1) and A_0"""
                ints_spf = SPFInts(
                    ints_prim, self.spf_coef, spf_coef_ket=spf_coef_init
                )
            norm = self.ci_coef.apply_dipole(ints_spf, ci_coef_init, matOp)
            if const.verbose > 1:
                logger.debug(
                    "-" * 40 + "\n" + f"iterations: {_iter} norm: {norm}"
                )

            """(3) convergence"""
            if self._is_converged(ci_coef_prev, spf_coef_prev, matOp):
                break
            if (
                _iter == const.maxstep - 1
            ):  # QUANTICS mctdh default = 10 iterations
                logger.warning(
                    f"Operate O|Ψ> is not converged in {_iter} iterations"
                )
        return norm

    def propagate_SM(
        self,
        matH: HamiltonianMixin,
        stepsize: float,
        istep: int,
    ):
        """

        propagate Standard Method Wave function (SPF == PRIM)

        Args:
            matH (hamiltonian_cls.HamiltonianMixin) : Polynomial SOP Hamiltonian or \
                grid MPO Hamiltonian
            stepsize (float) : the width of time step [a.u.]
            calc_spf_occ (Optional[bool]) : Calculate ``spf_occ``

        Returns:
            List[float] or None : spf_occ, \n
            where spf_occ = [mean field operator(mfop) overlap of each states]

        """
        assert const.standard_method

        if isinstance(self.ci_coef, MPSCoefMPO):
            pass
        elif const.doTDHamil or (not const.doTDHamil and self.ints_spf is None):
            if const.verbose == 4:
                helper._ElpTime.itrf -= time()
            self.ints_spf = SPFInts(self.ints_prim, self.spf_coef)
            if const.verbose == 4:
                helper._ElpTime.itrf += time()

        if const.verbose == 4:
            helper._ElpTime.ci -= time()
        if isinstance(self.ci_coef, MPSCoefParallel):
            assert isinstance(matH, TensorHamiltonian)
            self.ci_coef.propagate(
                stepsize,
                None,
                matH,
                load_balance=(istep - 1) % const.load_balance_interval == 0,
            )
        else:
            self.ci_coef.propagate(stepsize, self.ints_spf, matH)
        if const.verbose == 4:
            helper._ElpTime.ci += time()
        spf_occ = None
        return spf_occ

    def propagate_CMF(self, matH: HamiltonianMixin, stepsize_guess: float):
        """

        propagate WF with CMF (Constant Mean Field)

        Args:
            matH (hamiltonian_cls.HamiltonianMixin) : Hamiltonian
            stepsize_guess (float) : the width of time step [a.u.]

        Returns:
            tuple[float, list, float, float] : (g value, spf_occ, actual stepsize, next stepsize guess),\n
            where spf_occ = [mean field operator(mfop) overlap of each states]

        """
        nstate = matH.nstate

        """construct IntsSPF(t=0)"""
        if const.verbose == 4:
            helper._ElpTime.itrf -= time()
        ints_spf = SPFInts(self.ints_prim, self.spf_coef)
        if const.verbose == 4:
            helper._ElpTime.itrf += time()

        """construct MFOP(t=0)"""
        if const.verbose == 4:
            helper._ElpTime.mfop -= time()
        mfop_spf = self.ci_coef.construct_mfop(ints_spf, matH)
        if const.verbose == 4:
            helper._ElpTime.mfop += time()

        nstep_rk5 = 1
        stepsize = stepsize_guess
        while True:
            while True:
                """(1) propagate CI(t=0->h/2) with IntsSPF(t=0)"""
                if const.verbose == 4:
                    helper._ElpTime.ci -= time()
                ci_coef_trial = deepcopy(self.ci_coef)
                ci_coef_trial.propagate(stepsize / 2, ints_spf, matH)
                if const.verbose == 4:
                    helper._ElpTime.ci += time()

                """(2) propagate SPF(t=0->h/2) with MFOP(t=0)"""
                if const.verbose == 4:
                    helper._ElpTime.spf -= time()
                spf_coef_approx = deepcopy(self.spf_coef)
                for _ in range(nstep_rk5):
                    g = spf_coef_approx.propagate(
                        stepsize / 2 / nstep_rk5, self.ints_prim, mfop_spf, matH
                    )
                if const.verbose == 4:
                    helper._ElpTime.spf += time()

                if const.verbose == 4:
                    helper._ElpTime.mfop -= time()
                mfop_spf_halfstep = ci_coef_trial.construct_mfop(ints_spf, matH)
                if const.verbose == 4:
                    helper._ElpTime.mfop += time()

                """(3) propagate SPF(t=0->h/2) with MFOP(t=h/2)"""
                if const.verbose == 4:
                    helper._ElpTime.spf -= time()
                spf_coef_trial = deepcopy(self.spf_coef)
                for _ in range(nstep_rk5):
                    spf_coef_trial.propagate(
                        stepsize / 2 / nstep_rk5,
                        self.ints_prim,
                        mfop_spf_halfstep,
                        matH,
                    )
                if const.verbose == 4:
                    helper._ElpTime.spf += time()

                """Error estimation for SPF"""
                err_spf = 0.0
                spf_coef_diff = spf_coef_approx - spf_coef_trial
                mfop_rho = mfop_spf["rho"]
                for istate in range(spf_coef_diff.nstate):
                    for idof in range(spf_coef_diff.ndof):
                        rho = mfop_rho[(istate, istate)][idof]
                        diff_bra = np.conj(spf_coef_diff[istate][idof])
                        diff_ket = spf_coef_diff[istate][idof]
                        err_spf += np.einsum(
                            "kp,kp",
                            diff_bra,
                            np.einsum("kl,lp->kp", rho, diff_ket),
                        )
                # FASTER?                    err_spf += np.sum(diff_bra * (rho @ diff_ket))
                err_spf = (
                    err_spf.real + 1.0e-16
                )  # in order to escape zero division
                assert not err_spf < 0.0
                if err_spf < const.tol_CMF * 2.0:
                    stepsize_next = stepsize * min(
                        1.5, pow((const.tol_CMF * 2.0) / err_spf, 0.25)
                    )
                    if stepsize > const.max_stepsize:
                        stepsize_next = const.max_stepsize
                    break
                else:
                    stepsize *= (
                        pow((const.tol_CMF * 2.0) / err_spf, 0.25) * 0.7
                    )  # Beck's recommend
                    if stepsize > const.max_stepsize:
                        stepsize_next = const.max_stepsize

            """(4) propagate SPF(t=h/2->h) with MFOP(t=h/2)"""
            if const.verbose == 4:
                helper._ElpTime.spf -= time()
            for _ in range(nstep_rk5):
                spf_coef_trial.propagate(
                    stepsize / 2 / nstep_rk5,
                    self.ints_prim,
                    mfop_spf_halfstep,
                    matH,
                )
            if const.verbose == 4:
                helper._ElpTime.spf += time()

            """construct IntsSPF(t=h)"""
            if const.verbose == 4:
                helper._ElpTime.itrf -= time()
            ints_spf_fullstep = SPFInts(self.ints_prim, spf_coef_trial)
            if const.verbose == 4:
                helper._ElpTime.itrf += time()

            """(5) back propagation CI(t=h/2->) with IntsSPF(t=h)"""
            if const.verbose == 4:
                helper._ElpTime.ci -= time()
            ci_coef_approx = deepcopy(ci_coef_trial)
            ci_coef_approx.propagate(-stepsize / 2, ints_spf_fullstep, matH)
            if const.verbose == 4:
                helper._ElpTime.ci += time()

            """Error estimation for CI"""
            err_ci = 0.25 * self.ci_coef.distance(ci_coef_approx) ** 2
            assert not err_ci < 0.0
            if const.tol_CMF < 1e-13 and type(self.ci_coef) is MPSCoefSoP:
                break
            else:
                if (err_ci + err_spf) < const.tol_CMF * 2.0:
                    stepsize_next = stepsize * min(
                        1.5,
                        pow((const.tol_CMF * 2.0) / (err_spf + err_ci), 0.25),
                    )
                    if stepsize > const.max_stepsize:
                        stepsize_next = const.max_stepsize
                    break
                else:
                    stepsize *= (
                        pow(const.tol_CMF / (err_spf + err_ci), 0.25) * 0.7
                    )  # Beck's recommend
                    if stepsize > const.max_stepsize:
                        stepsize_next = const.max_stepsize

        """(6) propagation CI(t=h/2->h) with IntsSPF(t=h)"""
        if const.verbose == 4:
            helper._ElpTime.ci -= time()
        ci_coef_trial.propagate(stepsize / 2, ints_spf_fullstep, matH)
        if const.verbose == 4:
            helper._ElpTime.ci += time()

        self.spf_coef = spf_coef_trial
        self.ci_coef = ci_coef_trial
        spf_occ = [
            mfop_spf["rho"][(istate, istate)] for istate in range(nstate)
        ]

        return (g, spf_occ, stepsize, stepsize_next)
