"""
Matrix Product State (MPS) Coefficient MixIn Class
"""

from __future__ import annotations

import copy
import itertools
import math
from abc import ABC, abstractmethod
from functools import partial
from logging import getLogger
from time import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as linalg
from opt_einsum import contract

import pytdscf._helper as helper
from pytdscf import _integrator
from pytdscf._const_cls import const
from pytdscf._contraction import (
    multiplyH_MPS_direct,
    multiplyH_MPS_direct_MPO,
    multiplyK_MPS_direct,
    multiplyK_MPS_direct_MPO,
)
from pytdscf._site_cls import SiteCoef
from pytdscf._spf_cls import SPFInts
from pytdscf.hamiltonian_cls import (
    HamiltonianMixin,
    PolynomialHamiltonian,
    TensorHamiltonian,
)
from pytdscf.model_cls import Model

logger = getLogger("main").getChild(__name__)


class MPSCoef(ABC):
    r"""Matrix Product State Coefficient MixIn Class

    .. math::
       a_{\tau_1}^{j_1} a_{\tau_1\tau_2}^{j_2} \cdots a_{\tau_{f-1}}^{j_f}

    Attributes:
       lattice_info_states (List[LatticeInfo]) : Lattice Information of each electronic states
       superblock_states (List[List[SiteCoef]]) : Super Blocks (Tensor Cores) of each electronic states
       nstate (int) : Number of electronic states
       nsite (int) : Number of sites (tensor cores)
       ndof_per_sites (List[int]) : Number of DOFs per site. Defaults to all 1.
       with_matH_general (bool) : General Hamiltonian term exist or not
       site_is_dof (bool) : ``ndof_per_sites`` is all 1 or not. Defaults to ``True``.
    """

    superblock_states: list[list[SiteCoef]]
    lattice_info_states: list[LatticeInfo]

    def __init__(self):
        self.op_sys_sites_dipo: list | None = None
        self.ints_site_dipo: (
            dict[tuple[int, int], dict[str, list[np.ndarray] | list[jax.Array]]]
            | None
        ) = None
        self.ints_site: (
            dict[tuple[int, int], dict[str, list[np.ndarray] | list[jax.Array]]]
            | None
        ) = None
        self.op_sys_sites: list | None = None

    @classmethod
    @abstractmethod
    def alloc_random(cls, model: Model) -> MPSCoef:
        """Allocate MPS Coefficient randomly.

        Args:
            model (Model) : Your input information

        Returns:
            MPSCoef : MPSCoef class object

        """
        pass

    @staticmethod
    def _get_initial_condition(model):
        """Get initial condition of MPS Coefficient (common to all models)

        Args:
            model (Model) : Your input information

        Returns:
            Tuple[int, List[LatticeInfo], List[List[SiteCoef]], np.ndarray, float, int] : \
                Number of electronic states,\
                Lattice Information of each electronic states, \
                Super Blocks (Tensor Cores) of each electronic states, \
                Electronic state weight, Vibration state weight, \
                MPS bond dimension.

        """
        nstate = model.get_nstate()
        ndof = model.get_ndof()
        if (m_aux_max := model.m_aux_max) is None:
            m_aux_max = 10**9
        lattice_info_states = []
        superblock_states = []

        if (weight_estate := model.init_weight_ESTATE) is None:
            weight_estate = np.array([1.0] + [0.0] * (nstate - 1))
        else:
            weight_estate = np.array(weight_estate) / sum(weight_estate)
            if len(weight_estate) != nstate:
                raise ValueError(
                    "The length of weight_estate must be equal to nstate."
                )
            if min(weight_estate) < 0.0:
                raise ValueError(
                    "The elements of weight_estate must be positive."
                )
            if max(weight_estate) > 1.0:
                raise ValueError(
                    "The elements of weight_estate must be less than 1.0."
                )

        if (weight_vib := model.init_weight_VIBSTATE) is None:
            weight_vib = [
                [
                    [1.0]
                    + [0.0 for _ in range(1, model.get_nspf(istate, idof))]
                    for idof in range(ndof)
                ]
                for istate in range(nstate)
            ]
        else:
            if len(weight_vib) != nstate:
                raise ValueError(
                    "The length of weight_vib must be equal to nstate."
                )
            if len(weight_vib[0]) != ndof:
                raise ValueError(
                    "The length of weight_vib[0] must be equal to ndof"
                    + f" But len(weight_vib[0]) = {len(weight_vib[0])} != {ndof}"
                )
            for istate in range(nstate):
                for idof in range(ndof):
                    if not isinstance(weight_vib[istate][idof][0], float):
                        raise TypeError(
                            "The elements of weight_vib must be float"
                        )
                    nspf = model.get_nspf(istate, idof)
                    if (
                        max(weight_vib[istate][idof]) > nspf - 1
                        or min(weight_vib[istate][idof]) < 0
                    ):
                        raise ValueError(
                            "The elements of weight_vib must be less than nspf - 1 and greater than 0."
                        )

        for istate in range(nstate):
            logger.info(
                f"Initial MPS: {istate}-state with weights {weight_estate[istate]}"
            )
            for idof in range(ndof):
                if const.verbose > 2:
                    logger.info(
                        f"Initial MPS: {istate}-state {idof}-mode with weight "
                        + f"{np.array(weight_vib[istate][idof]) / np.linalg.norm(weight_vib[istate][idof])}"
                    )
        return (
            nstate,
            lattice_info_states,
            superblock_states,
            weight_estate,
            weight_vib,
            m_aux_max,
        )

    @abstractmethod
    def get_matH_sweep(self, matH):
        pass

    @abstractmethod
    def get_matH_tdh(self, matH, op_block_cas):
        pass

    @abstractmethod
    def get_matH_cas(self, matH, ints_spf: SPFInts):
        pass

    @abstractmethod
    def get_ints_site(
        self, ints_spf: SPFInts, onesite_name: str = "onesite"
    ) -> dict[tuple[int, int], dict[str, list[np.ndarray] | list[jax.Array]]]:
        """Get integral between p-site bra amd p-site ket in all states pair

        Args:
            ints_spf (SPFInts): SPF integrals
            onesite_name (str, optional) : Defaults to 'onesite'.

        Returns:
            Dict[Tuple[int,int],Dict[str, np.ndarray]]: Site integrals
        """
        pass

    @abstractmethod
    def construct_mfop_MPS(
        self,
        psite: int,
        op_sys: dict[tuple[int, int], dict[str, np.ndarray | jax.Array]],
        op_env: dict[tuple[int, int], dict[str, np.ndarray | jax.Array]],
        matH_cas,
        left_is_sys: bool,
        ints_spf: SPFInts | None = None,
        mps_coef_ket=None,
    ):
        pass

    @abstractmethod
    def construct_mfop(self, ints_spf: SPFInts, matH):
        pass

    @abstractmethod
    def construct_mfop_along_sweep(
        self,
        ints_site: dict[
            tuple[int, int], dict[str, list[np.ndarray] | list[jax.Array]]
        ],
        matH_cas,
        *,
        left_is_sys: bool,
    ):
        pass

    @abstractmethod
    def construct_mfop_along_sweep_TEMP4DIPOLE(
        self,
        ints_site: dict[
            tuple[int, int], dict[str, list[np.ndarray] | list[jax.Array]]
        ],
        matO_cas,
        *,
        left_is_sys: bool,
        mps_coef_ket=None,
    ):
        pass

    @abstractmethod
    def construct_op_zerosite(
        self,
        superblock_states: list[list[SiteCoef]],
        matH_cas: HamiltonianMixin | None = None,
    ) -> dict[tuple[int, int], dict[str, np.ndarray | jax.Array]]:
        """initialize op_block_psites
        Args:
            superblock_states (List[List[SiteCoef]]) : Super Blocks (Tensor Cores) of each electronic states
            matH_cas (Optional[HamiltonianMixin], optional): Hamiltonian. Defaults to None.

        Returns:
            Dict[Tuple[int,int], Dict[str, np.ndarray]] : block operator. \
        """
        pass

    @abstractmethod
    def renormalize_op_psite(
        self,
        psite: int,
        superblock_states: list[list[SiteCoef]],
        op_block_states: dict[
            tuple[int, int], dict[str, int | np.ndarray | jax.Array]
        ],
        ints_site: dict[
            tuple[int, int], dict[str, list[np.ndarray] | list[jax.Array]]
        ],
        matH,
        left_is_sys: bool,
        superblock_states_unperturb=None,
    ) -> dict[tuple[int, int], dict[str, int | np.ndarray | jax.Array]]:
        pass

    @abstractmethod
    def operators_for_superH(
        self,
        psite: int,
        op_sys: dict[tuple[int, int], dict[str, np.ndarray | jax.Array]],
        op_env: dict[tuple[int, int], dict[str, np.ndarray | jax.Array]],
        ints_site: dict[
            tuple[int, int], dict[str, list[np.ndarray] | list[jax.Array]]
        ],
        matH_cas: HamiltonianMixin,
        left_is_sys: bool,
    ) -> list[
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
    ]:
        """ LCR operator

        prepare operators for multiplying the full-matrix PolynomialHamiltonian on-the-fly

        Args:
            psite (int): site index on "C"
            op_sys (Dict[Tuple[int,int], Dict[str, np.ndarray | jax.Array]]): System operator
            op_env (Dict[Tuple[int,int], Dict[str, np.ndarray | jax.Array]]): Environment operator
            ints_site (Dict[Tuple[int,int],Dict[str, np.ndarray]]): Site integral
            matH_cas (PolynomialHamiltonian) : Hamiltonian
            left_is_sys (bool): Whether left block is System

        Returns:
            List[List[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]]: \
                [i-bra-state][j-ket-state]['q'] =  (op_l, op_c, op_r)
        """
        pass

    @abstractmethod
    def operators_for_superK(
        self,
        psite: int,
        op_sys: dict[tuple[int, int], dict[str, np.ndarray | jax.Array]],
        op_env: dict[tuple[int, int], dict[str, np.ndarray | jax.Array]],
        matH_cas: HamiltonianMixin,
        left_is_sys: bool,
    ) -> list[
        list[
            dict[
                str,
                tuple[
                    int | np.ndarray | jax.Array, int | np.ndarray | jax.Array
                ],
            ]
        ]
    ]:
        """ LsR operator

        construct full-matrix Kamiltonian

        Args:
            psite (int): site index on "C"
            op_sys (Dict[Tuple[int,int], Dict[str, np.ndarray | jax.Array]]): System operator
            op_env (Dict[Tuple[int,int], Dict[str, np.ndarray | jax.Array]]): Environment operator
            matH_cas (PolynomialHamiltonian) : Hamiltonian
            left_is_sys (bool): Whether left block is System

        Returns:
            List[List[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]]: \
                [i-bra-state][j-ket-state]['q'] =  (op_l, op_r)
        """
        pass

    def is_psite_canonical(self, psite: int) -> bool:
        """check if mps = L..L(p-1)C(p)R(p+1)..R

        Args:
            psite (int): guess of "C" site

        Returns:
            bool : Correct or Incorrect
        """
        for superblock in self.superblock_states:
            for isite, site_coef in enumerate(superblock):
                expected_gauge = (
                    "L" if isite < psite else "C" if isite == psite else "R"
                )
                if site_coef.gauge != expected_gauge:
                    return False
        return True

    def assert_psite_canonical(self, psite: int):
        assert self.is_psite_canonical(
            psite
        ), "wrong gauge status. It's assumed to be L..L(p-1)C(p)R(p+1)..R in superblock"

    def apply_dipole(self, ints_spf: SPFInts, ci_coef_init, matO) -> float:
        if (not const.standard_method) or (
            const.standard_method and self.ints_site_dipo is None
        ):
            self.ints_site_dipo = self.get_ints_site(ints_spf)
            matO_cas = (
                matO
                if "enable_tdh_dofs" not in const.keys
                else self.get_matH_cas(matO, ints_spf)
            )
            self.matO_sweep = self.get_matH_sweep(matO_cas)
        self.apply_dipole_along_sweep(
            ci_coef_init, self.ints_site_dipo, self.matO_sweep, left_is_sys=True
        )
        norm = self.apply_dipole_along_sweep(
            ci_coef_init,
            self.ints_site_dipo,
            self.matO_sweep,
            left_is_sys=False,
        )
        return norm

    def propagate(
        self, stepsize: float, ints_spf: SPFInts | None, matH: HamiltonianMixin
    ):
        if const.verbose == 4:
            helper._ElpTime.ci_etc -= time()
        if (
            const.doTDHamil
            or (not const.standard_method)
            or (
                not const.doTDHamil
                and const.standard_method
                and self.ints_site is None
            )
        ):
            assert isinstance(ints_spf, SPFInts)
            self.ints_site = self.get_ints_site(ints_spf)
            matO_cas = (
                matH
                if "enable_tdh_dofs" not in const.keys
                else self.get_matH_cas(matH, ints_spf)
            )
            self.matH_sweep = self.get_matH_sweep(matO_cas)
        if const.verbose == 4:
            helper._ElpTime.ci_etc += time()
        self.propagate_along_sweep(
            self.ints_site, self.matH_sweep, stepsize, left_is_sys=True
        )
        self.propagate_along_sweep(
            self.ints_site, self.matH_sweep, stepsize, left_is_sys=False
        )

    def construct_mfop_TEMP4DIPOLE(self, ints_spf: SPFInts, matO, ci_coef_ket):
        mps_copy_bra = copy.deepcopy(self)
        mps_copy_ket = copy.deepcopy(ci_coef_ket)
        ints_site = mps_copy_bra.get_ints_site(ints_spf)
        matO_cas = (
            matO
            if "enable_tdh_dofs" not in const.keys
            else self.get_matH_cas(matO, ints_spf)
        )
        matO_sweep = self.get_matH_sweep(matO_cas)
        if self.is_psite_canonical(0):
            mfop = mps_copy_bra.construct_mfop_along_sweep_TEMP4DIPOLE(
                ints_site,
                matO_sweep,
                left_is_sys=True,
                mps_coef_ket=mps_copy_ket,
            )
        elif self.is_psite_canonical(self.nsite - 1):
            mfop = mps_copy_bra.construct_mfop_along_sweep_TEMP4DIPOLE(
                ints_site,
                matO_sweep,
                left_is_sys=False,
                mps_coef_ket=mps_copy_ket,
            )
        else:
            raise AssertionError("The MPS must be canonical in terminal sites.")

        return mfop

    def distance(self, other: MPSCoef):
        return distance_MPS(self, other)

    def expectation(
        self, ints_spf: SPFInts, matOp: HamiltonianMixin, psite: int = 0
    ) -> complex | float:
        """Get Expectation Value at "C" = p-site

        Args:
            ints_spf (SPFInts): SPF integral
            matOp (HamiltonianMixin): Operator
            psite (Optional[int], optional): When calculate expectation value. Defaults to 0.

        Returns:
            complex or float : expectation value
        """
        self.assert_psite_canonical(psite)
        superblock_states = self.superblock_states
        # - inefficient impl-#
        if hasattr(matOp, "onesite_name"):
            ints_site = self.get_ints_site(ints_spf, matOp.onesite_name)
        else:
            ints_site = self.get_ints_site(ints_spf)
        matOp = self.get_matH_sweep(matOp)
        op_sys = self.construct_op_zerosite(superblock_states, matOp)
        op_env = self.construct_op_sites(
            superblock_states, ints_site, False, matOp
        )[psite]
        op_lcr = self.operators_for_superH(
            psite, op_sys, op_env, ints_site, matOp, True
        )

        """concatenate PolynomialHamiltonian & Coefficients over the electronic states"""
        matC_states: list[np.ndarray] | list[jax.Array]
        if const.use_jax:
            matC_states = [
                superblock[psite].data for superblock in superblock_states
            ]  # type: ignore
        else:
            matC_states = [
                np.array(superblock[psite]) for superblock in superblock_states
            ]
        if isinstance(matOp, PolynomialHamiltonian):
            multiplyH = multiplyH_MPS_direct(op_lcr, matC_states, matOp)
        else:
            multiplyH = multiplyH_MPS_direct_MPO(op_lcr, matC_states, matOp)  # type: ignore

        expectation_value = _integrator.expectation_Op(
            matC_states,  # type: ignore
            multiplyH,
            matC_states,  # type: ignore
        )
        return expectation_value

    def autocorr(self, ints_spf: SPFInts, psite: int = 0) -> complex:
        r"""Get auto-correlation value

        .. math ::
           a(t) = \langle\Psi(t)|\Psi(0)\rangle = \langle \Psi(t/2)^\ast|\Psi(t/2) \rangle

        Args:
            ints_spf (SPFInts): SPF integral
            psite (int, optional): "C" site index. Defaults to 0.

        Returns:
            complex: auto-correlation value
        """
        nstate = len(self.superblock_states)
        if const.use_jax and const.standard_method and nstate == 1:
            # If the site basis is not orthogonal, the following code will not work.
            cores = self.superblock_states[0]
            return complex(_autocorr_single_state_jax(cores))

        self.assert_psite_canonical(psite)
        superblock_states = self.superblock_states
        nstate = len(superblock_states)
        # - inefficient impl-#
        ints_site = self.get_ints_site(ints_spf)
        op_sys = self.construct_op_zerosite(superblock_states)
        op_env = self.construct_op_sites(superblock_states, ints_site, False)[
            psite
        ]
        op_lcr = self.operators_for_autocorr(
            psite, op_sys, op_env, ints_site, nstate, True
        )

        """concatenate PolynomialHamiltonian & Coefficients over the electronic states"""
        matC_states: list[np.ndarray] | list[jax.Array]
        if const.use_jax:
            matC_states = [
                superblock[psite].data for superblock in superblock_states
            ]  # type: ignore
        else:
            matC_states = [
                np.array(superblock[psite]) for superblock in superblock_states
            ]
        multiplyH = multiplyH_MPS_direct(op_lcr, matC_states)

        psivec = multiplyH.stack(matC_states)
        sigvec = multiplyH.stack(multiplyH.dot_autocorr(matC_states))

        autocorr_tdh = 1.0 + 0.0j
        if "enable_tdh_dofs" in const.keys:
            for statepair in itertools.product(range(nstate), repeat=2):
                autocorr_tdh *= complex(
                    np.prod(
                        [
                            ints_spf["auto"][statepair][idof]
                            for idof in self.dofs_tdh
                        ]
                    )
                )

        return complex(np.inner(psivec, sigvec)) * autocorr_tdh

    def pop_states(self, psite: int = 0) -> list[float]:
        """Get population for each MPS states in p-site canonical form

        Args:
            psite (Optional[int], optional): Defaults to 0.

        Returns:
            List[float]: populations
        """
        if psite != 0:
            raise NotImplementedError
        self.assert_psite_canonical(psite)
        if const.use_jax:
            populations = [
                float((jnp.linalg.norm(superblock[psite].data).real ** 2))
                for superblock in self.superblock_states
            ]
        else:
            populations = [
                superblock[psite].norm() ** 2
                for superblock in self.superblock_states
            ]
        return populations

    def norm(self, psite: int = 0) -> float:
        """Get All MPS norm in p-site-canonical form

        Args:
            psite (int, optional): Defaults to 0.

        Returns:
            float : norm
        """
        norm = math.sqrt(np.sum(self.pop_states(psite)))
        return norm

    def apply_dipole_along_sweep(
        self,
        mps_coef_init: MPSCoef,
        ints_site: dict[
            tuple[int, int], dict[str, list[np.ndarray] | list[jax.Array]]
        ],
        matO_cas,
        left_is_sys: bool,
    ) -> float:
        superblock_states = self.superblock_states
        superblock_states_unperturb = mps_coef_init.superblock_states

        nsite = len(superblock_states[0])

        op_sys = self.construct_op_zerosite(superblock_states, matO_cas)
        if self.op_sys_sites_dipo is None:
            op_env_sites = self.construct_op_sites(
                superblock_states,
                ints_site,
                not left_is_sys,
                matO_cas,
                superblock_states_unperturb,
            )
        else:
            if left_is_sys:
                # op_env_sites = copy.deepcopy(self.op_sys_sites_dipo)[::-1]
                op_env_sites = self.op_sys_sites_dipo[::-1]
            else:
                # op_env_sites = copy.deepcopy(self.op_sys_sites_dipo)
                op_env_sites = self.op_sys_sites_dipo[:]

        if const.standard_method:
            self.op_sys_sites_dipo = [op_sys]

        psites_sweep = (
            list(range(nsite)) if left_is_sys else list(range(nsite))[::-1]
        )
        for psite in psites_sweep:
            op_lcr = self.operators_for_superH(
                psite,
                op_sys,
                op_env_sites[psite],
                ints_site,
                matO_cas,
                left_is_sys,
            )

            norm = apply_superOp_direct(
                psite,
                superblock_states,
                op_lcr,
                matO_cas,
                superblock_states_unperturb,
            )

            if psite != psites_sweep[-1]:
                superblock_transLCR_psite(psite, superblock_states, left_is_sys)
                superblock_transLCR_psite(
                    psite, superblock_states_unperturb, left_is_sys
                )
                op_sys = self.renormalize_op_psite(
                    psite,
                    superblock_states,
                    op_sys,  # type: ignore
                    ints_site,
                    matO_cas,
                    left_is_sys,
                    superblock_states_unperturb,
                )
                if self.op_sys_sites_dipo is not None:
                    self.op_sys_sites_dipo.append(op_sys)
        return norm

    def propagate_along_sweep(
        self,
        ints_site: dict[
            tuple[int, int], dict[str, list[np.ndarray] | list[jax.Array]]
        ],
        matH_cas,
        stepsize,
        *,
        left_is_sys: bool,
    ):
        superblock_states = self.superblock_states
        nsite = len(superblock_states[0])

        if const.verbose == 4:
            helper._ElpTime.ci_rnm -= time()
        op_sys = self.construct_op_zerosite(superblock_states, matH_cas)
        if self.op_sys_sites is None:
            # If t==0 or Include SPF or time-dependent Hamiltonian
            op_env_sites = self.construct_op_sites(
                superblock_states, ints_site, not left_is_sys, matH_cas
            )
            if left_is_sys:
                op_env_sites = op_env_sites[::-1]
        else:
            op_env_sites = self.op_sys_sites[:]
        # When left_is_sys,
        # [sys0, sys1, ..., sysN-1, envM-1, ..., env1, env0]
        # When not left_is_sys,
        # [env0, env1, ..., envM-1, sysN-1, ..., sys1, sys0]

        if not const.standard_method or const.doTDHamil:
            self.op_sys_sites = [op_sys]
        else:
            self.op_sys_sites = None

        if const.verbose == 4:
            helper._ElpTime.ci_rnm += time()

        psites_sweep = (
            list(range(nsite)) if left_is_sys else list(range(nsite))[::-1]
        )
        for psite in psites_sweep:
            if const.verbose == 4:
                helper._ElpTime.ci_etc -= time()
            helper._Debug.site_now = psite
            op_env = op_env_sites.pop()
            op_lcr = self.operators_for_superH(
                psite,
                op_sys,
                op_env,
                ints_site,
                matH_cas,
                left_is_sys,
            )
            if const.verbose == 4:
                helper._ElpTime.ci_etc += time()

            if const.verbose == 4:
                helper._ElpTime.ci_exp -= time()
            self.exp_superH_propagation_direct(
                psite, superblock_states, op_lcr, matH_cas, stepsize
            )
            if const.verbose == 4:
                helper._ElpTime.ci_exp += time()

            if psite != psites_sweep[-1]:
                if const.verbose == 4:
                    helper._ElpTime.ci_rnm -= time()
                svalues, op_sys = self.trans_next_psite_LSR(
                    psite,
                    self.superblock_states,
                    op_sys,
                    ints_site,
                    matH_cas,
                    left_is_sys,
                )
                if const.verbose == 4:
                    helper._ElpTime.ci_rnm += time()

                if const.verbose == 4:
                    helper._ElpTime.ci_etc -= time()
                op_lr = self.operators_for_superK(
                    psite, op_sys, op_env, matH_cas, left_is_sys
                )
                if const.verbose == 4:
                    helper._ElpTime.ci_etc += time()

                if const.verbose == 4:
                    helper._ElpTime.ci_exp -= time()
                self.exp_superK_propagation_direct(
                    psite,
                    superblock_states,
                    op_lr,
                    matH_cas,
                    svalues,
                    stepsize,
                    left_is_sys,
                )
                if const.verbose == 4:
                    helper._ElpTime.ci_exp += time()
                if self.op_sys_sites is not None:
                    self.op_sys_sites.append(op_sys)

    def exp_superH_propagation_direct(
        self,
        psite: int,
        superblock_states: list[list[SiteCoef]],
        op_lcr: list[
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
        matH_cas: HamiltonianMixin,
        stepsize: float,
    ):
        """concatenate PolynomialHamiltonian & coefficients"""
        matC_states: list[np.ndarray] | list[jax.Array]
        if const.use_jax:
            matC_states = [
                superblock[psite].data for superblock in superblock_states
            ]  # type: ignore
        else:
            matC_states = [
                np.array(superblock[psite]) for superblock in superblock_states
            ]

        """exponentiation PolynomialHamiltonian"""
        if isinstance(matH_cas, PolynomialHamiltonian):
            multiplyH = multiplyH_MPS_direct(op_lcr, matC_states, matH_cas)
        else:
            assert isinstance(matH_cas, TensorHamiltonian)
            multiplyH = multiplyH_MPS_direct_MPO(op_lcr, matC_states, matH_cas)  # type: ignore

        if not const.doRelax:
            matC_states_new = _integrator.short_iterative_lanczos(
                -1.0j * stepsize / 2, multiplyH, matC_states, const.thresh_exp
            )

        elif const.doRelax == "improved":
            matC_states_new = _integrator.matrix_diagonalize_lanczos(
                multiplyH, matC_states
            )
            norm = get_C_sval_states_norm(matC_states_new)
            matC_states_new = [x / norm for x in matC_states_new]  # type: ignore

        else:
            matC_states_new = _integrator.short_iterative_lanczos(
                -1.0 * stepsize / 2, multiplyH, matC_states, const.thresh_exp
            )
            norm = get_C_sval_states_norm(matC_states_new)
            matC_states_new = [x / norm for x in matC_states_new]  # type: ignore

        """update(over-write) matC(psite)"""
        for istate, superblock in enumerate(superblock_states):
            superblock[psite] = SiteCoef(matC_states_new[istate], "C")

    def exp_superK_propagation_direct(
        self,
        psite: int,
        superblock_states: list[list[SiteCoef]],
        op_lr: list[
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
        matH_cas: HamiltonianMixin,
        svalues: list[np.ndarray] | list[jax.Array],
        stepsize: float,
        left_is_sys: bool = True,
    ):
        """concatenate PolynomialHamiltonian & coefficients"""
        svalues_states = svalues

        """exponentiation PolynomialHamiltonian"""
        if isinstance(matH_cas, PolynomialHamiltonian):
            multiplyK = multiplyK_MPS_direct(op_lr, matH_cas, svalues_states)
        else:
            assert isinstance(matH_cas, TensorHamiltonian)
            multiplyK = multiplyK_MPS_direct_MPO(
                op_lr,
                matH_cas,
                svalues_states,
            )

        if not const.doRelax:
            svalues_states_new = _integrator.short_iterative_lanczos(
                +1.0j * stepsize / 2,
                multiplyK,
                svalues_states,
                const.thresh_exp,
            )

        elif const.doRelax == "improved":
            svalues_states_new = svalues_states
        else:
            svalues_states_new = _integrator.short_iterative_lanczos(
                +1.0 * stepsize / 2, multiplyK, svalues_states, const.thresh_exp
            )
            norm = get_C_sval_states_norm(svalues_states_new)
            svalues_states_new = [x / norm for x in svalues_states_new]  # type: ignore

        """over-write sval"""
        for istate, superblock in enumerate(superblock_states):
            matC: np.ndarray | jax.Array
            if left_is_sys:
                """sval x R(p+1) -> C(p+1)"""
                matL = superblock[psite]
                matR = superblock[psite + 1]
                assert matL.gauge == "L"
                assert matR.gauge == "R"
                sval = svalues_states_new[istate]
                if const.use_jax:
                    matC = jnp.einsum("ij,jbc->ibc", sval, matR.data)
                else:
                    matC = np.tensordot(sval, matR, axes=1)
                superblock[psite + 1] = SiteCoef(matC, "C")
            else:
                """L(p-1) x sval -> C(p-1)"""
                matR = superblock[psite]
                matL = superblock[psite - 1]
                assert matR.gauge == "R"
                assert matL.gauge == "L"
                sval = svalues_states_new[istate]
                if const.use_jax:
                    matC = jnp.einsum("ijk,kb->ijb", matL.data, sval)
                else:
                    matC = np.tensordot(matL, sval, axes=1)
                superblock[psite - 1] = SiteCoef(matC, "C")

    def _get_normalized_reduced_density(
        self, istate: int, remain_nleg: tuple[int, ...]
    ) -> np.ndarray | jax.Array:
        """
        Wavefunction is written by

        |Ψ> = C[0] R[1] R[2] ... R[f-1]

        if dof_pair = (0, 2), then the reduced density matrix is

        |ρ>[0, 2] = Tr_{1, 3, ..., f-1} |Ψ><Ψ|

        Contraction of tensor cores should be executed from right to left.

        """
        assert self.is_psite_canonical(0)
        cores = [
            self.superblock_states[istate][isite].data
            for isite in range(len(remain_nleg))
        ]
        if isinstance(cores[0], jax.Array):
            return _get_normalized_reduced_density_jax(cores, remain_nleg)
        else:
            core = cores.pop()
            nleg = remain_nleg[-1]
            if nleg == 0:
                raise ValueError("The number of legs must be greater than 0.")
            elif nleg == 1:
                """
                i‾|‾k k‾|‾˙˙˙‾|
                  j     |     |
                a_|_k k_|_..._|
                """
                subscript = "ijk,ajk->iaj"
            elif nleg == 2:
                """
                i‾|‾k k‾|‾˙˙˙‾|
                  j     |     |
                  l     |     |
                a_|_k k_|_..._|
                """
                subscript = "ijk,alk->iajl"
            else:
                raise ValueError("The number of legs must be less than 3.")
            density = np.einsum(subscript, np.conj(core), core)
            isite = len(remain_nleg) - 1
            while cores:
                isite -= 1
                nleg = remain_nleg[isite]
                core = cores.pop()
                if nleg == 2:
                    # when dof_pair contains one isite
                    """
                    l‾|‾i i‾|‾˙˙˙‾|
                      n     ...   |
                      m     ...   |
                    b_|_a a_|_..._|
                    """
                    subscript = "lmi,bna,ia...->lbmn..."
                elif nleg == 1:
                    """
                    l‾|‾i i‾|‾˙˙˙‾|
                      m     ...   |
                    b_|_a a_|_..._|
                    """
                    subscript = "lmi,bma,ia...->lbm..."
                elif nleg == 0:
                    subscript = "lmi,bma,ia...->lb..."
                else:
                    raise ValueError("The number of legs must be less than 3.")
                density = contract(subscript, np.conj(core), core, density)
            assert isite == 0
            return density[0, 0, ...]

    def get_reduced_densities(
        self, remain_nleg: tuple[int, ...]
    ) -> list[np.ndarray]:
        reduced_densities = []
        nstate = len(self.superblock_states)
        for istate in range(nstate):
            _reduced_density = self._get_normalized_reduced_density(
                istate, remain_nleg
            )
            if isinstance(_reduced_density, jax.Array):
                reduced_density = np.array(_reduced_density)
            elif isinstance(_reduced_density, np.ndarray):
                reduced_density = _reduced_density
            else:
                raise ValueError("The type of reduced_density is invalid.")
            reduced_densities.append(reduced_density)
        return reduced_densities

    def get_CI_coef_state(
        self,
        J: tuple[int, ...] | None = None,
        trans_arrays: list[np.ndarray] | list[jax.Array] | None = None,
        istate: int = 0,
    ):
        """Get CI coefficient of given J=(j₁,j₂,...,j_f) from CI coefficients of all states

        **Only MPS-SM is supported**

        Args:
            J (Tuple[int, ...]) : The CI coefficient index (tensor legs). Default is None.
                One has to set either J or trans_arrays.
            trans_arrays (List[np.ndarray]) : transformation array for each DOFs. Default is None.
            istate (int) : state index. Default is 0.
        """

        ndof = self.lattice_info_states[istate].nsite
        nprims = self.lattice_info_states[istate].ndof_per_sites
        _trans_arrays: list[np.ndarray] | list[jax.Array]
        if trans_arrays is None:
            if J is None:
                raise ValueError("Either `J` or `trans_arrays` must be set.")
            _trans_arrays = []
            for idof in range(ndof):
                _trans_array = np.zeros(nprims[idof], dtype=complex)
                _trans_array[J[idof]] = 1.0
                _trans_arrays.append(_trans_array)  # type: ignore
        else:
            _trans_arrays = trans_arrays
            if len(trans_arrays) != ndof:
                raise ValueError(
                    "The length of `trans_arrays` must be equal to the number of DOFs."
                )

        mps = self.superblock_states[istate]

        retval = None
        for core, trans_array in zip(mps, _trans_arrays, strict=True):
            if isinstance(core.data, jax.Array):
                if retval is None:
                    retval = jnp.einsum("ijk,j->k", core.data, trans_array)
                else:
                    retval = jnp.einsum(
                        "i,ijk,j->k", retval, core.data, trans_array
                    )
            else:
                if retval is None:
                    retval = np.einsum("ijk,j->k", core, trans_array)
                else:
                    retval = np.einsum(
                        "i,ik->k",
                        retval,
                        np.einsum("ijk,j->ik", core, trans_array),
                    )
        assert isinstance(retval, np.ndarray | jax.Array)
        return complex(retval[0])

    def construct_op_sites(
        self,
        superblock_states: list[list[SiteCoef]],
        ints_site: dict[
            tuple[int, int], dict[str, list[np.ndarray] | list[jax.Array]]
        ],
        set_op_left: bool,
        matH_cas: HamiltonianMixin | None = None,
        superblock_states_unperturb=None,
    ) -> list[dict[tuple[int, int], dict[str, np.ndarray | jax.Array]]]:
        """Construct Environment Operator

        Args:
            superblock_states (List[List[SiteCoef]]) : Super Blocks (Tensor Cores) of each electronic states
            ints_site (Dict[Tuple[int,int],Dict[str, np.ndarray]]): Site integral
            set_op_left (bool) : Set environment operator from left side
            matH_cas (HamiltonianMixin) : Hamiltonian

        Returns:
            List[Dict[Tuple[int,int], Dict[str, np.ndarray | jax.Array]]] : Env. Operator
        """

        nsite = len(superblock_states[0])
        op_block_isites = [
            self.construct_op_zerosite(superblock_states, matH_cas)
        ]

        psites_sweep = (
            list(range(nsite)) if set_op_left else list(range(nsite))[::-1]
        )
        for isite, psite in enumerate(psites_sweep[:-1]):
            """construct op_block for <L(0)L(1)...L(isite)|Op|L(0)L(1)...L(isite)>"""
            op_block_isites.append(
                self.renormalize_op_psite(
                    psite,
                    superblock_states,
                    op_block_isites[isite],  # type: ignore
                    ints_site,
                    matH_cas,
                    set_op_left,
                    superblock_states_unperturb,
                )
            )

        return op_block_isites if set_op_left else op_block_isites[::-1]

    def trans_next_psite_LSR(
        self,
        psite: int,
        superblock_states: list[list[SiteCoef]],
        op_sys: dict[tuple[int, int], dict[str, np.ndarray | jax.Array]],
        ints_site: dict[
            tuple[int, int], dict[str, list[np.ndarray] | list[jax.Array]]
        ],
        matH_cas,
        left_is_sys,
        superblock_states_ket=None,
        regularize=False,
    ):
        """..C(p) R(p+1).. -> ..L(p) sval R(p+1)"""

        def _transCR2LR_psite(superblock_states):
            svalues = []
            for superblock in superblock_states:
                if left_is_sys:
                    svalues.append(
                        superblock_transCR2LR_psite(
                            psite, superblock, regularize
                        )
                    )
                else:
                    svalues.append(
                        superblock_transLC2LR_psite(
                            psite, superblock, regularize
                        )
                    )
            return svalues

        svalues = _transCR2LR_psite(superblock_states)
        if superblock_states_ket:
            _ = _transCR2LR_psite(superblock_states_ket)
        op_sys_next = self.renormalize_op_psite(
            psite,
            superblock_states,
            op_sys,  # type: ignore
            ints_site,
            matH_cas,
            left_is_sys,
            superblock_states_ket,
        )

        return svalues, op_sys_next

    def trans_next_psite_LCR(
        self,
        psite: int,
        superblock_states: list[list[SiteCoef]],
        op_sys: dict[tuple[int, int], dict[str, np.ndarray | jax.Array]],
        ints_site: dict[
            tuple[int, int], dict[str, list[np.ndarray] | list[jax.Array]]
        ],
        matH_cas,
        left_is_sys: bool,
        superblock_states_ket=None,
    ):
        """..C(p) R(p+1).. -> ..L(p) C(p+1)"""

        def _transCR2LC_psite(superblock_states):
            for superblock in superblock_states:
                if left_is_sys:
                    superblock_transCR2LC_psite(psite, superblock)
                else:
                    superblock_transLC2CR_psite(psite, superblock)

        _transCR2LC_psite(superblock_states)
        if superblock_states_ket:
            _transCR2LC_psite(superblock_states_ket)

        op_sys_next = self.renormalize_op_psite(
            psite,
            superblock_states,
            op_sys,  # type: ignore
            ints_site,
            matH_cas,
            left_is_sys,
            superblock_states_ket,
        )
        return op_sys_next

    def operators_for_autocorr(
        self,
        psite: int,
        op_sys: dict[tuple[int, int], dict[str, np.ndarray | jax.Array]],
        op_env: dict[tuple[int, int], dict[str, np.ndarray | jax.Array]],
        ints_site: dict[
            tuple[int, int], dict[str, list[np.ndarray] | list[jax.Array]]
        ],
        nstate: int,
        left_is_sys: bool,
    ) -> list[
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
    ]:
        """auto correlation operator

        prepare operators for multiplying the full-matrix auto-correlation operator on-the-fly

        Args:
            psite (int): site index on "C"
            op_sys (Dict[Tuple[int,int], Dict[str, np.ndarray | jax.Array]]): System operator
            op_env (Dict[Tuple[int,int], Dict[str, np.ndarray | jax.Array]]): Environment operator
            ints_site (Dict[Tuple[int,int],Dict[str, np.ndarray | jax.Array]]): Site integral
            nstate (int): number of state
            left_is_sys (bool): Whether left block is System

        Returns:
            List[List[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]]: \
                [i-bra-state][j-ket-state]['auto'] =  (op_l, op_c, op_r)
        """
        op_lcr: list[
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
        ]
        op_lcr = [[None for j in range(nstate)] for i in range(nstate)]  # type: ignore
        for istate in range(nstate):
            statepair = (istate, istate)
            op_l_auto = (
                op_sys[statepair]["auto"]
                if left_is_sys
                else op_env[statepair]["auto"]
            )
            op_r_auto = (
                op_env[statepair]["auto"]
                if left_is_sys
                else op_sys[statepair]["auto"]
            )
            op_c_auto = ints_site[statepair]["auto"][psite]

            op_lcr[istate][istate] = {"auto": (op_l_auto, op_c_auto, op_r_auto)}

        return op_lcr


def _prod(arr: list[int] | np.ndarray) -> int:
    """Numpy.ndarray.prod induces overflow. This is alternative."""
    dum = 1
    for i in arr:
        dum *= i
    return dum


class LatticeInfo:
    """MPS Tensor Lattice Information class

    Args:
        nspf_list_sites (List[List[int]]) : Number of SPFs for each sites. \
            ``nspf_list_sites[i][j]`` means i sites, j DOFs.

    Attributes:
        nspf_list_sites (List[List[int]]) : Number of SPFs for each sites
        nsite (int) : Number of sites
        dim_of_sites (List[int]) : Maximum dimension of sites for each sites
        ndof_per_sites (List[int]) : Number of DOFs foe each sites.

    """

    def __init__(self, nspf_list_sites: list[list[int]]):
        self.nspf_list_sites = nspf_list_sites
        self.nsite = len(self.nspf_list_sites)
        self.dim_of_sites = [_prod(x) for x in nspf_list_sites]
        self.ndof_per_sites = [len(nspf_list) for nspf_list in nspf_list_sites]

    @classmethod
    def init_by_zip_dofs(
        cls, nspf_list_full: list[int], ndof_per_sites: list[int]
    ) -> LatticeInfo:
        """Initialization

        Args:
            nspf_list_full (List[int]): Number of SPF for each DOFs
            ndof_per_sites (List[int]): Number of DOFs for each sites

        Returns:
            LatticeInfo: Lattice informations
        """
        nsite = len(ndof_per_sites)
        nspf_list_sites = [
            nspf_list_full[
                sum(ndof_per_sites[:isite]) : sum(ndof_per_sites[: isite + 1])
            ]
            for isite in range(nsite)
        ]
        return LatticeInfo(nspf_list_sites)

    def alloc_superblock_random(
        self,
        m_aux_max: int,
        scale: float,
        weight_vib: list[list[float]],
        *,
        site_unitary: list[np.ndarray] | None = None,
    ) -> list[SiteCoef]:
        """Allocate superblock (core) elements randomly

        Args:
            m_aux_max (int): Bond dimension = Max rank of MPS after renormalization.
            scale (float, optional): Normalization scale. Defaults to 1.0.
            weight_vib (List[List[float]]): Weight of each DOFs for each sites.
            dvr_unitary (List[np.ndarray], optional): Unitary matrix which translate φ to χ.

        Returns:
            List[SiteCoef]: site coefficient for each sites
        """
        superblock = []
        for isite in range(self.nsite):
            is_lend = isite == 0
            is_rend = isite == self.nsite - 1
            dim_centr = self.dim_of_sites[isite]
            if is_lend:
                dim_left = 1
            else:
                dim_left = min(m_aux_max, _prod(self.dim_of_sites[:isite]))
            if is_rend:
                dim_right = 1
            else:
                dim_right = min(
                    m_aux_max, _prod(self.dim_of_sites[isite + 1 :])
                )
            """A(l,n,r)"""
            m_aux_l = min(dim_left, dim_centr * dim_right, m_aux_max)
            m_aux_r = min(dim_left * dim_centr, dim_right, m_aux_max)
            matC = SiteCoef.init_random(
                self.dim_of_sites[isite],
                m_aux_l,
                m_aux_r,
                weight_vib[isite],
                is_lend,
                is_rend,
            )

            if site_unitary is not None:
                if (unitary := site_unitary[isite]) is not None:
                    if const.use_jax:
                        matC = SiteCoef(
                            jnp.einsum("abc,bd->adc", matC.data, unitary),
                            gauge="C",
                        )
                    else:
                        matC = SiteCoef(
                            np.einsum("abc,bd->adc", matC, unitary), gauge="C"
                        )
            superblock.append(matC)

        for isite in range(self.nsite - 1, 0, -1):
            matR, sval = superblock[isite].gauge_trf("C2R")
            superblock[isite] = matR
            matC = superblock[isite - 1]
            if const.use_jax:
                superblock[isite - 1] = SiteCoef(
                    jnp.einsum("abc,cd->abd", matC.data, sval), gauge="C"
                )
            else:
                superblock[isite - 1] = SiteCoef(
                    np.einsum("abc,cd->abd", matC, sval), gauge="C"
                )

        superblock[0] *= scale / np.linalg.norm(superblock[0].data)

        logger.debug("Initial MPS Lattice")
        logger.debug(helper.get_tensornetwork_diagram_MPS(superblock))
        return superblock


def get_C_sval_states_norm(
    matC_or_sval_states: list[np.ndarray] | list[jax.Array],
) -> float:
    """matC in all electronic states norm

    Args:
        matC_or_sval_states (List[np.ndarray | jax.Array]): i-electronic states matC or sval

    Returns:
        float: norm
    """
    if const.use_jax:
        norm = math.sqrt(
            np.sum([jnp.linalg.norm(x) ** 2 for x in matC_or_sval_states])
        )
    else:
        norm = math.sqrt(
            np.sum([linalg.norm(x.flatten()) ** 2 for x in matC_or_sval_states])
        )
    return norm


def apply_superOp_direct(
    psite: int,
    superblock_states: list[list[SiteCoef]],
    op_lcr,
    matO_cas: HamiltonianMixin,
    superblock_states_unperturb: list[list[SiteCoef]],
) -> float:
    """concatenate PolynomialHamiltonian & coefficients"""
    matC_states_init: list[np.ndarray] | list[jax.Array]
    if const.use_jax:
        matC_states_init = [
            superblock_init[psite].data
            for superblock_init in superblock_states_unperturb
        ]  # type: ignore
    else:
        matC_states_init = [
            np.array(superblock_init[psite])
            for superblock_init in superblock_states_unperturb
        ]

    """exponentiation PolynomialHamiltonian"""
    if isinstance(matO_cas, PolynomialHamiltonian):
        multiplyOp = multiplyH_MPS_direct(op_lcr, matC_states_init, matO_cas)
    else:
        multiplyOp = multiplyH_MPS_direct_MPO(
            op_lcr, matC_states_init, matO_cas
        )

    matC_states_new = multiplyOp.dot(matC_states_init)
    norm = get_C_sval_states_norm(matC_states_new)
    matC_states_new = [x / norm for x in matC_states_new]

    """update(over-write) matC(psite)"""
    for istate, superblock in enumerate(superblock_states):
        superblock[psite] = SiteCoef(matC_states_new[istate], "C")
    return norm


def superblock_transLCR_psite(
    psite: int,
    superblock_states: list[list[SiteCoef]],
    left_is_sys: bool,
    regularize: bool = False,
):
    if left_is_sys:
        for superblock in superblock_states:
            superblock_transCR2LC_psite(psite, superblock, regularize)
    else:
        for superblock in superblock_states:
            superblock_transLC2CR_psite(psite, superblock, regularize)


def superblock_transCR2LC_psite(
    psite: int, superblock: list[SiteCoef], regularize: bool = False
):
    """..C(p) R(p+1).. -> ..L(p) C(p+1)"""
    matC = superblock[psite]
    matL, sval = matC.gauge_trf("C2L", regularize)
    matR = superblock[psite + 1]
    if const.use_jax:
        matC = SiteCoef(jnp.einsum("ij,jbc->ibc", sval, matR.data), "C")
    else:
        matC = SiteCoef(np.tensordot(sval, matR, axes=1), "C")
    superblock[psite] = matL
    superblock[psite + 1] = matC


def superblock_transLC2CR_psite(
    psite: int, superblock: list[SiteCoef], regularize: bool = False
):
    """..L(p-1) C(p).. -> ..C(p-1) R(p)"""
    matC = superblock[psite]
    matR, sval = matC.gauge_trf("C2R", regularize)
    matL = superblock[psite - 1]
    if const.use_jax:
        matC = SiteCoef(jnp.einsum("ijk,kb->ijb", matL.data, sval), "C")
    else:
        matC = SiteCoef(np.tensordot(matL, sval, axes=1), "C")
    superblock[psite] = matR
    superblock[psite - 1] = matC


def superblock_transCR2LR_psite(
    psite: int, superblock: list[SiteCoef], regularize: bool = False
) -> jax.Array | np.ndarray:
    """..C(p) R(p+1).. -> ..L(p) R(p+1)"""
    matC = superblock[psite]
    matL, sval = matC.gauge_trf("C2L", regularize)
    superblock[psite] = matL
    return sval


def superblock_transLC2LR_psite(
    psite: int, superblock: list[SiteCoef], regularize: bool = False
) -> np.ndarray | jax.Array:
    """..L(p-1) C(p).. -> ..L(p-1) R(p)"""
    matC = superblock[psite]
    matR, sval = matC.gauge_trf("C2R", regularize)
    superblock[psite] = matR
    return sval


def ints_spf2site_sum(
    ints_spf: SPFInts | list[jax.Array] | list[np.ndarray],
    nspf_list_sites_bra: list[list[int]],
    nspf_list_sites_ket: list[list[int]],
) -> list[np.ndarray] | list[jax.Array]:
    """Convert SPF integral to Site integral by summing up

    Args:
        ints_spf (SPFInts): SPF integrals
        nspf_list_sites_bra (List[List[int]]): number of spf on i-state, j-site in bra
        nspf_list_sites_ket (List[List[int]]): number of spf on i-state, j-site in ket

    Returns:
        List[np.ndarray | jax.Array]: Site integrals
    """
    ints_site = []
    for isite, (nspf_list_bra, nspf_list_ket) in enumerate(
        zip(nspf_list_sites_bra, nspf_list_sites_ket, strict=True)
    ):
        kdof_bgn = (
            0
            if isite == 0
            else len(
                list(itertools.chain.from_iterable(nspf_list_sites_bra[:isite]))
            )
        )
        ndof_site = len(nspf_list_sites_bra[isite])
        dum: np.ndarray | jax.Array
        dum = np.zeros(
            [np.prod(nspf_list_bra), np.prod(nspf_list_ket)], dtype=complex
        )
        for n_bra, J_bra in enumerate(
            itertools.product(*[range(x) for x in nspf_list_bra])
        ):
            for n_ket, J_ket in enumerate(
                itertools.product(*[range(x) for x in nspf_list_ket])
            ):
                for i in range(ndof_site):
                    kdof = kdof_bgn + i
                    if ints_spf[kdof] is not None:
                        if list(J_bra[:i] + J_bra[i + 1 :]) == list(
                            J_ket[:i] + J_ket[i + 1 :]
                        ):
                            dum[n_bra, n_ket] += ints_spf[kdof][
                                J_bra[i], J_ket[i]
                            ]
        if const.use_jax:
            dum = jnp.array(dum, dtype=jnp.complex128)
        ints_site.append(dum)
    return ints_site  # type: ignore


def ints_spf2site_prod(
    ints_spf: SPFInts | list[jax.Array] | list[np.ndarray],
    nspf_list_sites_bra: list[list[int]],
    nspf_list_sites_ket: list[list[int]],
) -> list[np.ndarray] | list[jax.Array]:
    """Convert SPF integral to Site integral for one-site operators

    This function is implemented for when nsite < ndof.

    Args:
        ints_spf (SPFInts): SPF integrals
        nspf_list_sites_bra (List[List[int]]): number of spf on i-state, j-site in bra
        nspf_list_sites_ket (List[List[int]]): number of spf on i-state, j-site in ket

    Returns:
        List[np.ndarray | jax.Array]: Site integrals
    """
    ints_site = []
    for isite, (nspf_list_bra, nspf_list_ket) in enumerate(
        zip(nspf_list_sites_bra, nspf_list_sites_ket, strict=True)
    ):
        kdof_bgn = (
            0
            if isite == 0
            else len(
                list(itertools.chain.from_iterable(nspf_list_sites_bra[:isite]))
            )
        )
        ndof_site = len(nspf_list_sites_bra[isite])
        dum: np.ndarray | jax.Array
        dum = np.ones(
            [np.prod(nspf_list_bra), np.prod(nspf_list_ket)], dtype=complex
        )
        for n_bra, J_bra in enumerate(
            itertools.product(*[range(x) for x in nspf_list_bra])
        ):
            for n_ket, J_ket in enumerate(
                itertools.product(*[range(x) for x in nspf_list_ket])
            ):
                for i in range(ndof_site):
                    kdof = kdof_bgn + i
                    dum[n_bra, n_ket] *= ints_spf[kdof][J_bra[i], J_ket[i]]
        if const.use_jax:
            dum = jnp.array(dum, dtype=jnp.complex128)
        ints_site.append(dum)
    return ints_site  # type: ignore


def distance_MPS(mps_A_inp: MPSCoef, mps_B_inp: MPSCoef) -> float:
    """Distance between old MPS and new MPS

    Args:
        mps_A_inp (MPSCoef): old MPS
        mps_B_inp (MPSCoef): new MPS

    Returns:
        float: distance (error)
    """
    dist_max = -1e9
    nsite = mps_A_inp.nsite
    nstate = mps_A_inp.nstate

    mps_A = copy.deepcopy(mps_A_inp)
    mps_B = copy.deepcopy(mps_B_inp)

    for psite in range(nsite):
        innerdot = 0.0
        for istate in range(nstate):
            superblock_A = mps_A.superblock_states[istate]
            superblock_B = mps_B.superblock_states[istate]
            matC_A = superblock_A[psite]
            matC_B = superblock_B[psite]
            innerdot += (matC_A - matC_B).norm() ** 2
            if psite < nsite - 1:
                superblock_transCR2LC_psite(psite, superblock_A)
                superblock_transCR2LC_psite(psite, superblock_B)
        if const.use_jax:
            dist_max = max(dist_max, innerdot**0.5)
        else:
            dist_max = max(dist_max, innerdot**0.5)

    return dist_max


def print_gauge(superblock_states: list[list[SiteCoef]]):
    for istate, superblock in enumerate(superblock_states):
        logger.debug(f"gauge of state-{istate}: ")
        for isite, site_coef in enumerate(superblock):
            logger.debug(f"{isite}: {site_coef.gauge}")


@partial(jax.jit, static_argnames=("remain_nleg",))
def _get_normalized_reduced_density_jax(
    cores: list[jax.Array],
    remain_nleg: tuple[int, ...],
) -> jax.Array:
    """
    Wavefunction is written by

    |Ψ> = C[0] R[1] R[2] ... R[f-1]

    if dof_pair = (0, 2), then the reduced density matrix is

    |ρ>[0, 2] = Tr_{1, 3, ..., f-1} |Ψ><Ψ|

    Contraction of tensor cores should be executed from right to left.

    """
    core = cores.pop()
    nleg = remain_nleg[-1]
    if nleg == 2:
        """
        i‾|‾k k‾|‾˙˙˙‾|
          j     |     |
          l     |     |
        a_|_k k_|_..._|
        """
        subscript = "ijk,alk->iajl"
    elif nleg == 1:
        """
        i‾|‾k k‾|‾˙˙˙‾|
          j     |     |
        a_|_k k_|_..._|
        """
        subscript = "ijk,ajk->iaj"
    else:
        raise ValueError(
            "The number of legs must be either 1 or 2 at the last site."
        )
    density = jnp.einsum(subscript, jnp.conj(core), core)
    isite = len(remain_nleg) - 1
    while cores:
        core = cores.pop()
        isite -= 1
        nleg = remain_nleg[isite]
        if nleg == 2:
            """
            l‾|‾i i‾|‾˙˙˙‾|
              n     ...   |
              m     ...   |
            b_|_a a_|_..._|
            """
            subscript = "lmi,bna,ia...->lbmn..."
        elif nleg == 1:
            """
            l‾|‾i i‾|‾˙˙˙‾|
              m     ...   |
            b_|_a a_|_..._|
            """
            subscript = "lmi,bma,ia...->lbm..."
        elif nleg == 0:
            subscript = "lmi,bma,ia...->lb..."
        else:
            raise ValueError("The number of legs must be less than 3.")
        density = jnp.einsum(subscript, jnp.conj(core), core, density)
    assert isite == 0
    return density[0, 0, ...]


@jax.jit
def _autocorr_single_state_jax(cores: list[jax.Array]) -> jax.Array:
    """
    i‾|‾k
      j |
    a_|_k
    """
    a = cores.pop()
    adag = jnp.conj(a)
    autocorr = jnp.einsum("ij,aj->ia", adag[:, :, 0], a[:, :, 0])
    while cores:
        a = cores.pop()
        adag = jnp.conj(a)
        """
        l‾|‾i‾|
          j   |
        b_|_a_|
        """
        autocorr = jnp.einsum("lji,bja,ia->lb", adag, a, autocorr)
    return autocorr[0, 0]
