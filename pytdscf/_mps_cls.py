"""
Matrix Product State (MPS) Coefficient MixIn Class
"""

from __future__ import annotations

import copy
import itertools
import math
from abc import ABC, abstractmethod
from functools import partial
from time import time
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as linalg
from loguru import logger as _logger
from opt_einsum import contract

import pytdscf._helper as helper
from pytdscf import _integrator
from pytdscf._const_cls import const
from pytdscf._contraction import (
    _block_type,
    _op_keys,
    multiplyH_MPS_direct,
    multiplyH_MPS_direct_MPO,
    multiplyK_MPS_direct,
    multiplyK_MPS_direct_MPO,
)
from pytdscf._site_cls import (
    SiteCoef,
    truncate_sigvec,
    validate_Atensor,
    validate_Btensor,
)
from pytdscf._spf_cls import SPFInts
from pytdscf.hamiltonian_cls import (
    HamiltonianMixin,
    PolynomialHamiltonian,
    TensorHamiltonian,
)
from pytdscf.model_cls import Model

logger = _logger.bind(name="main")


def ci_exp_time(func):
    def wrapper(*args, **kwargs):
        if const.verbose == 4:
            helper._ElpTime.ci_exp -= time()
            result = func(*args, **kwargs)
            helper._ElpTime.ci_exp += time()
        else:
            result = func(*args, **kwargs)
        return result

    return wrapper


def ci_rnm_time(func):
    def wrapper(*args, **kwargs):
        if const.verbose == 4:
            helper._ElpTime.ci_rnm -= time()
            result = func(*args, **kwargs)
            helper._ElpTime.ci_rnm += time()
        else:
            result = func(*args, **kwargs)
        return result

    return wrapper


class MPSCoef(ABC):
    r"""Matrix Product State Coefficient MixIn Class

    .. math::
       a_{\tau_1}^{j_1} a_{\tau_1\tau_2}^{j_2} \cdots a_{\tau_{f-1}}^{j_f}

    Attributes:
       lattice_info_states (List[LatticeInfo]) : Lattice Information of each states
       superblock_states (List[List[SiteCoef]]) : Superblocks (Tensor Cores) of each states
       nstate (int) : Number of MPS (not length of chain)
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
            dict[
                tuple[int, int],
                dict[_op_keys, _block_type],
            ]
            | None
        ) = None
        self.ints_site: (
            dict[
                tuple[int, int],
                dict[_op_keys, _block_type],
            ]
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
            logger.debug(
                f"Initial MPS: {istate}-state with weights {weight_estate[istate]}"
            )
            for idof in range(ndof):
                if const.verbose > 2:
                    logger.debug(
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
    def get_matH_sweep(self, matH) -> HamiltonianMixin:
        pass

    @abstractmethod
    def get_matH_tdh(self, matH, op_block_cas):
        pass

    @abstractmethod
    def get_matH_cas(self, matH, ints_spf: SPFInts | None):
        pass

    @abstractmethod
    def get_ints_site(
        self, ints_spf: SPFInts | None, onesite_name: str = "onesite"
    ) -> dict[tuple[int, int], dict[_op_keys, _block_type]]:
        """Get integral between p-site bra amd p-site ket in all states pair

        Args:
            ints_spf (SPFInts): SPF integrals
            onesite_name (str, optional) : Defaults to 'onesite'.

        Returns:
            Dict[Tuple[int,int],Dict[_op_keys, np.ndarray]]: Site integrals
        """
        pass

    @abstractmethod
    def construct_mfop_MPS(
        self,
        psite: int,
        op_sys: dict[tuple[int, int], dict[_op_keys, _block_type]],
        op_env: dict[tuple[int, int], dict[_op_keys, _block_type]],
        matH_cas,
        A_is_sys: bool,
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
        ints_site: dict[tuple[int, int], dict[_op_keys, _block_type]],
        matH_cas,
        *,
        begin_site: int,
        end_site: int,
    ):
        pass

    @abstractmethod
    def construct_mfop_along_sweep_TEMP4DIPOLE(
        self,
        ints_site: dict[tuple[int, int], dict[_op_keys, _block_type]],
        matO_cas,
        *,
        begin_site: int,
        end_site: int,
        mps_coef_ket=None,
    ):
        pass

    @abstractmethod
    def construct_op_zerosite(
        self,
        superblock_states: list[list[SiteCoef]],
        operator: HamiltonianMixin | None = None,
    ) -> dict[tuple[int, int], dict[_op_keys, _block_type]]:
        """initialize op_block_psites
        Args:
            superblock_states (List[List[SiteCoef]]) : Super Blocks (Tensor Cores) of each electronic states
            operator (Optional[HamiltonianMixin], optional): Operator (such as Hamiltonian). Defaults to None.

        Returns:
            Dict[Tuple[int,int], Dict[_op_keys, np.ndarray]] : block operator. \
                'ovlp' and 'auto' operator are 2-rank tensor, 'diag_mpo' is 3-rank tensor, 'nondiag_mpo' is 4-rank tensor. \
                '~_summed' means complementary operator.
        """
        pass

    @staticmethod
    @abstractmethod
    def renormalize_op_psite(
        *,
        psite: int,
        superblock_states: list[list[SiteCoef]],
        op_block_states: dict[tuple[int, int], dict[_op_keys, _block_type]],
        ints_site: dict[tuple[int, int], dict[_op_keys, _block_type]] | None,
        hamiltonian: HamiltonianMixin | None,
        A_is_sys: bool,
        superblock_states_ket=None,
        superblock_states_bra=None,
    ) -> dict[tuple[int, int], dict[_op_keys, _block_type]]:
        pass

    @abstractmethod
    def operators_for_superH(
        self,
        psite: int,
        op_sys: dict[tuple[int, int], dict[_op_keys, _block_type]],
        op_env: dict[tuple[int, int], dict[_op_keys, _block_type]],
        ints_site: dict[tuple[int, int], dict[_op_keys, _block_type]] | None,
        hamiltonian: HamiltonianMixin,
        A_is_sys: bool,
    ) -> list[
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
    ]:
        """ LCR operator

        prepare operators for multiplying the full-matrix PolynomialHamiltonian on-the-fly

        Args:
            psite (int): site index on "Psi"
            op_sys (Dict[Tuple[int,int], Dict[_op_keys, _block_type]]): System operator
            op_env (Dict[Tuple[int,int], Dict[_op_keys, _block_type]]): Environment operator
            ints_site (Dict[Tuple[int,int],Dict[_op_keys, np.ndarray]]): Site integral
            matH_cas (PolynomialHamiltonian) : Hamiltonian
            A_is_sys (bool): Whether left block is System

        Returns:
            List[List[Dict[_op_keys, Tuple[_block_type, _block_type, _block_type]]]]: \
                [i-bra-state][j-ket-state]['q'] =  (op_l, op_c, op_r)
        """
        pass

    @abstractmethod
    def operators_for_superK(
        self,
        psite: int,
        op_sys: dict[tuple[int, int], dict[_op_keys, _block_type]],
        op_env: dict[tuple[int, int], dict[_op_keys, _block_type]],
        hamiltonian: HamiltonianMixin,
        A_is_sys: bool,
    ) -> list[
        list[
            dict[
                _op_keys,
                tuple[_block_type, _block_type],
            ]
        ]
    ]:
        """ LsR operator

        construct full-matrix Kamiltonian

        Args:
            psite (int): site index on "Psi"
            op_sys (Dict[Tuple[int,int], Dict[_op_keys, np.ndarray | jax.Array]]): System operator
            op_env (Dict[Tuple[int,int], Dict[_op_keys, np.ndarray | jax.Array]]): Environment operator
            matH_cas (PolynomialHamiltonian) : Hamiltonian
            A_is_sys (bool): Whether left block is System

        Returns:
            List[List[Dict[_op_keys, Tuple[np.ndarray, np.ndarray, np.ndarray]]]]: \
                [i-bra-state][j-ket-state]['q'] =  (op_l, op_r)
        """
        pass

    def is_psite_canonical(self, psite: int, numerical: bool = False) -> bool:
        """check if mps = A..A(p-1)Psi(p)B(p+1)..B

        Args:
            psite (int): guess of "Psi" site

        Returns:
            bool : Correct or Incorrect
        """
        for superblock in self.superblock_states:
            for isite, site_coef in enumerate(superblock):
                expected_gauge = (
                    "A" if isite < psite else "Psi" if isite == psite else "B"
                )
                if site_coef.gauge != expected_gauge:
                    return False
                if numerical:
                    if site_coef.gauge == "A":
                        validate_Atensor(site_coef)
                    elif site_coef.gauge == "B":
                        validate_Btensor(site_coef)
        return True

    def assert_psite_canonical(self, psite: int, numerical: bool = False):
        assert self.is_psite_canonical(psite, numerical), (
            "wrong gauge status. It's assumed to be A..A(p-1)Psi(p)B(p+1)..B in superblock"

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
        nsite = len(self.superblock_states[0])
        self.apply_dipole_along_sweep(
            ci_coef_init,
            self.ints_site_dipo,
            self.matO_sweep,
            begin_site=0,
            end_site=nsite - 1,
        )
        norm = self.apply_dipole_along_sweep(
            ci_coef_init,
            self.ints_site_dipo,
            self.matO_sweep,
            begin_site=nsite - 1,
            end_site=0,
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
            # assert isinstance(ints_spf, SPFInts)
            self.ints_site = self.get_ints_site(ints_spf)
            matO_cas = (
                matH
                if "enable_tdh_dofs" not in const.keys
                else self.get_matH_cas(matH, ints_spf)
            )
            self.matH_sweep = self.get_matH_sweep(matO_cas)
        if const.verbose == 4:
            helper._ElpTime.ci_etc += time()
        nsite = len(self.superblock_states[0])
        self.propagate_along_sweep(
            self.ints_site,
            self.matH_sweep,
            stepsize,
            begin_site=0,
            end_site=nsite - 1,
        )
        self.propagate_along_sweep(
            self.ints_site,
            self.matH_sweep,
            stepsize,
            begin_site=nsite - 1,
            end_site=0,
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
        nsite = len(self.superblock_states[0])
        if self.is_psite_canonical(0):
            mfop = mps_copy_bra.construct_mfop_along_sweep_TEMP4DIPOLE(
                ints_site,
                matO_sweep,
                begin_site=0,
                end_site=nsite - 1,
                mps_coef_ket=mps_copy_ket,
            )
        elif self.is_psite_canonical(self.nsite - 1):
            mfop = mps_copy_bra.construct_mfop_along_sweep_TEMP4DIPOLE(
                ints_site,
                matO_sweep,
                begin_site=nsite - 1,
                end_site=0,
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
        """Get Expectation Value at "Psi" = p-site

        Args:
            ints_spf (SPFInts): SPF integral
            matOp (HamiltonianMixin): Operator
            psite (Optional[int], optional): When calculate expectation value. Defaults to 0.

        Returns:
            complex or float : expectation value
        """
        assert psite == 0, f"psite = {psite} is not supported"
        self.assert_psite_canonical(psite, numerical=const.pytest_enabled)
        superblock_states = self.superblock_states
        # - inefficient impl-#
        if hasattr(matOp, "onesite_name"):
            ints_site = self.get_ints_site(ints_spf, matOp.onesite_name)
        else:
            ints_site = self.get_ints_site(ints_spf)
        matOp = self.get_matH_sweep(matOp)
        op_sys = self.construct_op_zerosite(
            superblock_states,
            operator=matOp,
        )
        nsite = len(superblock_states[0])
        op_env_sites = self.construct_op_sites(
            superblock_states,
            ints_site=ints_site,
            begin_site=nsite - 1,
            end_site=0,
            matH_cas=matOp,
        )
        op_env = op_env_sites.pop()
        op_lcr = self.operators_for_superH(
            psite, op_sys, op_env, ints_site, matOp, True
        )

        """concatenate PolynomialHamiltonian & Coefficients over the electronic states"""
        matPsi_states: list[np.ndarray] | list[jax.Array]
        if const.use_jax:
            matPsi_states = [
                superblock[psite].data for superblock in superblock_states
            ]  # type: ignore
        else:
            matPsi_states = [
                np.array(superblock[psite]) for superblock in superblock_states
            ]
        if isinstance(matOp, PolynomialHamiltonian):
            multiplyH = multiplyH_MPS_direct(
                op_lcr_states=op_lcr,
                psi_states=matPsi_states,
                hamiltonian=matOp,
            )
        elif isinstance(matOp, TensorHamiltonian):
            multiplyH = multiplyH_MPS_direct_MPO(
                op_lcr_states=op_lcr,
                psi_states=matPsi_states,
                hamiltonian=matOp,
            )
        else:
            raise NotImplementedError(f"{type(matOp)=}")

        expectation_value = _integrator.expectation_Op(
            matPsi_states,  # type: ignore
            multiplyH,
            matPsi_states,  # type: ignore
        )
        return expectation_value

    def autocorr(self, ints_spf: SPFInts, psite: int = 0) -> complex:
        r"""Get auto-correlation value

        .. math ::
           a(t) = \langle\Psi(t)|\Psi(0)\rangle = \langle \Psi(t/2)^\ast|\Psi(t/2) \rangle

        Args:
            ints_spf (SPFInts): SPF integral
            psite (int, optional): "Psi" site index. Defaults to 0.

        Returns:
            complex: auto-correlation value
        """
        assert psite == 0, f"psite = {psite} is not supported"
        nstate = len(self.superblock_states)
        self.assert_psite_canonical(psite)
        superblock_states = self.superblock_states
        nstate = len(superblock_states)
        # - inefficient impl-#
        ints_site = self.get_ints_site(ints_spf)
        op_sys = self.construct_op_zerosite(
            superblock_states,
            operator=None,
        )
        nsite = len(superblock_states[0])
        op_env = self.construct_op_sites(
            superblock_states,
            ints_site=ints_site,
            begin_site=nsite - 1,
            end_site=0,
        ).pop()
        op_lcr = self.operators_for_autocorr(
            psite, op_sys, op_env, ints_site, nstate, True
        )

        """concatenate PolynomialHamiltonian & Coefficients over the electronic states"""
        matPsi_states: list[np.ndarray] | list[jax.Array]
        if const.use_jax:
            matPsi_states = [
                superblock[psite].data for superblock in superblock_states
            ]  # type: ignore
        else:
            matPsi_states = [
                np.array(superblock[psite]) for superblock in superblock_states
            ]
        multiplyH = multiplyH_MPS_direct(
            op_lcr_states=op_lcr,
            psi_states=matPsi_states,
            hamiltonian=None,
        )

        psivec = multiplyH.stack(matPsi_states)
        sigvec = multiplyH.stack(multiplyH.dot_autocorr(matPsi_states))

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
        ints_site: dict[tuple[int, int], dict[_op_keys, _block_type]],
        matO_cas: HamiltonianMixin,
        begin_site: int = 0,
        end_site: int | None = None,
    ) -> float:
        nsite = len(self.superblock_states[0])
        if end_site is None:
            end_site = nsite - 1
            assert begin_site == 0
        assert isinstance(end_site, int)
        A_is_sys = begin_site < end_site
        superblock_states = self.superblock_states
        superblock_states_ket = mps_coef_init.superblock_states
        op_sys = self.construct_op_zerosite(
            superblock_states,
            operator=matO_cas,
        )
        if self.op_sys_sites_dipo is None:
            op_env_sites = self.construct_op_sites(
                superblock_states,
                ints_site=ints_site,
                begin_site=end_site,
                end_site=begin_site,
                matH_cas=matO_cas,
                superblock_states_ket=superblock_states_ket,
            )
        else:
            op_env_sites = self.op_sys_sites_dipo[:]

        if const.standard_method:
            self.op_sys_sites_dipo = [op_sys]

        if A_is_sys:
            psites_sweep = range(0, nsite, +1)
            end_site = nsite - 1
        else:
            psites_sweep = range(nsite - 1, -1, -1)
            end_site = 0
        for psite in psites_sweep:
            op_env = op_env_sites.pop()
            op_lcr = self.operators_for_superH(
                psite,
                op_sys,
                op_env,
                ints_site,
                matO_cas,
                A_is_sys,
            )

            norm = apply_superOp_direct(
                psite,
                superblock_states,
                op_lcr,
                matO_cas,
                superblock_states_ket,
            )

            if psite != end_site:
                superblock_trans_APsiB_psite(
                    psite, superblock_states, toAPsi=A_is_sys
                )
                superblock_trans_APsiB_psite(
                    psite, superblock_states_ket, toAPsi=A_is_sys
                )
                op_sys = self.renormalize_op_psite(
                    psite=psite,
                    superblock_states=superblock_states,
                    op_block_states=op_sys,
                    ints_site=ints_site,
                    hamiltonian=matO_cas,
                    A_is_sys=A_is_sys,
                    superblock_states_ket=superblock_states_ket,
                )
                if self.op_sys_sites_dipo is not None:
                    self.op_sys_sites_dipo.append(op_sys)
        return norm

    def propagate_along_sweep(
        self,
        ints_site: dict[tuple[int, int], dict[_op_keys, _block_type]] | None,
        matH_cas: HamiltonianMixin,
        stepsize: float,
        *,
        begin_site: int,
        end_site: int,
        op_sys_initial: dict[tuple[int, int], dict[_op_keys, _block_type]]
        | None = None,
        skip_end_site: bool = False,
    ) -> dict[tuple[int, int], dict[_op_keys, _block_type]]:
        """Propagate MPS along a sweep

        Args:
            ints_site (dict[tuple[int, int], dict[_op_keys, list[np.ndarray] | list[jax.Array]]]): interaction site
            matH_cas (HamiltonianMixin): Hamiltonian
            stepsize (float): time step

        Returns:
            dict[tuple[int, int], dict[_op_keys, _block_type]]: System block operators at the end of the sweep
        """
        assert max(begin_site, end_site) < len(self.superblock_states[0])
        A_is_sys = begin_site <= end_site
        step = +1 if A_is_sys else -1
        to: Literal["->", "<-"] = "->" if A_is_sys else "<-"
        nsite = (
            end_site - begin_site + 1 if A_is_sys else begin_site - end_site + 1
        )
        superblock_states = self.superblock_states

        if op_sys_initial is None:
            op_sys_initial = self.construct_op_zerosite(
                superblock_states,
                operator=matH_cas,
            )
        op_sys = op_sys_initial
        if self.op_sys_sites is None:
            # If t==0 or Include SPF or time-dependent Hamiltonian
            op_env_sites = self.construct_op_sites(
                superblock_states,
                ints_site=ints_site,
                begin_site=end_site,
                end_site=begin_site,
                matH_cas=matH_cas,
            )
        else:
            # Using [:] creates a shallow copy, so pop() operations on op_env_sites
            # won't affect the original self.op_sys_sites list but its elements has the
            # same reference to the original self.op_sys_sites
            op_env_sites = self.op_sys_sites[:]
        # When A_is_sys,
        # [sys0, sys1, ..., sysN-1, envM-1, ..., env1, env0]
        # Otherwise,
        # [env0, env1, ..., envM-1, sysN-1, ..., sys1, sys0]

        if const.standard_method and not const.doTDHamil:
            # When SPF is not employed and Hamiltonian is time-independent,
            # the block operators are the same as the one calculated in previous time step
            # thus, we can record op_sys to reduce additional calculation
            self.op_sys_sites = [op_sys_initial]
        else:
            # Either SPF is employed or Hamiltonian is time-dependent,
            self.op_sys_sites = None

        if const.adaptive:
            assert len(superblock_states) == 1, (
                "Only one superblock is implemented for adaptive calculation"
            )
            assert isinstance(matH_cas, TensorHamiltonian) and const.use_mpo, (
                "Only MPO is implemented for adaptive calculation"
            )
            superblock_states_full = [
                get_superblock_full(superblock_states[0], delta_rank=const.dD)
            ]

        psites_sweep = range(begin_site, end_site + step, step)
        for psite in psites_sweep:
            if const.verbose == 4:
                helper._ElpTime.ci_etc -= time()
            if skip_end_site and psite == end_site:
                return op_sys
            helper._Debug.site_now = psite
            op_env = op_env_sites.pop()

            if const.adaptive and psite != end_site:
                if to == "->":
                    L, C, _ = superblock_states[0][psite].data.shape
                    R = superblock_states[0][psite + 1].data.shape[0]
                else:
                    _, C, R = superblock_states[0][psite].data.shape
                    L = superblock_states[0][psite - 1].data.shape[2]
                if _is_max_rank := is_max_rank(superblock_states[0][psite], to):
                    pass
                else:
                    op_env_previous = op_env_sites[-1]
                    newD, error, op_env_D_bra, op_env_D_braket = (
                        self.get_adaptive_rank_and_block(
                            psite=psite,
                            superblock_states=superblock_states,
                            superblock_states_full=superblock_states_full,
                            op_env_previous=op_env_previous,
                            hamiltonian=matH_cas,  # type: ignore
                            to=to,
                        )
                    )
                    op_env = op_env_D_bra
                    if to == "->":
                        R = newD
                    else:
                        L = newD
                tensor_shapes_out = (L, C, R)
            else:
                tensor_shapes_out = None
            op_lcr = self.operators_for_superH(
                psite=psite,
                op_sys=op_sys,
                op_env=op_env,
                ints_site=ints_site,
                hamiltonian=matH_cas,
                A_is_sys=A_is_sys,
            )
            if const.verbose == 4:
                helper._ElpTime.ci_etc += time()
            try:
                self.exp_superH_propagation_direct(
                    psite=psite,
                    superblock_states=superblock_states,
                    op_lcr=op_lcr,
                    matH_cas=matH_cas,
                    stepsize=stepsize,
                    tensor_shapes_out=tensor_shapes_out,
                )
            except Exception:
                from loguru import logger as _logger

                logger = _logger.bind(name="main")
                logger.error(
                    f"{psite=} {to=} {tensor_shapes_out=} {superblock_states[0][psite].data.shape=}"
                )
                raise

            if psite != end_site:
                svalues, op_sys = self.trans_next_psite_AsigmaB(
                    psite=psite,
                    superblock_states=superblock_states,
                    op_sys=op_sys,
                    ints_site=ints_site,
                    matH_cas=matH_cas,
                    PsiB2AB=A_is_sys,
                )

                if const.verbose == 4:
                    helper._ElpTime.ci_etc -= time()

                if const.adaptive and not _is_max_rank:
                    op_env = op_env_D_braket

                op_lr = self.operators_for_superK(
                    psite=psite,
                    op_sys=op_sys,
                    op_env=op_env,
                    hamiltonian=matH_cas,
                    A_is_sys=A_is_sys,
                )
                if const.verbose == 4:
                    helper._ElpTime.ci_etc += time()

                svalues = self.exp_superK_propagation_direct(
                    op_lr=op_lr,
                    hamiltonian=matH_cas,
                    svalues=svalues,
                    stepsize=stepsize,
                )
                if const.adaptive:
                    if False:
                        if to == "->":
                            Asite, svalues[0], Bsite = truncate_sigvec(
                                superblock_states[0][psite],
                                svalues[0],
                                superblock_states[0][psite + 1],
                                const.p_svd,
                            )
                        else:
                            Asite, svalues[0], Bsite = truncate_sigvec(
                                superblock_states[0][psite - 1],
                                svalues[0],
                                superblock_states[0][psite],
                                const.p_svd,
                            )
                    if psite == begin_site:
                        op_sys_prev = op_sys_initial
                    else:
                        if self.op_sys_sites is None:
                            raise ValueError("op_sys_sites is not set")
                        op_sys_prev = self.op_sys_sites[-1]
                    op_sys = self.renormalize_op_psite(
                        psite=psite,
                        superblock_states=superblock_states,
                        op_block_states=op_sys_prev,
                        ints_site=None,
                        hamiltonian=matH_cas,
                        A_is_sys=A_is_sys,
                    )

                self.trans_next_psite_APsiB(
                    psite=psite,
                    superblock_states=superblock_states,
                    svalues=svalues,
                    A_is_sys=A_is_sys,
                )

                if self.op_sys_sites is not None:
                    self.op_sys_sites.append(op_sys)
                    assert (
                        len(self.op_sys_sites) + len(op_env_sites) == nsite + 1
                    ), (
                        f"{len(self.op_sys_sites)=} + {len(op_env_sites)=} == {nsite+1=}"
                    )

        return op_sys

    @ci_exp_time
    def exp_superH_propagation_direct(
        self,
        psite: int,
        superblock_states: list[list[SiteCoef]],
        op_lcr: list[
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
        matH_cas: HamiltonianMixin,
        stepsize: float,
        tensor_shapes_out: tuple[int, ...] | None = None,
    ):
        """concatenate PolynomialHamiltonian & coefficients"""
        matPsi_states: list[np.ndarray] | list[jax.Array]
        matPsi_states = [
            superblock[psite].data for superblock in superblock_states
        ]  # type: ignore

        """exponentiation PolynomialHamiltonian"""
        if isinstance(matH_cas, PolynomialHamiltonian):
            multiplyH = multiplyH_MPS_direct(
                op_lcr_states=op_lcr,
                psi_states=matPsi_states,
                hamiltonian=matH_cas,
                tensor_shapes_out=tensor_shapes_out,
            )
        else:
            assert isinstance(matH_cas, TensorHamiltonian)
            multiplyH = multiplyH_MPS_direct_MPO(
                op_lcr_states=op_lcr,
                psi_states=matPsi_states,
                hamiltonian=matH_cas,
                tensor_shapes_out=tensor_shapes_out,
            )

        if not const.doRelax:
            matPsi_states_new = _integrator.short_iterative_lanczos(
                -1.0j * stepsize / 2, multiplyH, matPsi_states, const.thresh_exp
            )

        elif const.doRelax == "improved":
            matPsi_states_new = _integrator.matrix_diagonalize_lanczos(
                multiplyH, matPsi_states
            )
            norm = get_C_sval_states_norm(matPsi_states_new)
            matPsi_states_new = [x / norm for x in matPsi_states_new]  # type: ignore

        else:
            matPsi_states_new = _integrator.short_iterative_lanczos(
                -1.0 * stepsize / 2, multiplyH, matPsi_states, const.thresh_exp
            )
            if not const.nonHermitian:
                norm = get_C_sval_states_norm(matPsi_states_new)
                matPsi_states_new = [x / norm for x in matPsi_states_new]  # type: ignore

        """update(over-write) matPsi(psite)"""
        for istate, superblock in enumerate(superblock_states):
            superblock[psite] = SiteCoef(
                data=matPsi_states_new[istate], gauge="Psi", isite=psite
            )

    @ci_exp_time
    def exp_superK_propagation_direct(
        self,
        op_lr: list[
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
        hamiltonian: HamiltonianMixin,
        svalues: list[np.ndarray] | list[jax.Array],
        stepsize: float,
        tensor_shapes_out: tuple[int, ...] | None = None,
    ):
        """concatenate PolynomialHamiltonian & coefficients"""
        svalues_states = svalues

        """exponentiation PolynomialHamiltonian"""
        if isinstance(hamiltonian, PolynomialHamiltonian):
            multiplyK = multiplyK_MPS_direct(
                op_lr_states=op_lr,
                psi_states=svalues_states,
                hamiltonian=hamiltonian,
                tensor_shapes_out=tensor_shapes_out,
            )
        else:
            assert isinstance(hamiltonian, TensorHamiltonian)
            multiplyK = multiplyK_MPS_direct_MPO(
                op_lr_states=op_lr,
                psi_states=svalues_states,
                hamiltonian=hamiltonian,
                tensor_shapes_out=tensor_shapes_out,
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
            if not const.nonHermitian:
                norm = get_C_sval_states_norm(svalues_states_new)
                svalues_states_new = [x / norm for x in svalues_states_new]  # type: ignore
        return svalues_states_new

    def trans_next_psite_APsiB(
        self,
        psite: int,
        superblock_states: list[list[SiteCoef]],
        svalues: list[np.ndarray] | list[jax.Array],
        A_is_sys: bool,
    ):
        """over-write sval"""
        for sval, superblock in zip(svalues, superblock_states, strict=True):
            if A_is_sys:
                """sval x B(p+1) -> Psi(p+1)"""
                matA = superblock[psite]
                matB = superblock[psite + 1]
                assert matA.gauge == "A", f"{matA.gauge=}"
                assert matB.gauge == "B", f"{matB.gauge=}"
                if const.use_jax:
                    matB.data = jnp.einsum("ij,jbc->ibc", sval, matB.data)
                else:
                    matB.data = np.tensordot(sval, matB.data, axes=(1, 0))
                matB.gauge = "Psi"
            else:
                """A(p-1) x sval -> Psi(p-1)"""
                matA = superblock[psite - 1]
                matB = superblock[psite]
                assert matB.gauge == "B", (
                    f"matB.gauge should be B, but {matB.gauge}"
                )
                assert matA.gauge == "A", (
                    f"matA.gauge should be A, but {matA.gauge}"
                )
                if const.use_jax:
                    matA.data = jnp.einsum("ijk,kb->ijb", matA.data, sval)
                else:
                    matA.data = np.tensordot(matA.data, sval, axes=(2, 0))
                matA.gauge = "Psi"

    def _get_normalized_reduced_density(
        self, istate: int, remain_nleg: tuple[int, ...]
    ) -> np.ndarray | jax.Array:
        """
        Wavefunction is written by

        |Ψ> = Psi[0] B[1] B[2] ... B[f-1]

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

    @ci_rnm_time
    def construct_op_sites(
        self,
        superblock_states: list[list[SiteCoef]],
        *,
        ints_site: dict[tuple[int, int], dict[_op_keys, _block_type]] | None,
        begin_site: int,
        end_site: int,
        matH_cas: HamiltonianMixin | None = None,
        op_initial_block: dict[tuple[int, int], dict[_op_keys, _block_type]]
        | None = None,
        superblock_states_ket=None,
        superblock_states_bra=None,
    ) -> list[dict[tuple[int, int], dict[_op_keys, _block_type]]]:
        """Construct Environment Operator

        Args:
            superblock_states (List[List[SiteCoef]]) : Super Blocks (Tensor Cores) of each electronic states
            ints_site (Dict[Tuple[int,int],Dict[_op_keys, np.ndarray]]): Site integral
            begin_site (int) : begin site index
            end_site (int) : end site index
            matH_cas (HamiltonianMixin) : Hamiltonian
            op_initial_block (Dict[Tuple[int,int], Dict[_op_keys, _block_type]]) : initial block operators.
            superblock_states_ket (List[List[SiteCoef]]) : Super Blocks (Tensor Cores) of each electronic states

        Returns:
            List[Dict[Tuple[int,int], Dict[_op_keys, _block_type]]] : Env. Operator
        """

        assert begin_site != end_site or len(superblock_states[0]) == 1, (
            f"{begin_site=} == {end_site=}"
        )
        if op_initial_block is None:
            op_initial_block = self.construct_op_zerosite(
                superblock_states,
                operator=matH_cas,
            )
        op_block_isites = [op_initial_block]
        set_op_left = begin_site < end_site

        step = 1 if set_op_left else -1
        psites_sweep = range(begin_site, end_site, step)

        for psite in psites_sweep:
            """construct op_block for <A(0)A(1)...A(isite)|Op|A(0)A(1)...A(isite)>"""
            op_block_isites.append(
                self.renormalize_op_psite(
                    psite=psite,
                    superblock_states=superblock_states,
                    op_block_states=op_block_isites[-1],
                    ints_site=ints_site,
                    hamiltonian=matH_cas,
                    A_is_sys=set_op_left,
                    superblock_states_ket=superblock_states_ket,
                    superblock_states_bra=superblock_states_bra,
                )
            )

        return op_block_isites

    @ci_rnm_time
    def trans_next_psite_AsigmaB(
        self,
        psite: int,
        superblock_states: list[list[SiteCoef]],
        op_sys: dict[tuple[int, int], dict[_op_keys, _block_type]],
        ints_site: dict[tuple[int, int], dict[_op_keys, _block_type]] | None,
        matH_cas: HamiltonianMixin,
        *,
        PsiB2AB: bool,
        superblock_states_ket=None,
        superblock_states_bra=None,
        regularize=False,
    ) -> tuple[
        list[np.ndarray] | list[jax.Array],
        dict[tuple[int, int], dict[_op_keys, _block_type]],
    ]:
        """..Psi(p) B(p+1).. -> ..A(p) sval B(p+1)"""

        def _trans_PsiB2AB_psite(
            superblock_states,
        ) -> list[np.ndarray] | list[jax.Array]:
            svalues = []
            for superblock in superblock_states:
                if PsiB2AB:
                    svalues.append(
                        superblock_trans_PsiB2AB_psite(
                            psite, superblock, regularize=regularize
                        )
                    )
                else:
                    svalues.append(
                        superblock_trans_APsi2AB_psite(
                            psite, superblock, regularize=regularize
                        )
                    )
            return svalues  # type: ignore

        svalues = _trans_PsiB2AB_psite(superblock_states)
        if superblock_states_ket:
            _ = _trans_PsiB2AB_psite(superblock_states_ket)
        op_sys_next = self.renormalize_op_psite(
            psite=psite,
            superblock_states=superblock_states,
            op_block_states=op_sys,
            ints_site=ints_site,
            hamiltonian=matH_cas,
            A_is_sys=PsiB2AB,
            superblock_states_ket=superblock_states_ket,
            superblock_states_bra=superblock_states_bra,
        )

        return svalues, op_sys_next

    def operators_for_autocorr(
        self,
        psite: int,
        op_sys: dict[tuple[int, int], dict[_op_keys, _block_type]],
        op_env: dict[tuple[int, int], dict[_op_keys, _block_type]],
        ints_site: dict[tuple[int, int], dict[_op_keys, _block_type]],
        nstate: int,
        A_is_sys: bool,
    ) -> list[
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
    ]:
        """auto correlation operator

        prepare operators for multiplying the full-matrix auto-correlation operator on-the-fly

        Args:
            psite (int): site index on "Psi"
            op_sys (Dict[Tuple[int,int], Dict[_op_keys, np.ndarray | jax.Array]]): System operator
            op_env (Dict[Tuple[int,int], Dict[_op_keys, np.ndarray | jax.Array]]): Environment operator
            ints_site (Dict[Tuple[int,int],Dict[_op_keys, np.ndarray | jax.Array]]): Site integral
            nstate (int): number of state
            A_is_sys (bool): Whether left block is System

        Returns:
            List[List[Dict[_op_keys, Tuple[np.ndarray, np.ndarray, np.ndarray]]]]: \
                [i-bra-state][j-ket-state]['auto'] =  (op_l, op_c, op_r)
        """
        op_lcr: list[
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
        ]
        op_lcr = [[None for j in range(nstate)] for i in range(nstate)]  # type: ignore
        for istate in range(nstate):
            statepair = (istate, istate)
            op_l_auto = (
                op_sys[statepair]["auto"]
                if A_is_sys
                else op_env[statepair]["auto"]
            )
            op_r_auto = (
                op_env[statepair]["auto"]
                if A_is_sys
                else op_sys[statepair]["auto"]
            )
            auto_site = ints_site[statepair]["auto"]
            assert isinstance(auto_site, list)
            op_c_auto = auto_site[psite]

            op_lcr[istate][istate] = {"auto": (op_l_auto, op_c_auto, op_r_auto)}

        return op_lcr

    def get_psi_sigvec_psi_fullblock(
        self,
        psite: int,
        superblock_states: list[list[SiteCoef]],
        to: Literal["->", "<-"],
        op_block: dict[tuple[int, int], dict[_op_keys, _block_type]],
        delta_rank: int,
        hamiltonian: TensorHamiltonian,
    ):
        """
        if to == "right":
        calculate AAσB=AAΨ',
        then return Ψ, σ, Ψ'
        if to == "left":
        calculate AσBB=Ψ'BB,
        then return Ψ', σ, Ψ
        """
        if len(superblock_states) > 1:
            raise NotImplementedError("multi MPS is not implemented")
        psi_site = superblock_states[0][psite].copy()
        nsite = len(superblock_states[0])
        superblock_states_trans: list[list[SiteCoef]]
        superblock_states_trans = [[None for _ in range(nsite)]]  # type: ignore
        superblock_states_trans_bra = [[None for _ in range(nsite)]]
        match to:
            case "->":
                B_site = superblock_states[0][psite + 1]
                A_site, sigvec = psi_site.gauge_trf(key="Psi2Asigma")
                psi_prime_site = np.tensordot(sigvec, B_site.data, axes=(1, 0))
                A_site_full = A_site.thin_to_full(delta_rank=delta_rank)
                superblock_states_trans[0][psite] = A_site
                superblock_states_trans_bra[0][psite] = A_site_full
                op_block_A_full = self.renormalize_op_psite(
                    psite=psite,
                    superblock_states=superblock_states_trans,
                    op_block_states=op_block,
                    ints_site=None,
                    hamiltonian=hamiltonian,
                    A_is_sys=True,
                    superblock_states_bra=superblock_states_trans_bra,
                    superblock_states_ket=None,
                )
                return psi_site.data, sigvec, psi_prime_site, op_block_A_full
            case "<-":
                A_site = superblock_states[0][psite - 1]
                B_site, sigvec = psi_site.gauge_trf(key="Psi2sigmaB")
                psi_prime_site = np.tensordot(A_site.data, sigvec, axes=(2, 0))
                B_site_full = B_site.thin_to_full(delta_rank=delta_rank)
                superblock_states_trans[0][psite] = B_site
                superblock_states_trans_bra[0][psite] = B_site_full
                op_block_B_full = self.renormalize_op_psite(
                    psite=psite,
                    superblock_states=superblock_states_trans,
                    op_block_states=op_block,
                    ints_site=None,
                    hamiltonian=hamiltonian,
                    A_is_sys=False,
                    superblock_states_bra=superblock_states_trans_bra,
                    superblock_states_ket=None,
                )
                return psi_prime_site, sigvec, psi_site.data, op_block_B_full
            case _:
                raise ValueError(f"{to=} is not valid")

    def get_rank_and_projection_error(
        self,
        psite: int,
        Dmax: int,
        p: float,
        op_sys_full: dict[tuple[int, int], dict[_op_keys, _block_type]],
        op_sys_thin: dict[tuple[int, int], dict[_op_keys, _block_type]],
        op_env_full: dict[tuple[int, int], dict[_op_keys, _block_type]],
        op_env_thin: dict[tuple[int, int], dict[_op_keys, _block_type]],
        hamiltonian: TensorHamiltonian,
        psi_states_left: list[np.ndarray] | list[jax.Array],
        sigvec_states: list[np.ndarray] | list[jax.Array],
        psi_states_right: list[np.ndarray] | list[jax.Array],
        to: Literal["->", "<-"],
    ) -> tuple[int, float]:
        """
        calculate f(D) = |H(D', D)Psi_left|^2 - |K(D)sig|^2 + |H(D, D')Psi_right|^2
        """
        Dmin1, Dmin2 = sigvec_states[0].shape
        # assert Dmin == Dtarget, f"{Dmin=} != {Dtarget=}"
        Dleft, d_left, Dtarget1 = psi_states_left[0].shape
        assert Dmin2 == Dtarget1, f"{Dmin2=} != {Dtarget1=}"
        Dtarget2, d_right, Dright = psi_states_right[0].shape
        assert Dmin1 == Dtarget2, f"{Dmin1=} != {Dtarget2=}"
        Dmax = min((Dmax, Dleft * d_left, Dright * d_right))
        Dmin = min(Dmin1, Dmin2)
        assert Dmin <= Dmax, f"{Dmin=} <= {Dmax=}"
        if Dmin == Dmax:
            return Dmin, 0.0
        total_error_prev: float
        op_env_D = truncate_op_block(op_env_full, Dmax, mode="bra")
        op_sys_D = truncate_op_block(op_sys_full, Dmax, mode="bra")
        op_lcr1 = self.operators_for_superH(
            psite=psite if to == "->" else psite - 1,
            op_sys=op_sys_thin if to == "->" else op_sys_D,
            op_env=op_env_D if to == "->" else op_env_thin,
            ints_site=None,
            hamiltonian=hamiltonian,
            A_is_sys=to == "->",
        )
        op_lcr2 = self.operators_for_superH(
            psite=psite + 1 if to == "->" else psite,
            op_sys=op_sys_D if to == "->" else op_sys_thin,
            op_env=op_env_thin if to == "->" else op_env_D,
            ints_site=None,
            hamiltonian=hamiltonian,
            A_is_sys=to == "->",
        )
        op_lr = self.operators_for_superK(
            psite=psite if to == "->" else psite - 1,
            op_sys=op_sys_D,
            op_env=op_env_D,
            hamiltonian=hamiltonian,
            A_is_sys=to == "->",
        )
        if isinstance(hamiltonian, TensorHamiltonian):
            Heff_left = multiplyH_MPS_direct_MPO(
                op_lcr1,
                psi_states_left,
                hamiltonian,
                tensor_shapes_out=(
                    psi_states_left[0].shape[0],
                    psi_states_left[0].shape[1],
                    Dmax,
                ),
            )
            Heff_right = multiplyH_MPS_direct_MPO(
                op_lcr2,
                psi_states_right,
                hamiltonian,
                tensor_shapes_out=(
                    Dmax,
                    psi_states_right[0].shape[1],
                    psi_states_right[0].shape[2],
                ),
            )
            Keff = multiplyK_MPS_direct_MPO(
                op_lr,
                sigvec_states,
                hamiltonian,
                tensor_shapes_out=(Dmax, Dmax),
            )
        else:
            raise NotImplementedError("only support TensorHamiltonian")
        psi_states_left_in = psi_states_left
        psi_states_right_in = psi_states_right
        sigvec_states_in = sigvec_states
        psi_states_left_out = Heff_left.dot(psi_states_left_in)
        # psi_states_left_vec = Heff_left.stack(
        #     psi_states_left_out, extend=False
        # )
        psi_states_right_out = Heff_right.dot(psi_states_right_in)
        # psi_states_right_vec = Heff_right.stack(
        #     psi_states_right_out, extend=False
        # )
        sigvec_states_out = Keff.dot(sigvec_states_in)
        # sigvec_states_vec = Keff.stack(sigvec_states_out, extend=False)

        for D in range(Dmin, Dmax + 1):
            psi_states_left_vec = psi_states_left_out[0][:, :, :D].ravel()
            psi_states_right_vec = psi_states_right_out[0][:D, :, :].ravel()
            sigvec_states_vec = sigvec_states_out[0][:D, :D].ravel()
            Hleft_error = np.inner(
                psi_states_left_vec.conj(), psi_states_left_vec
            ).real.item()
            Hright_error = np.inner(
                psi_states_right_vec.conj(), psi_states_right_vec
            ).real.item()
            K_error = np.inner(
                sigvec_states_vec.conj(), sigvec_states_vec
            ).real.item()
            total_error = Hleft_error - K_error + Hright_error
            # total_error = K_error
            # logger.debug(f"{D=}, {total_error=:4e}")
            if D > Dmin:
                metric = (total_error - total_error_prev) / total_error  # noqa: F821
                # logger.debug(f"ric=:4e}")
                if metric < p:
                    # return D, metric # always increment at least 1
                    return D - 1, metric
            # uv run ruff check --fix --unsafe-fixes may delete following line but it is necessary
            total_error_prev = total_error  # noqa: F841

        return max(Dmin, D), 0.0

    def get_op_block_full(
        self,
        psite: int,
        superblock_states: list[list[SiteCoef]],
        superblock_states_full: list[list[SiteCoef]],
        op_block_previous: dict[tuple[int, int], dict[_op_keys, _block_type]],
        hamiltonian: TensorHamiltonian,
        to: Literal["->", "<-"],
        mode: Literal["bra", "braket"],
    ):
        if len(superblock_states) != 1:
            raise NotImplementedError("only support single superblock")

        match to:
            case "->":
                step = +1
            case "<-":
                step = -1

        match mode:
            case "bra":
                superblock_states_bra: list[list[SiteCoef]] | None = (
                    superblock_states_full
                )
                superblock_states_ket = None
            case "braket":
                superblock_states = superblock_states_full
                superblock_states_bra = None
                superblock_states_ket = None

        block_full = self.renormalize_op_psite(
            psite=psite + step,
            superblock_states=superblock_states,
            op_block_states=op_block_previous,
            ints_site=None,
            hamiltonian=hamiltonian,
            A_is_sys=to == "<-",
            superblock_states_ket=superblock_states_ket,
            superblock_states_bra=superblock_states_bra,
        )
        return block_full

    def get_adaptive_rank_and_block(
        self,
        *,
        psite: int,
        superblock_states: list[list[SiteCoef]],
        superblock_states_full: list[list[SiteCoef]],
        op_env_previous: dict[tuple[int, int], dict[_op_keys, _block_type]],
        hamiltonian: TensorHamiltonian,
        to: Literal["->", "<-"],
    ) -> tuple[
        int,
        float,
        dict[tuple[int, int], dict[_op_keys, _block_type]],
        dict[tuple[int, int], dict[_op_keys, _block_type]],
    ]:
        """
        Get adaptive rank and block

        Args:
            psite (int): Site index
            p (float): Error tolerance
            Dmax (int): Maximum rank
            superblock_states_full (list[list[SiteCoef]]): Full superblock states
            op_env_previous (dict[tuple[int, int], dict[_op_keys, _block_type]]): Previous operator environment
            hamiltonian (TensorHamiltonian): Hamiltonian
            to (Literal["->", "<-"]): Direction

        Returns:
            tuple[
                int,
                float,
                dict[tuple[int, int], dict[_op_keys, _block_type]],
                dict[tuple[int, int], dict[_op_keys, _block_type]]
            ]: Adaptive rank, error, environmental block with new_rank in bra and braket.
        """
        assert len(superblock_states_full) == 1, (
            "only support single superblock"
        )
        op_env_full_braket = self.get_op_block_full(
            psite=psite,
            superblock_states=superblock_states,
            superblock_states_full=superblock_states_full,
            op_block_previous=op_env_previous,
            hamiltonian=hamiltonian,
            to=to,
            mode="braket",
        )
        # logger.debug(f"{op_env_full_braket=}")
        op_env_full_bra = self.get_op_block_full(
            psite=psite,
            superblock_states=superblock_states,
            superblock_states_full=superblock_states_full,
            op_block_previous=op_env_previous,
            hamiltonian=hamiltonian,
            to=to,
            mode="bra",
        )
        if isinstance(self.op_sys_sites, list):
            op_sys_thin = self.op_sys_sites[-1]
        else:
            raise ValueError("op_sys_sites is not list")
        if to == "->":
            # actual_delta_rank = get_actual_delta_rank(
            #     superblock_states[0], psite+1, const.dD
            # )
            Dmax = min(
                const.Dmax,
                # superblock_states[0][psite].data.shape[2] + actual_delta_rank,
                superblock_states_full[0][psite + 1].data.shape[0],
            )
            delta_rank = Dmax - superblock_states[0][psite].data.shape[2]
        else:
            # actual_delta_rank = get_actual_delta_rank(
            #     superblock_states[0], psite-1, const.dD
            # )
            Dmax = min(
                const.Dmax,
                # superblock_states[0][psite].data.shape[0] + actual_delta_rank,
                superblock_states_full[0][psite - 1].data.shape[2],
            )
            delta_rank = Dmax - superblock_states[0][psite].data.shape[0]
        psi_left, sigvec, psi_right, op_sys_full_bra = (
            self.get_psi_sigvec_psi_fullblock(
                psite=psite,
                superblock_states=superblock_states,
                to=to,
                op_block=op_sys_thin,
                delta_rank=delta_rank,
                hamiltonian=hamiltonian,
            )
        )
        psi_states_left = [psi_left]
        sigvec_states = [sigvec]
        psi_states_right = [psi_right]
        try:
            newD, error = self.get_rank_and_projection_error(
                psite=psite,
                Dmax=Dmax,
                p=const.p_proj,
                op_sys_full=op_sys_full_bra,
                op_sys_thin=op_sys_thin,
                op_env_full=op_env_full_bra,
                op_env_thin=op_env_previous,
                hamiltonian=hamiltonian,
                psi_states_left=psi_states_left,
                sigvec_states=sigvec_states,
                psi_states_right=psi_states_right,
                to=to,
            )
        except Exception as e:
            from loguru import logger as _logger

            logger = _logger.bind(name="main")
            logger.error(
                f"{psite=}, {superblock_states_full[0][psite].data.shape=}"
            )
            logger.error(
                f"{psite-1=}, {superblock_states_full[0][psite-1].data.shape=}"
            )
            logger.error(
                f"{psite+1=}, {superblock_states_full[0][psite+1].data.shape=}"
            )
            logger.error(f"{e=}")
            raise e
        # logger.debug(f"{newD=}, {error=:3e}")
        op_env_D_bra = truncate_op_block(op_env_full_bra, newD, mode="bra")
        op_env_D_braket = truncate_op_block(
            op_env_full_braket, newD, mode="braket"
        )
        match to:
            case "->":
                superblock_states[0][psite + 1].data = superblock_states_full[
                    0
                ][psite + 1].data[:newD, :, :]
            case "<-":
                superblock_states[0][psite - 1].data = superblock_states_full[
                    0
                ][psite - 1].data[:, :, :newD]
        return newD, error, op_env_D_bra, op_env_D_braket


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

    @helper.rank0_only
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
        superblock: list[SiteCoef] = []
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
                isite=isite,
                ndim=self.dim_of_sites[isite],
                m_aux_l=m_aux_l,
                m_aux_r=m_aux_r,
                vibstate=weight_vib[isite],
                is_lend=is_lend,
                is_rend=is_rend,
            )

            if site_unitary is not None:
                if (unitary := site_unitary[isite]) is not None:
                    if const.use_jax:
                        matC = SiteCoef(
                            jnp.einsum("abc,bd->adc", matC.data, unitary),
                            gauge="C",
                            isite=isite,
                        )
                    else:
                        matC = SiteCoef(
                            np.einsum("abc,bd->adc", matC, unitary),
                            gauge="C",
                            isite=isite,
                        )
            superblock.append(matC)
        for isite in range(self.nsite - 1, 0, -1):
            matB, sval = superblock[isite].gauge_trf("C2sigmaB")
            superblock[isite] = matB
            matC = superblock[isite - 1]
            if const.use_jax:
                data = jnp.einsum("abc,cd->abd", matC.data, sval)
            else:
                data = np.einsum("abc,cd->abd", matC, sval)
            superblock[isite - 1] = SiteCoef(data, gauge="C", isite=isite - 1)

        if not const.nonHermitian:
            superblock[0] *= scale / np.linalg.norm(superblock[0].data)
        else:
            superblock[0] *= scale
        superblock[0].gauge = "Psi"

        logger.debug("Initial MPS Lattice")
        logger.debug(helper.get_tensornetwork_diagram_MPS(superblock))
        return superblock

    def __repr__(self) -> str:
        return (
            f"LatticeInfo(nsite={self.nsite}, dim_of_sites={self.dim_of_sites})"
        )


def get_C_sval_states_norm(
    matPsi_or_sval_states: list[np.ndarray] | list[jax.Array],
) -> float:
    """matPsi in all electronic states norm

    Args:
        matPsi_or_sval_states (List[np.ndarray | jax.Array]): i-electronic states matPsi or sval

    Returns:
        float: norm
    """
    if const.use_jax:
        norm = math.sqrt(
            np.sum([jnp.linalg.norm(x) ** 2 for x in matPsi_or_sval_states])
        )
    else:
        norm = math.sqrt(
            np.sum([linalg.norm(x.ravel()) ** 2 for x in matPsi_or_sval_states])
        )
    from loguru import logger as _logger

    logger = _logger.bind(name="rank")
    logger.debug(f"{norm=}")
    return norm


def apply_superOp_direct(
    psite: int,
    superblock_states: list[list[SiteCoef]],
    op_lcr,
    matO_cas: HamiltonianMixin,
    superblock_states_ket: list[list[SiteCoef]],
) -> float:
    """concatenate PolynomialHamiltonian & coefficients"""
    matPsi_states_init: list[np.ndarray] | list[jax.Array]
    if const.use_jax:
        matPsi_states_init = [
            superblock_init[psite].data
            for superblock_init in superblock_states_ket
        ]  # type: ignore
    else:
        matPsi_states_init = [
            np.array(superblock_init[psite])
            for superblock_init in superblock_states_ket
        ]

    """exponentiation PolynomialHamiltonian"""
    if isinstance(matO_cas, PolynomialHamiltonian):
        multiplyOp = multiplyH_MPS_direct(
            op_lcr_states=op_lcr,
            psi_states=matPsi_states_init,
            hamiltonian=matO_cas,
        )
    elif isinstance(matO_cas, TensorHamiltonian):
        multiplyOp = multiplyH_MPS_direct_MPO(
            op_lcr_states=op_lcr,
            psi_states=matPsi_states_init,
            hamiltonian=matO_cas,
        )
    else:
        raise NotImplementedError(f"{type(matO_cas)=}")

    matPsi_states_new = multiplyOp.dot(matPsi_states_init)
    norm = get_C_sval_states_norm(matPsi_states_new)
    matPsi_states_new = [x / norm for x in matPsi_states_new]

    """update(over-write) matPsi(psite)"""
    for istate, superblock in enumerate(superblock_states):
        superblock[psite] = SiteCoef(
            data=matPsi_states_new[istate], gauge="Psi", isite=psite
        )
    return norm


def superblock_trans_APsiB_psite(
    psite: int,
    superblock_states: list[list[SiteCoef]],
    *,
    toAPsi: bool,
    regularize: bool = False,
):
    if toAPsi:
        for superblock in superblock_states:
            superblock_trans_PsiB2APsi_psite(
                psite, superblock, regularize=regularize
            )
    else:
        for superblock in superblock_states:
            superblock_trans_APsi2PsiB_psite(
                psite, superblock, regularize=regularize
            )


def superblock_trans_PsiB2APsi_psite(
    psite: int, superblock: list[SiteCoef], *, regularize: bool = False
):
    """..Psi(p) B(p+1).. -> ..A(p) Psi(p+1)"""
    matPsi = superblock[psite]
    matA, sval = matPsi.gauge_trf("Psi2Asigma", regularize)
    matB = superblock[psite + 1]
    if const.use_jax:
        matB.data = jnp.einsum("ij,jbc->ibc", sval, matB.data)
    else:
        matB.data = np.tensordot(sval, matB.data, axes=1)
    matB.gauge = "Psi"
    superblock[psite] = matA


def superblock_trans_APsi2PsiB_psite(
    psite: int, superblock: list[SiteCoef], *, regularize: bool = False
):
    """..A(p-1) Psi(p).. -> ..Psi(p-1) B(p)"""
    matPsi = superblock[psite]
    matB, sval = matPsi.gauge_trf("Psi2sigmaB", regularize)
    matA = superblock[psite - 1]
    if const.use_jax:
        matA.data = jnp.einsum("ijk,kb->ijb", matA.data, sval)
    else:
        matA.data = np.tensordot(matA, sval, axes=1)
    matA.gauge = "Psi"
    superblock[psite] = matB


def superblock_trans_PsiB2AB_psite(
    psite: int, superblock: list[SiteCoef], *, regularize: bool = False
) -> jax.Array | np.ndarray:
    """..Psi(p) B(p+1).. -> ..A(p) sigma(p) B(p+1)"""
    matPsi = superblock[psite]
    matA, sval = matPsi.gauge_trf("Psi2Asigma", regularize)
    superblock[psite] = matA
    return sval


def superblock_trans_APsi2AB_psite(
    psite: int, superblock: list[SiteCoef], *, regularize: bool = False
) -> np.ndarray | jax.Array:
    """..A(p-1) Psi(p).. -> ..A(p-1) B(p)"""
    matPsi = superblock[psite]
    matB, sval = matPsi.gauge_trf("Psi2sigmaB", regularize)
    superblock[psite] = matB
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
            matPsi_A = superblock_A[psite]
            matPsi_B = superblock_B[psite]
            innerdot += (matPsi_A - matPsi_B).norm() ** 2
            if psite < nsite - 1:
                superblock_trans_PsiB2APsi_psite(psite, superblock_A)
                superblock_trans_PsiB2APsi_psite(psite, superblock_B)
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

    |Ψ> = Psi[0] B[1] B[2] ... B[f-1]

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
def _ovlp_single_state_jax(
    bra_cores: list[jax.Array], ket_cores: list[jax.Array]
) -> jax.Array:
    if len(bra_cores) == 1:
        return _ovlp_single_state_jax_from_left(bra_cores, ket_cores)[0, 0]
    center = len(bra_cores) // 2
    # JAX can process asynchrnously
    ovlp_left = _ovlp_single_state_jax_from_left(
        bra_cores[:center], ket_cores[:center]
    )
    ovlp_right = _ovlp_single_state_jax_from_right(
        bra_cores[center:], ket_cores[center:]
    )
    return jnp.einsum("ab,ab->", ovlp_left, ovlp_right)


@jax.jit
def _ovlp_single_state_jax_from_left(
    bra_cores: list[jax.Array], ket_cores: list[jax.Array]
) -> jax.Array:
    a = bra_cores[0]
    b = ket_cores[0]
    block = jnp.einsum("ab,ac->bc", a[0, :, :], b[0, :, :])
    for i in range(1, len(bra_cores)):
        a = bra_cores[i]
        b = ket_cores[i]
        block = jnp.einsum("bed,cef,bc->df", a, b, block)
    return block


@jax.jit
def _ovlp_single_state_jax_from_right(
    bra_cores: list[jax.Array], ket_cores: list[jax.Array]
) -> jax.Array:
    a = bra_cores[-1]
    b = ket_cores[-1]
    block = jnp.einsum("ab,cb->ac", a[:, :, 0], b[:, :, 0])
    for i in range(len(bra_cores) - 2, -1, -1):
        a = bra_cores[i]
        b = ket_cores[i]
        block = jnp.einsum("dea,fec,ac->df", a, b, block)
    return block


def canonicalize(
    superblock: list[SiteCoef],
    orthogonal_center: int,
    incremental: bool = False,
):
    """Canonicalize the MPS at the psite

    Args:
        superblock (List[SiteCoef]): MPS converted into A(1)A(2),...,A(p-1)Psi(p)B(p+1),...,B(f-1) form. The given MPS will be updated.
        orthogonal_center (int): The site index to be orthogonalized `psite`
        incremental (bool): If True, guess current state as A(1)A(2),...,A(q-1)Psi(q)B(q+1),...,B(f-1) form
            and update only difference. If False, treat current state as no-gauge.
    """
    nsite = len(superblock)
    if nsite == 1:
        return
    if incremental:
        current_center = None
        for isite in range(nsite):
            match superblock[isite].gauge:
                case "Psi":
                    if current_center is not None:
                        raise ValueError("There are multiple Psi sites.")
                    current_center = isite
                case "A":
                    if current_center is not None:
                        raise ValueError("Gauge A found in right side of Psi.")
                case "B":
                    if current_center is None:
                        raise ValueError("Gauge B found in right side of Psi.")
                case _:
                    raise ValueError(
                        f"Invalid gauge: {superblock[isite].gauge}"
                    )
        if current_center is None:
            raise ValueError("No Psi site found.")
        assert isinstance(current_center, int)
        if current_center == orthogonal_center:
            return
        elif current_center < orthogonal_center:
            canonicalizeA(
                superblock[current_center : orthogonal_center + 1],
            )
            return
        else:
            canonicalizeB(
                superblock[orthogonal_center : current_center + 1],
            )
            return
    else:
        canonicalizeB(
            superblock[orthogonal_center:]
        )  # Psi(orthogonal_center)B(orthogonal_center+1),...,B(f-1)
        if orthogonal_center == 0:
            return
        canonicalizeA(
            superblock[:orthogonal_center]
        )  # A(0),...,A(orthogonal_center-2)Psi(orthogonal_center-1)
        Psi1 = superblock[orthogonal_center - 1]
        Psi2 = superblock[orthogonal_center]
        lam = CC2ALambdaB(Psi1, Psi2)
        if isinstance(lam, jax.Array):
            data = jnp.einsum("j,jkl->jkl", lam, Psi2.data)
        else:
            data = np.einsum("j,jkl->jkl", lam, Psi2.data)
        Psi2.data = data
        Psi2.gauge = "Psi"


def canonicalizeA(
    superblock: list[SiteCoef],
):
    """

    Convert MPS into A(1)A(2),...,A(end-1)Psi(end) form.

    Args:
        superblock (List[SiteCoef]): MPS converted into A(1)A(2),...,A(end-1)Psi(end). The given MPS will be updated.
    """
    use_jax = isinstance(superblock[0].data, jax.Array)
    sval: jax.Array | np.ndarray | None = None
    if use_jax:
        einsum = jnp.einsum
    else:
        einsum = np.einsum  # type: ignore
    for i, coef in enumerate(superblock):
        if sval is not None:
            coef.data = einsum("ij,jkl->ikl", sval, coef.data)
        coef.gauge = "Psi"
        if i != len(superblock) - 1:
            matA, sval = coef.gauge_trf("Psi2Asigma")
            coef.data = matA.data
            coef.gauge = "A"


def canonicalizeB(
    superblock: list[SiteCoef],
):
    """

    Convert MPS into Psi(1)B(2),...,B(end-1)B(end) form.

    Args:
        superblock (List[SiteCoef]): MPS converted into Psi(1)B(2),...,B(end-1)B(end). The given MPS will be updated.
    """
    use_jax = isinstance(superblock[0].data, jax.Array)
    sval: jax.Array | np.ndarray | None = None
    if use_jax:
        einsum = jnp.einsum
    else:
        einsum = np.einsum  # type: ignore
    for i, coef in enumerate(superblock[::-1]):
        if sval is not None:
            coef.data = einsum("ijk,kl->ijl", coef.data, sval)
        coef.gauge = "Psi"
        if i != len(superblock) - 1:
            matB, sval = coef.gauge_trf("Psi2sigmaB")
            coef.data = matB.data
            coef.gauge = "B"


def CC2ALambdaB(
    left_core: SiteCoef,
    right_core: SiteCoef,
) -> np.ndarray | jax.Array:
    """
    Perform SVD on two-site tensor core

    Args:
        left_core (SiteCoef): left core tensor to be updated
        right_core (SiteCoef): right core tensor to be updated

    Returns:
        np.ndarray | jax.Array: singular values

    """
    use_jax = isinstance(left_core.data, jax.Array)
    a, b, c = left_core.data.shape
    c, d, e = right_core.data.shape
    twodot_data = left_core.data.reshape(a * b, c) @ right_core.data.reshape(
        c, d * e
    )
    if use_jax:
        u, lam, vh = jnp.linalg.svd(twodot_data, full_matrices=False)
    else:
        u, lam, vh = np.linalg.svd(twodot_data, full_matrices=False)
    left_core.data = u[:, :c].reshape(a, b, c)
    left_core.gauge = "A"
    right_core.data = vh[:c, :].reshape(c, d, e)
    right_core.gauge = "B"
    return lam[:c]


def contract_all_superblock(
    superblock: list[SiteCoef],
) -> np.ndarray | jax.Array:
    """
    C[1]C[2]...C[end-1]C[end] to C[1,2,...,end]

    Args:
        superblock (List[SiteCoef]): MPS

    Returns:
        np.ndarray | jax.Array: Contracted tensor
    """
    core = superblock[-1].data[:, :, 0]
    use_jax = isinstance(core, jax.Array)
    if use_jax:
        einsum = jnp.einsum
    else:
        einsum = np.einsum  # type: ignore
    for coef in superblock[-2::-1]:
        core = einsum("ijk,k...->ij...", coef.data, core)
    return core[0, ...]


def truncate_op_block(
    op_block: dict[tuple[int, int], dict[_op_keys, _block_type]],
    D: int,
    mode: Literal["bra", "braket"],
) -> dict[tuple[int, int], dict[_op_keys, _block_type]]:
    op_block_truncated = {}
    for state_key, op_block_state in op_block.items():
        op_block_state_truncated = {}
        for key, value in op_block_state.items():
            assert isinstance(value, np.ndarray | jax.Array)
            assert len(value.shape) == 3, f"{value.shape=} is not 3"
            if mode == "bra":
                if value.shape[0] < D:
                    raise ValueError(
                        f"{value.shape=} is smaller than {D=} for {key=}"
                    )
                op_block_state_truncated[key] = value[:D, :, :]
            elif mode == "braket":
                if value.shape[0] < D or value.shape[2] < D:
                    raise ValueError(
                        f"{value.shape=} is smaller than {D=} for {key=}"
                    )
                op_block_state_truncated[key] = value[:D, :, :D]
            else:
                raise ValueError(f"{mode=} is not valid")
        op_block_truncated[state_key] = op_block_state_truncated
    return op_block_truncated  # type: ignore


def get_superblock_full(
    superblock: list[SiteCoef],
    delta_rank: int,
) -> list[SiteCoef]:
    superblock_full = []
    for isite, core in enumerate(superblock):
        if core.gauge == "Psi":
            superblock_full.append(core.copy())
        else:
            actual_delta_rank = get_actual_delta_rank(
                superblock, isite, delta_rank
            )
            # l, c, r = core.data.shape
            # if core.gauge == "A":
            #     actual_delta_rank = max(0, min(delta_rank, l * c - r))
            # else:
            #     actual_delta_rank = max(0, min(delta_rank, c * r - l))
            superblock_full.append(
                core.thin_to_full(delta_rank=actual_delta_rank)
            )

    return superblock_full


def get_actual_delta_rank(
    superblock: list[SiteCoef],
    isite: int,
    delta_rank: int,
) -> int:
    core = superblock[isite]
    nsite = len(superblock)
    actual_delta_rank = 0
    match core.gauge:
        case "A":
            if isite == nsite - 1:
                pass
            else:
                l1, c1, r1 = core.data.shape
                l2, c2, r2 = superblock[isite + 1].data.shape
                assert l2 == r1
                actual_delta_rank = max(
                    min(delta_rank, min(l1 * c1 - r1, c2 * r2 - l2)), 0
                )
        case "B":
            if isite == 0:
                pass
            else:
                l1, c1, r1 = core.data.shape
                l2, c2, r2 = superblock[isite - 1].data.shape
                assert l1 == r2
                actual_delta_rank = max(
                    min(delta_rank, min(c1 * r1 - l1, l2 * c2 - r2)), 0
                )
        case _:
            raise ValueError(f"{core.gauge=} is not valid")
    return actual_delta_rank


def is_max_rank(coef: SiteCoef, to: Literal["->", "<-"] = "->") -> bool:
    assert const.adaptive
    L, C, R = coef.data.shape
    if to == "->":
        if L * C <= R or R >= const.Dmax:
            return True
    else:
        if L >= C * R or L >= const.Dmax:
            return True
    return False
