"""MPO-based MPS class"""

from __future__ import annotations

import itertools
import math
from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from discvar import HarmonicOscillator as HO
from loguru import logger as _logger

from pytdscf._const_cls import const
from pytdscf._contraction import (
    _block_type,
    _core_type,
    _op_keys,
    contract_with_site,
    contract_with_site_mpo,
)
from pytdscf._mpo_cls import MatrixProductOperators, OperatorCore
from pytdscf._mps_cls import (
    LatticeInfo,
    MPSCoef,
    ints_spf2site_prod,
)
from pytdscf._site_cls import SiteCoef
from pytdscf._spf_cls import SPFInts
from pytdscf.basis.ho import HarmonicOscillator as _HO
from pytdscf.hamiltonian_cls import TensorHamiltonian
from pytdscf.model_cls import Model

logger = _logger.bind(name="main")


class MPSCoefMPO(MPSCoef):
    r"""Matrix Product State Coefficient in Matrix Product Operator Formulation

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

    @classmethod
    def alloc_random(cls, model: Model) -> MPSCoefMPO:
        """Allocate MPS Coefficient randomly.

        Args:
            model (Model) : Your input informations about MPS framework

        Returns:
            MPSCoefMPO : MPSCoefMPO class object

        """

        (
            nstate,
            weight_estate,
            weight_vib,
            m_aux_max,
        ) = super()._get_initial_condition(model)
        lattice_info_states = []
        superblock_states = []

        mps_coef = cls()
        dofs_cas = list(range(model.get_ndof()))
        nspf_list_cas = [
            [model.get_nspf_list(istate)[idof] for idof in dofs_cas]
            for istate in range(model.get_nstate())
        ]
        ndof_per_sites_cas = [1] * len(dofs_cas)

        for istate in range(nstate):
            lattice_info = LatticeInfo.init_by_zip_dofs(
                nspf_list_cas[istate], ndof_per_sites_cas
            )
            weight = weight_estate[istate]
            if model.init_HartreeProduct is not None:
                superblock = lattice_info.alloc_superblock_random(
                    m_aux_max,
                    math.sqrt(weight),
                    core_weight=model.init_HartreeProduct[istate],
                )
            else:
                dvr_unitary = [
                    dvr_prim.get_unitary()
                    if isinstance(
                        dvr_prim := model.get_primbas(istate, idof), HO
                    )
                    or isinstance(dvr_prim, _HO)
                    else None
                    for idof in range(model.get_ndof())
                ]
                superblock = lattice_info.alloc_superblock_random(
                    m_aux_max,
                    math.sqrt(weight),
                    core_weight=weight_vib[istate],
                    site_unitary=dvr_unitary,
                )
            superblock_states.append(superblock)
            lattice_info_states.append(lattice_info)

        mps_coef.dofs_cas = dofs_cas
        mps_coef.lattice_info_states = lattice_info_states
        mps_coef.superblock_states = superblock_states
        mps_coef.nstate = len(lattice_info_states)
        mps_coef.nsite = lattice_info_states[0].nsite
        mps_coef.ndof_per_sites = lattice_info_states[0].ndof_per_sites
        mps_coef.site_is_dof = all(
            ndof_isite == 1 for ndof_isite in mps_coef.ndof_per_sites
        )
        if not mps_coef.site_is_dof:
            raise NotImplementedError
        mps_coef.reshape_mat = {}
        if model.space == "liouville":
            mps_coef.reshape_mat = mps_coef.define_reshape_mat(
                model.subspace_inds
            )
        if model.subspace_inds is not None:
            mps_coef.project_subspace(model.subspace_inds, m_aux_max)

        return mps_coef

    def define_reshape_mat(
        self, subspace_inds: dict[int, tuple[int, ...]] | None
    ):
        """
        Define reshape function for twin-space
        """
        assert len(self.lattice_info_states) == 1, "Only one state is supported"
        superblock = self.superblock_states[0]

        # Use a dedicated dictionary name to store reshape callables.
        reshape_funcs: dict[int, Callable] = {}

        def _reshape_core(
            data: np.ndarray | jax.Array,
            *,
            j_sqrt: int,
            P_inds: tuple[int, ...] | None = None,
        ) -> np.ndarray | jax.Array:
            """
            data.shape = (len(P_inds) if P_inds else j, k)
            ->  (i, j_sqrt, j_sqrt, k)
            """
            if P_inds is None:
                mat = data
                i, _, k = data.shape
            else:
                # when subspace projection is applied
                # Bond dimension can be reduced by projection
                i, _, k = data.shape

                if isinstance(data, jax.Array):
                    # For GPU efficiency, perform direct operations on JAX arrays
                    mat = jnp.zeros(
                        (i, j_sqrt**2, k),
                        dtype=jnp.complex128,
                    )
                    mat = mat.at[:, P_inds, :].set(data)
                else:
                    mat = np.zeros(
                        (i, j_sqrt**2, k),
                        dtype=np.complex128,
                    )
                    mat[:, P_inds, :] = data
            return mat.reshape(i, j_sqrt, j_sqrt, k, order="C")

        for isite, core in enumerate(superblock):
            _, j, _ = core.shape
            j_sqrt = math.isqrt(j)

            if subspace_inds is not None and isite in subspace_inds:
                P_inds = subspace_inds[isite]
                reshape_funcs[isite] = partial(
                    _reshape_core, j_sqrt=j_sqrt, P_inds=P_inds
                )
            else:
                reshape_funcs[isite] = partial(
                    _reshape_core, j_sqrt=j_sqrt, P_inds=None
                )

        return reshape_funcs

    def project_subspace(
        self, subspace_inds: dict[int, tuple[int, ...]], m_aux_max: int
    ):
        assert len(self.lattice_info_states) == 1, "Only one state is supported"
        lattice_info = self.lattice_info_states[0]
        superblock = self.superblock_states[0]
        for isite, P_inds in subspace_inds.items():
            Q_inds = tuple(
                set(np.arange(superblock[isite].shape[1])) - set(P_inds)
            )
            core_Q = superblock[isite].data[:, Q_inds, :]
            if np.allclose(core_Q, np.zeros_like(core_Q)):
                logger.warning(
                    f"Nonzero values are projected out for {isite} site."
                )
            superblock[isite].data = superblock[isite].data[:, P_inds, :]
            lattice_info.dim_of_sites[isite] = len(P_inds)
            lattice_info.nspf_list_sites[isite] = len(P_inds)
        # Recalculate bond dimension
        nsite = len(superblock)
        for isite in range(nsite):
            m_aux_l, m_aux_r = lattice_info.get_bond_dim(isite, m_aux_max)
            superblock[isite].data = superblock[isite].data[
                :m_aux_l, :, :m_aux_r
            ]

    def get_matH_sweep(self, matH: TensorHamiltonian) -> TensorHamiltonian:
        return matH

    def get_matH_tdh(self, matH, op_block_cas):
        raise NotImplementedError
        pass

    def get_matH_cas(self, matH, ints_spf: SPFInts):
        raise NotImplementedError
        return self.get_hamiltonian(matH, ints_spf)

    def get_hamiltonian(self, matH, ints_spf: SPFInts):
        raise NotImplementedError
        pass

    def get_ints_site(
        self, ints_spf: SPFInts | None
    ) -> (
        dict[tuple[int, int], dict[str, list[np.ndarray] | list[jax.Array]]]
        | None
    ):
        """Get integral between p-site bra and p-site ket in all states pair

        Args:
            ints_spf (SPFInts): SPF integrals. Only supported onesite operator integral.

        Returns:
            Dict[Tuple[int,int],Dict[str, np.ndarray | jax.Array]]: Site integrals
        """
        if ints_spf is None:
            return None
        ints_spf_ovlp = (
            None if "ovlp" not in ints_spf.op_keys() else ints_spf["ovlp"]
        )
        ints_spf_autocorr = (
            None if "auto" not in ints_spf.op_keys() else ints_spf["auto"]
        )
        lattice_info_states = self.lattice_info_states
        ints_site = {}
        for (istate_bra, lattice_info_bra), (
            istate_ket,
            lattice_info_ket,
        ) in itertools.product(enumerate(lattice_info_states), repeat=2):
            ints_site_ops = {}
            statepair = (istate_bra, istate_ket)
            isDiag = istate_bra == istate_ket

            """<n1|Op|n2> for each site"""
            if ints_spf_autocorr and isDiag:
                if const.use_jax:
                    ints_spf_autocorr_cas = [
                        jnp.array(
                            ints_spf_autocorr[statepair][idof],
                            dtype=jnp.complex128,
                        )
                        for idof in self.dofs_cas
                    ]
                else:
                    ints_spf_autocorr_cas = [
                        ints_spf_autocorr[statepair][idof]
                        for idof in self.dofs_cas
                    ]
                ints_site_ops["auto"] = ints_spf2site_prod(
                    ints_spf_autocorr_cas,
                    lattice_info_bra.nspf_list_sites,
                    lattice_info_ket.nspf_list_sites,
                )
            elif ints_spf_ovlp:
                if const.use_jax:
                    ints_spf_ovlp_cas = [
                        jnp.array(
                            ints_spf_ovlp[statepair][idof], dtype=jnp.complex128
                        )
                        for idof in self.dofs_cas
                    ]
                else:
                    ints_spf_ovlp_cas = [
                        ints_spf_ovlp[statepair][idof] for idof in self.dofs_cas
                    ]
                ints_site_ops["ovlp"] = ints_spf2site_prod(
                    ints_spf_ovlp_cas,
                    lattice_info_bra.nspf_list_sites,
                    lattice_info_ket.nspf_list_sites,
                )
                for i in range(len(ints_site_ops["ovlp"])):
                    ints_site_idof = ints_site_ops["ovlp"][i]
                    if np.allclose(
                        ints_site_idof, np.eye(*ints_site_idof.shape)
                    ):
                        if isinstance(ints_site_idof, np.ndarray):
                            ints_site_ops["ovlp"][i] = myndarray(
                                ints_site_idof, is_identity=True
                            )
                        else:
                            ints_site_idof.is_identity = True
                    else:
                        if isinstance(ints_site_idof, np.ndarray):
                            ints_site_ops["ovlp"][i] = myndarray(
                                ints_site_idof, is_identity=False
                            )
                        else:
                            ints_site_idof.is_identity = False
            if len(ints_site_ops) != 0:
                ints_site[statepair] = ints_site_ops
        return ints_site

    def construct_mfop_MPS(
        self,
        psite: int,
        op_sys: dict[tuple[int, int], dict[str, np.ndarray | jax.Array]],
        op_env: dict[tuple[int, int], dict[str, np.ndarray | jax.Array]],
        hamiltonian,
        A_is_sys: bool,
        ints_spf: SPFInts | None = None,
        mps_coef_bra=None,
    ):
        raise NotImplementedError

    def construct_mfop(self, ints_spf: SPFInts, matH):
        raise NotImplementedError

    def construct_mfop_along_sweep(
        self,
        ints_site: dict[tuple[int, int], dict[str, np.ndarray]],
        hamiltonian,
        *,
        A_is_sys: bool,
    ):
        raise NotImplementedError
        pass

    def construct_mfop_along_sweep_TEMP4DIPOLE(
        self,
        ints_site: dict[tuple[int, int], dict[str, np.ndarray]],
        matO_cas,
        *,
        A_is_sys: bool,
        mps_coef_bra=None,
    ):
        raise NotImplementedError
        pass

    def construct_op_zerosite(
        self,
        superblock_states: list[list[SiteCoef]],
        operator: TensorHamiltonian | None = None,
    ) -> dict[tuple[int, int], dict[_op_keys, _block_type]]:
        """initialize op_block_psites
        Args:
            superblock_states (List[List[SiteCoef]]) : Super Blocks (Tensor Cores) of each electronic states
            operator (Optional[TensorHamiltonian]) : Operator (such as Hamiltonian). Defaults to None.

        Returns:
            Dict[Tuple[int,int], Dict[str, np.ndarray | jax.Array]] : block operator. 'ovlp' and 'auto'\
                operator are 2-rank tensor, 'diag_mpo' is 3-rank tensor, 'nondiag_mpo' is 4-rank tensor. \
                '~_summed' means complementary operator.
        """

        autocorr_only = operator is None

        op_block_states = {}
        for istate_bra, istate_ket in itertools.product(
            range(len(superblock_states)), repeat=2
        ):
            op_block_ops = {}
            statepair = (istate_bra, istate_ket)
            isDiag = istate_bra == istate_ket

            if autocorr_only:
                if isDiag:
                    if const.use_jax:
                        ones = get_ones()
                        ones.is_identity = True
                    else:
                        ones = myndarray(
                            np.ones((1, 1, 1), dtype=complex), is_identity=True
                        )
                    op_block_ops["auto"] = ones
            else:
                assert isinstance(operator, TensorHamiltonian)
                if (
                    isDiag
                    or operator.coupleJ[istate_bra][istate_ket] != 0.0
                    or operator.mpo[istate_bra][istate_ket] is not None
                ):
                    if const.use_jax:
                        ones = get_ones()
                        ones.is_identity = True
                    else:
                        ones = myndarray(
                            np.ones((1, 1, 1), dtype=complex), is_identity=True
                        )
                    op_block_ops["ovlp"] = ones

            if len(op_block_ops) != 0:
                op_block_states[statepair] = op_block_ops

        return op_block_states  # type: ignore

    @staticmethod
    def renormalize_op_psite(
        *,
        psite: int,
        superblock_states: list[list[SiteCoef]],
        op_block_states: dict[
            tuple[int, int],
            dict[_op_keys, _block_type],
        ],
        ints_site: dict[tuple[int, int], dict[str, np.ndarray | jax.Array]]
        | None,
        hamiltonian: TensorHamiltonian,
        A_is_sys: bool,
        superblock_states_ket: list[list[SiteCoef]] | None = None,
        superblock_states_bra: list[list[SiteCoef]] | None = None,
    ) -> dict[tuple[int, int], dict[_op_keys, _block_type]]:
        """Contract with MPO and MPS and MPS renormalization

        Only grid-based DVR MPS-Standard Method is supported.

        Args:
            psite (int): site index
            superblock_states (List[List[SiteCoef]]): tensor core
            op_block_states (Dict[Tuple[int,int], Dict[_op_keys, _block_type]]): operator on psite
            ints_site (Dict[Tuple[int,int], Dict[str, np.ndarray | jax.Array]]): integral between p-site.\
                operator is only 'ovlp' or 'auto'
            hamiltonian (TensorHamiltonian): hamiltonian in mpo formulation.
            A_is_sys (bool): Left block is MPS system block.
            superblock_states_ket (List[List[SiteCoef]]): tensor core for ket state

        Returns:
            Dict[Tuple[int,int], Dict[_op_keys, _block_type]]: [i,j]['foo'] = \
                contracted operator named 'foo' between i-state p-site and j-state p-site
        """

        def _get_block_next_auto(
            op_block_statepair: dict[_op_keys, Any],
            ints_site_statepair: dict[str, Any] | None,
            psite: int,
            matLorR_bra: SiteCoef,
            matLorR_ket: SiteCoef,
        ) -> np.ndarray | jax.Array:
            op_block_auto = op_block_statepair["auto"]
            if ints_site_statepair is None:
                op_psite_auto = matLorR_bra.shape[1]
            else:
                op_psite_auto = ints_site_statepair["auto"][psite]
            return contract_with_site(
                matLorR_bra.conj(),
                matLorR_ket,
                op_block_auto,
                op_psite_auto,
            )

        def _get_block_next_ovlp(
            op_block_statepair: dict[_op_keys, Any],
            ints_site_statepair: dict[str, Any] | None,
            psite: int,
            matLorR_bra: SiteCoef,
            matLorR_ket: SiteCoef,
            isDiag: bool,
            bra_is_ket: bool,
        ) -> tuple[np.ndarray | jax.Array, int, int | np.ndarray | jax.Array]:
            op_block_ovlp = op_block_statepair["ovlp"]
            if ints_site_statepair is None:
                op_psite_ovlp = matLorR_bra.shape[1]
            else:
                op_psite_ovlp = ints_site_statepair["ovlp"][psite]
            is_identity_next = True
            if op_block_ovlp.is_identity:
                # np.testing.assert_allclose(op_block_ovlp, np.eye(*op_block_ovlp.shape)):
                assert isinstance(op_block_ovlp, np.ndarray | jax.Array)
                op_block_ovlp = op_block_ovlp.shape[0]
            else:
                is_identity_next = False
            if not bra_is_ket:
                is_identity_next = False
            elif isinstance(op_psite_ovlp, int):
                pass
            elif op_psite_ovlp.is_identity:
                # np.testinig.assert_allclose(op_psite_ovlp, np.eye(*op_psite_ovlp.shape)):
                assert isinstance(op_psite_ovlp, np.ndarray | jax.Array), (
                    f"{op_psite_ovlp=}"
                )
                op_psite_ovlp = op_psite_ovlp.shape[0]
            else:
                is_identity_next = False
            is_identity_next &= isDiag
            result: np.ndarray | jax.Array
            if is_identity_next:
                if matLorR_bra.gauge == "A":
                    new_shape = (
                        matLorR_bra.shape[2],
                        matLorR_ket.shape[2],
                    )
                else:
                    new_shape = (matLorR_bra.shape[0], matLorR_ket.shape[0])
                if const.use_jax:
                    result = jnp.eye(*new_shape, dtype=jnp.complex128)[
                        :, jnp.newaxis, :
                    ]
                else:
                    result = np.eye(*new_shape, dtype=complex)[:, np.newaxis, :]
            else:
                result = contract_with_site_mpo(
                    matLorR_bra, matLorR_ket, op_block_ovlp, op_psite_ovlp
                )
            if isinstance(result, np.ndarray):
                result = myndarray(result, is_identity_next)
            else:
                result.is_identity = is_identity_next
            return result, op_psite_ovlp, op_block_ovlp

        def _set_block_next_key(
            op_psite_mpo: OperatorCore,
            op_block_statepair: dict[_op_keys, Any],
            op_block_ovlp: _block_type,
            matLorR_bra: SiteCoef,
            matLorR_ket: SiteCoef,
            A_is_sys: bool,
            op_block_next_ops: dict[_op_keys, Any],
        ) -> None:
            if (key := op_psite_mpo.key) == "ovlp":
                raise ValueError("key 'ovlp' is not expected")

            if (op_psite_mpo.is_left_side and A_is_sys) or (
                op_psite_mpo.is_right_side and not A_is_sys
            ):
                """skip canonical block"""
                op_block_mpo = op_block_ovlp
            else:
                op_block_mpo = op_block_statepair[key]
            contracted_system = contract_with_site_mpo(
                mat_bra=matLorR_bra,
                mat_ket=matLorR_ket,
                op_LorR=op_block_mpo,
                op_site=op_psite_mpo,
            )
            if (op_psite_mpo.is_right_side and A_is_sys) or (
                op_psite_mpo.is_left_side and not A_is_sys
            ):
                """summed_up"""
                if "summed" in op_block_next_ops:
                    op_block_next_ops["summed"] += contracted_system
                else:
                    op_block_next_ops["summed"] = contracted_system
            else:
                op_block_next_ops[key] = contracted_system

        def _set_block_next_summed(
            op_block_statepair: dict[str | _op_keys, Any],
            op_psite_ovlp: int,
            matLorR_bra: SiteCoef,
            matLorR_ket: SiteCoef,
            op_block_next_ops: dict[str | _op_keys, Any],
        ) -> None:
            """Set block summed operator

            Args:
                op_block_statepair: Operator block for state pair
                op_psite_ovlp: Operator point site overlap
                matLorR_bra: Left or right matrix for bra
                matLorR_ket: Left or right matrix for ket
                op_block_next_ops: Next operator blocks
            """
            op_block_summed = op_block_statepair["summed"]
            assert isinstance(op_psite_ovlp, int), f"{op_psite_ovlp=}"
            contracted_system_summed = contract_with_site_mpo(
                matLorR_bra,
                matLorR_ket,
                op_block_summed,
                op_psite_ovlp,
            )
            if "summed" in op_block_next_ops:
                op_block_next_ops["summed"] += contracted_system_summed
            else:
                op_block_next_ops["summed"] = contracted_system_summed

        if ket_is_not_custom := (superblock_states_ket is None):
            superblock_states_ket = superblock_states
        else:
            if not const.standard_method:
                raise NotImplementedError
        if bra_is_not_custom := (superblock_states_bra is None):
            superblock_states_bra = superblock_states
        else:
            if not const.standard_method:
                raise NotImplementedError
        bra_is_ket = bra_is_not_custom and ket_is_not_custom

        op_block_next = {}
        for (istate_bra, superblock_bra), (
            istate_ket,
            superblock_ket,
        ) in itertools.product(
            enumerate(superblock_states_bra), enumerate(superblock_states_ket)
        ):
            op_block_next_ops: dict[_op_keys, Any] = {}
            statepair = (istate_bra, istate_ket)
            isDiag = istate_bra == istate_ket

            matLorR_bra = superblock_bra[psite]
            matLorR_ket = superblock_ket[psite]

            if statepair not in op_block_states:
                continue

            op_block_statepair = op_block_states[statepair]
            if ints_site is None:
                ints_site_statepair = None
            else:
                ints_site_statepair = ints_site[statepair]
            if "auto" in op_block_statepair:
                if isDiag:
                    op_block_next_ops["auto"] = _get_block_next_auto(
                        op_block_statepair=op_block_statepair,
                        ints_site_statepair=ints_site_statepair,
                        psite=psite,
                        matLorR_bra=matLorR_bra,
                        matLorR_ket=matLorR_ket,
                    )

            elif "ovlp" in op_block_statepair:
                op_block_next_ops["ovlp"], op_psite_ovlp, op_block_ovlp = (
                    _get_block_next_ovlp(
                        op_block_statepair=op_block_statepair,
                        ints_site_statepair=ints_site_statepair,
                        psite=psite,
                        matLorR_bra=matLorR_bra,
                        matLorR_ket=matLorR_ket,
                        isDiag=isDiag,
                        bra_is_ket=bra_is_ket,
                    )
                )

                if not (
                    isinstance(hamiltonian, TensorHamiltonian)
                    and isinstance(
                        mpos := hamiltonian.mpo[istate_bra][istate_ket],
                        MatrixProductOperators,
                    )
                ):
                    continue

                for op_psite_mpo in mpos.calc_point[psite]:
                    _set_block_next_key(
                        op_psite_mpo=op_psite_mpo,
                        op_block_statepair=op_block_statepair,
                        op_block_ovlp=op_block_ovlp,
                        matLorR_bra=matLorR_bra,
                        matLorR_ket=matLorR_ket,
                        A_is_sys=A_is_sys,
                        op_block_next_ops=op_block_next_ops,
                    )

                if "summed" in op_block_statepair:
                    _set_block_next_summed(
                        op_block_statepair=op_block_statepair,
                        op_psite_ovlp=op_psite_ovlp,
                        matLorR_bra=matLorR_bra,
                        matLorR_ket=matLorR_ket,
                        op_block_next_ops=op_block_next_ops,
                    )

            else:
                raise ValueError(
                    "op_block_statepair must have 'auto' or 'ovlp'"
                )

            if len(op_block_next_ops) != 0:
                op_block_next[statepair] = op_block_next_ops

        return op_block_next

    def operators_for_superH(
        self,
        psite: int,
        op_sys: dict[tuple[int, int], dict[_op_keys, _block_type]],
        op_env: dict[tuple[int, int], dict[_op_keys, _block_type]],
        ints_site: dict[tuple[int, int], dict[str, np.ndarray | jax.Array]]
        | None,
        hamiltonian: TensorHamiltonian,
        A_is_sys: bool,
    ) -> list[
        list[dict[_op_keys, tuple[_block_type, _core_type, _block_type]]]
    ]:
        """LCR operator

        prepare operators for multiplying the full-matrix PolynomialHamiltonian on-the-fly

        Args:
            psite (int): site index on "Psi"
            op_sys (Dict[Tuple[int,int], Dict[_op_keys, np.ndarray | jax.Array]]): System operator
            op_env (Dict[Tuple[int,int], Dict[_op_keys, np.ndarray | jax.Array]]): Environment operator
            ints_site (Dict[Tuple[int,int],Dict[str, np.ndarray]]): Site integral
            hamiltonian (TensorHamiltonian) : Hamiltonian
            A_is_sys (bool): Whether left block is System

        Returns:
            like op_lcr[i-bra-state][j-ket-state][(0,1,2)] =  (op_l, op_c, op_r)
        """

        def _get_op_ovlp(
            op_sys: dict[tuple[int, int], dict[_op_keys, _block_type]],
            op_env: dict[tuple[int, int], dict[_op_keys, _block_type]],
            ints_site: dict[tuple[int, int], dict[str, np.ndarray | jax.Array]]
            | None,
            statepair: tuple[int, int],
            psite: int,
            A_is_sys: bool,
        ) -> tuple[_block_type, _block_type, _block_type]:
            op_l_ovlp = (
                op_sys[statepair]["ovlp"]
                if A_is_sys
                else op_env[statepair]["ovlp"]
            )
            op_r_ovlp = (
                op_env[statepair]["ovlp"]
                if A_is_sys
                else op_sys[statepair]["ovlp"]
            )
            if ints_site is None:
                op_c_ovlp = 0
            else:
                op_c_ovlp = ints_site[statepair]["ovlp"][psite]  # type: ignore

            # type(op) is 'int' if it is a unit-matrix --> already applied bra != ket spfs.
            if op_l_ovlp.is_identity:  # type: ignore
                # np.testing.assert_allclose(op_l_ovlp, np.eye(*op_l_ovlp.shape)):
                assert isinstance(op_l_ovlp, np.ndarray | jax.Array)
                op_l_ovlp = op_l_ovlp.shape[0]
            if ints_site is not None and op_c_ovlp.is_identity:
                # np.testing.assert_allclose(op_c_ovlp, np.eye(*op_c_ovlp.shape)):
                assert isinstance(op_c_ovlp, np.ndarray | jax.Array)
                op_c_ovlp = op_c_ovlp.shape[0]
            if op_r_ovlp.is_identity:  # type: ignore
                # np.testing.assert_allclose(op_r_ovlp, np.eye(*op_r_ovlp.shape)):
                assert isinstance(op_r_ovlp, np.ndarray | jax.Array)
                op_r_ovlp = op_r_ovlp.shape[0]

            return op_l_ovlp, op_c_ovlp, op_r_ovlp

        def _set_op_lcr_states_key(
            op_lcr_states: dict[
                _op_keys, tuple[_block_type, _core_type, _block_type]
            ],
            op_l: dict[_op_keys, _block_type],
            op_r: dict[_op_keys, _block_type],
            op_c_core: list[OperatorCore],
            op_l_ovlp: _block_type,
            op_c_ovlp: _core_type,
            op_r_ovlp: _block_type,
        ) -> None:
            if "summed" in op_l:
                op_lcr_states["summ_l"] = (
                    op_l["summed"],
                    op_c_ovlp,
                    op_r_ovlp,
                )
            if "summed" in op_r:
                op_lcr_states["summ_r"] = (
                    op_l_ovlp,
                    op_c_ovlp,
                    op_r["summed"],
                )
            for core in op_c_core:
                key = core.key
                op_r_key = op_r[key] if key in op_r else op_r_ovlp
                op_l_key = op_l[key] if key in op_l else op_l_ovlp
                op_lcr_states[key] = (
                    op_l_key,
                    core,
                    op_r_key,
                )

        nstate = len(hamiltonian.coupleJ)
        op_lcr: list[
            list[
                dict[
                    _op_keys,
                    tuple[
                        _block_type,
                        _core_type,
                        _block_type,
                    ],
                ]
            ]
        ]
        op_lcr = [[None for j in range(nstate)] for i in range(nstate)]  # type: ignore
        for istate_bra, istate_ket in itertools.product(
            list(range(nstate)), repeat=2
        ):
            coupleJ = hamiltonian.coupleJ[istate_bra][istate_ket]
            statepair = (istate_bra, istate_ket)
            isDiag = istate_bra == istate_ket

            op_l_ovlp: _block_type
            op_c_ovlp: _core_type
            op_r_ovlp: _block_type
            mpos = hamiltonian.mpo[istate_bra][istate_ket]
            if (
                coupleJ != 0.0
                or isDiag
                or isinstance(mpos, MatrixProductOperators)
            ):
                op_l_ovlp, op_c_ovlp, op_r_ovlp = _get_op_ovlp(
                    op_sys, op_env, ints_site, statepair, psite, A_is_sys
                )
                op_lcr[istate_bra][istate_ket] = {
                    "ovlp": (op_l_ovlp, op_c_ovlp, op_r_ovlp)
                }
                op_l = op_sys[statepair] if A_is_sys else op_env[statepair]
                op_r = op_env[statepair] if A_is_sys else op_sys[statepair]
                assert isinstance(mpos, MatrixProductOperators)
                op_c_core: list[OperatorCore] = mpos.calc_point[psite]

                _set_op_lcr_states_key(
                    op_lcr_states=op_lcr[istate_bra][istate_ket],
                    op_l=op_l,
                    op_r=op_r,
                    op_c_core=op_c_core,
                    op_l_ovlp=op_l_ovlp,
                    op_c_ovlp=op_c_ovlp,
                    op_r_ovlp=op_r_ovlp,
                )

        return op_lcr

    def operators_for_superK(
        self,
        psite: int,
        op_sys: dict[tuple[int, int], dict[_op_keys, _block_type]],
        op_env: dict[tuple[int, int], dict[_op_keys, _block_type]],
        hamiltonian: TensorHamiltonian,
        A_is_sys: bool,
    ) -> list[list[dict[_op_keys, tuple[_block_type, _block_type]]]]:
        """ LsR operator

        construct full-matrix Kamiltonian

        Args:
            psite (int): site index on "Psi"
            op_sys (Dict[Tuple[int,int], Dict[_op_keys, np.ndarray | jax.Array]]): System operator
            op_env (Dict[Tuple[int,int], Dict[_op_keys, np.ndarray | jax.Array]]): Environment operator
            hamiltonian (PolynomialHamiltonian) : Hamiltonian
            A_is_sys (bool): Whether left block is System

        Returns:
            List[List[Dict[_op_keys, Tuple[np.ndarray, np.ndarray, np.ndarray]]]]: \
                [i-bra-state][j-ket-state][(0,1,2)] =  (op_l, op_r)
        """

        def _get_op_ovlp(
            op_sys: dict[tuple[int, int], dict[_op_keys, _block_type]],
            op_env: dict[tuple[int, int], dict[_op_keys, _block_type]],
            statepair: tuple[int, int],
            A_is_sys: bool,
        ) -> tuple[_block_type, _block_type]:
            op_l_ovlp = (
                op_sys[statepair]["ovlp"]
                if A_is_sys
                else op_env[statepair]["ovlp"]
            )
            op_r_ovlp = (
                op_env[statepair]["ovlp"]
                if A_is_sys
                else op_sys[statepair]["ovlp"]
            )

            # type(op) is 'int' if it is a unit-matrix --> already applied bra != ket spfs.
            if op_l_ovlp.is_identity:  # type: ignore
                # np.testing.assert_allclose(op_l_ovlp, np.eye(*op_l_ovlp.shape)):
                assert isinstance(op_l_ovlp, np.ndarray | jax.Array)
                op_l_ovlp = op_l_ovlp.shape[0]
            if op_r_ovlp.is_identity:  # type: ignore
                # np.testing.assert_allclose(op_r_ovlp, np.eye(*op_r_ovlp.shape)):
                assert isinstance(op_r_ovlp, np.ndarray | jax.Array)
                op_r_ovlp = op_r_ovlp.shape[0]

            return op_l_ovlp, op_r_ovlp

        def _set_op_lr_states_key(
            op_lr_state: dict[_op_keys, tuple[_block_type, _block_type]] | None,
            op_l: dict[_op_keys, _block_type],
            op_r: dict[_op_keys, _block_type],
            op_l_ovlp: _block_type,
            op_r_ovlp: _block_type,
            keys,
            psite: int,
        ) -> None:
            """Set operator blocks for left and right parts of the system

            Args:
                op_lr_state: Operator blocks for specific state pair
                op_l: Left operator block
                op_r: Right operator block
                op_l_ovlp: Left overlap operator
                op_r_ovlp: Right overlap operator
                op_sys: System operator blocks for specific state pair
                psite: Current site index
            """
            if "summed" in op_l:
                if op_lr_state is None:
                    op_lr_state = {}
                op_lr_state["summ_l"] = (
                    op_l["summed"],
                    op_r_ovlp,
                )
            if "summed" in op_r:
                if op_lr_state is None:
                    op_lr_state = {}
                op_lr_state["summ_r"] = (
                    op_l_ovlp,
                    op_r["summed"],
                )
            for key in keys:
                if key in ["summed", "ovlp"]:
                    # Already included in 'summed' or 'ovlp' above
                    continue
                op_r_key = op_r[key] if key in op_r else op_r_ovlp
                op_l_key = op_l[key] if key in op_l else op_l_ovlp
                assert key in op_r, (
                    f"Right side key {key} should be included in summed at {psite}-site"
                )
                assert key in op_l, (
                    f"Left side key {key} should be included in summed at {psite}-site"
                )
                if op_lr_state is None:
                    op_lr_state = {key: (op_l_key, op_r_key)}
                else:
                    op_lr_state[key] = (
                        op_l_key,
                        op_r_key,
                    )

        nstate = len(hamiltonian.coupleJ)
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
        ]
        op_lr = [[None for j in range(nstate)] for i in range(nstate)]  # type: ignore

        op_l_ovlp: _block_type
        op_r_ovlp: _block_type

        for istate_bra, istate_ket in itertools.product(
            list(range(nstate)), repeat=2
        ):
            coupleJ = hamiltonian.coupleJ[istate_bra][istate_ket]
            statepair = (istate_bra, istate_ket)
            isDiag = istate_bra == istate_ket

            mpos = hamiltonian.mpo[istate_bra][istate_ket]
            if (
                coupleJ != 0.0
                or isDiag
                or isinstance(mpos, MatrixProductOperators)
            ):
                op_l_ovlp, op_r_ovlp = _get_op_ovlp(
                    op_sys=op_sys,
                    op_env=op_env,
                    statepair=statepair,
                    A_is_sys=A_is_sys,
                )

                op_lr[istate_bra][istate_ket] = {"ovlp": (op_l_ovlp, op_r_ovlp)}

                op_l = op_sys[statepair] if A_is_sys else op_env[statepair]
                op_r = op_env[statepair] if A_is_sys else op_sys[statepair]

                _set_op_lr_states_key(
                    op_lr_state=op_lr[istate_bra][istate_ket],
                    op_l=op_l,
                    op_r=op_r,
                    op_l_ovlp=op_l_ovlp,
                    op_r_ovlp=op_r_ovlp,
                    keys=op_sys[statepair].keys(),
                    psite=psite,
                )

        return op_lr

    def grid_pop(self, J: tuple[int, ...], istate=0) -> float:
        """Calculate grid population

        Args:
            J (Tuple[int]): grid index
            istate (int): i-electronic states

        Returns:
            float: population
        """
        assert const.doDVR
        assert const.standard_method
        assert const.use_jax
        superblock = self.superblock_states[istate]
        nsite = len(superblock)
        pop = superblock[0].data.numpy()[:, J[0], :]
        for isite in range(1, nsite):
            pop = np.einsum(
                "ab,bc->ac", pop, superblock[isite].data.numpy()[:, J[isite], :]
            )
        return float(np.linalg.norm(pop)) ** 2


class myndarray(np.ndarray):
    """
    Numpy ndarray with additional attribute
    """

    def __new__(cls, input_array, is_identity=None):
        obj = np.asarray(input_array).view(cls)
        obj.is_identity = is_identity
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.is_identity = getattr(obj, "is_identity", None)

    def __repr__(self):
        return f"myndarray(shape={self.shape}, is_identity={self.is_identity})"


@jax.jit
def get_ones():
    return jnp.ones((1, 1, 1), dtype=jnp.complex128)
