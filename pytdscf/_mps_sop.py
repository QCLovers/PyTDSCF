"""MPS which supports Sum of Products (SOP) form, especially polynomial Hamiltonian"""

from __future__ import annotations

import copy
import itertools
import math
from collections import defaultdict, namedtuple
from logging import getLogger
from time import time

import jax
import jax.numpy as jnp
import numpy as np

import pytdscf._helper as helper
from pytdscf._const_cls import const
from pytdscf._contraction import (
    _block_type,
    contract_with_site,
    contract_with_site_concat,
    mfop_site,
    mfop_site_concat,
)
from pytdscf._mps_cls import (
    LatticeInfo,
    MPSCoef,
    ints_spf2site_prod,
    ints_spf2site_sum,
    superblock_trans_APsiB_psite,
)
from pytdscf._site_cls import SiteCoef
from pytdscf._spf_cls import SPFInts
from pytdscf.hamiltonian_cls import (
    PolynomialHamiltonian,
    TermOneSiteForm,
    TermProductForm,
)
from pytdscf.model_cls import Model

logger = getLogger("main").getChild(__name__)


def construct_matH_general_at_psite(matH, psite: int, A_is_sys):
    matH_general_renorm_summed = [
        [[] for j in range(matH.nstate)] for i in range(matH.nstate)
    ]
    matH_general_renorm_byterm = [
        [[] for j in range(matH.nstate)] for i in range(matH.nstate)
    ]
    matH_general_superH_byterm = [
        [[] for j in range(matH.nstate)] for i in range(matH.nstate)
    ]
    matH_general_superK_byterm = [
        [[] for j in range(matH.nstate)] for i in range(matH.nstate)
    ]
    for istate_bra, istate_ket in itertools.product(
        range(matH.nstate), repeat=2
    ):
        terms_renorm_summed = []
        terms_renorm_byterm = []
        terms_superH_byterm = []
        terms_superK_byterm = []

        for term_prod in matH.general[istate_bra][istate_ket]:
            key_sys = "left" if A_is_sys else "right"
            key_env = "right" if A_is_sys else "left"

            site_op_key = term_prod.blockop_key_sites["centr"][psite]

            """renormalize_op_psite"""
            if not term_prod.is_op_ovlp([key_env], psite):
                terms_renorm_byterm.append(term_prod)
            if (
                term_prod.is_op_ovlp([key_env], psite)
                and not site_op_key == "ovlp"
            ):
                terms_renorm_summed.append(term_prod)

            """operators_for_superH & construct_mfop_MPS"""
            if term_prod.is_op_ovlp([key_sys, "centr"], psite):
                pass
            elif term_prod.is_op_ovlp([key_env, "centr"], psite):
                pass
            else:
                terms_superH_byterm.append(term_prod)

            """operators_for_superK"""
            if term_prod.is_op_ovlp([key_sys, "centr"], psite):
                pass
            elif term_prod.is_op_ovlp([key_env], psite):
                pass
            else:
                terms_superK_byterm.append(term_prod)

        matH_general_renorm_summed[istate_bra][istate_ket] = terms_renorm_summed
        matH_general_renorm_byterm[istate_bra][istate_ket] = terms_renorm_byterm
        matH_general_superH_byterm[istate_bra][istate_ket] = terms_superH_byterm
        matH_general_superK_byterm[istate_bra][istate_ket] = terms_superK_byterm

    return (
        matH_general_renorm_summed,
        matH_general_renorm_byterm,
        matH_general_superH_byterm,
        matH_general_superK_byterm,
    )


class MPSCoefSoP(MPSCoef):
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

    @classmethod
    def alloc_random(cls, model: Model) -> MPSCoefSoP:
        """Allocate MPS Coefficient randomly.

        Args:
            model (Model) : Your input information

        Returns:
            MPSCoef : MPSCoef class object

        """

        (
            nstate,
            lattice_info_states,
            superblock_states,
            weight_estate,
            weight_vib,
            m_aux_max,
        ) = super()._get_initial_condition(model)

        mps_coef = cls()
        if "enable_tdh_dofs" in const.keys:
            assert all(
                ndof_isite == 1 for ndof_isite in model.get_ndof_per_sites()
            )
            dofs_cas = []
            for idof in range(model.get_ndof()):
                if any(
                    [
                        model.get_nspf_list(istate)[idof] != 1
                        for istate in range(model.get_nstate())
                    ]
                ):
                    dofs_cas.append(idof)
        else:
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
            superblock = lattice_info.alloc_superblock_random(
                m_aux_max, math.sqrt(weight), weight_vib[istate]
            )

            superblock_states.append(superblock)
            lattice_info_states.append(lattice_info)

        mps_coef.dofs_cas = dofs_cas
        if "enable_tdh_dofs" in const.keys:
            mps_coef.dofs_tdh = list(
                set(range(model.get_ndof())) - set(dofs_cas)
            )
        mps_coef.lattice_info_states = lattice_info_states
        mps_coef.superblock_states = superblock_states
        mps_coef.nstate = len(lattice_info_states)
        mps_coef.nsite = lattice_info_states[0].nsite
        mps_coef.ndof_per_sites = lattice_info_states[0].ndof_per_sites
        mps_coef.with_matH_general = model.hamiltonian.general is not None
        mps_coef.site_is_dof = all(
            ndof_isite == 1 for ndof_isite in mps_coef.ndof_per_sites
        )
        assert (
            not mps_coef.with_matH_general or mps_coef.site_is_dof
        ), "dof(in ints_spf) and site(in MPS) have to be identical for matH.general"

        return mps_coef

    def get_matH_sweep(
        self, matH: PolynomialHamiltonian
    ) -> PolynomialHamiltonian:
        """
        Args:
            matH (PolynomialHamiltonian): Original Hamiltonian

        Returns:
            PolynomialHamiltonian: Hamiltonian for sweep algorithm, \
                which has `general_new` attributes.
        """
        matH_sweep = PolynomialHamiltonian(
            len(self.dofs_cas), matH.nstate, matH.name
        )
        matH_sweep.coupleJ = copy.deepcopy(matH.coupleJ)
        matH_sweep.onesite = copy.deepcopy(matH.onesite)
        matH_sweep.general = copy.deepcopy(matH.general)

        GeneralTerms = namedtuple(
            "GeneralTerms", ["renorm_summed", "renorm", "superH", "superK"]
        )

        matH_sweep.general_new = {"forward": [], "backward": []}
        for A_is_sys in [True, False]:
            for psite in range(len(self.superblock_states[0])):
                terms = GeneralTerms(
                    *construct_matH_general_at_psite(matH, psite, A_is_sys)
                )
                if A_is_sys:
                    matH_sweep.general_new["forward"].append(terms)
                else:
                    matH_sweep.general_new["backward"].append(terms)

        return matH_sweep

    def get_matH_tdh(self, matH, op_block_cas):
        assert (
            "enable_summed_op" not in const.keys
        ), "CANNOT use summed_op and tdh_dofs at the same time"

        matH_tdh = PolynomialHamiltonian(matH.nstate, len(self.dofs_tdh))

        if matH.onesite:
            for istate_bra, istate_ket in itertools.product(
                range(matH.nstate), repeat=2
            ):
                statepair = (istate_bra, istate_ket)
                for term in matH.onesite[istate_bra][istate_ket]:
                    idof = term.op_dof
                    if idof in self.dofs_tdh:
                        matH_tdh.onesite[istate_bra][istate_ket].append(
                            TermOneSiteForm(
                                term.coef,
                                self.dofs_tdh.index(idof),
                                term.op_key,
                            )
                        )
                    else:
                        matH_tdh.coupleJ[istate_bra][istate_ket] += (
                            op_block_cas[statepair]["onesite"][0, 0]
                        )

        if matH.general:
            for istate_bra, istate_ket in itertools.product(
                range(matH.nstate), repeat=2
            ):
                statepair = (istate_bra, istate_ket)
                terms_general_tdh = {}
                for term_prod in matH.general[istate_bra][istate_ket]:
                    coef_tdh = term_prod.coef
                    op_dofs_cas = []
                    op_dofs_tdh = []
                    op_keys_cas = []
                    op_keys_tdh = []
                    for idof, op_key in term_prod.mode_ops.items():
                        if idof in self.dofs_cas:
                            op_dofs_cas.append(self.dofs_cas.index(idof))
                            op_keys_cas.append(op_key)
                        else:
                            op_dofs_tdh.append(self.dofs_tdh.index(idof))
                            op_keys_tdh.append(op_key)

                    if len(op_dofs_cas) != 0:
                        key_cas = TermProductForm.convert_key(
                            op_dofs_cas, op_keys_cas
                        )
                        assert op_block_cas[statepair][key_cas].shape == (
                            1,
                            1,
                        ), op_block_cas[statepair][key_cas]
                        """<CAS|op[dof_cas]|CAS> => coef"""
                        coef_tdh *= op_block_cas[statepair][key_cas][0, 0]

                    if len(op_dofs_tdh) != 0:
                        key_tdh = TermProductForm.convert_key(
                            op_dofs_tdh, op_keys_tdh
                        )
                        if key_tdh in terms_general_tdh:
                            terms_general_tdh[key_tdh].coef += coef_tdh
                        else:
                            terms_general_tdh[key_tdh] = TermProductForm(
                                coef_tdh, op_dofs_tdh, op_keys_tdh
                            )
                    else:
                        """matH_cas.coupleJ(e1,e2) += <TDH|TDH> <CAS|op[only in dof_cas]|CAS> """
                        matH_tdh.coupleJ[istate_bra][istate_ket] += coef_tdh

                terms_general_tdh_list = list(terms_general_tdh.values())
                for term_prod in terms_general_tdh_list:
                    term_prod.set_blockop_key(matH_tdh.ndof)

                matH_tdh.general[istate_bra][istate_ket] = (
                    terms_general_tdh_list  # type: ignore
                )

        return matH_tdh

    def get_matH_cas(self, matH, ints_spf: SPFInts):
        matH_cas = PolynomialHamiltonian(len(self.dofs_cas), matH.nstate)
        matH_cas.coupleJ = copy.deepcopy(matH.coupleJ)
        matH_cas.onesite = copy.deepcopy(matH.onesite)

        """matH_cas.coupleJ(e1,e2) += <TDH|onesite(e1,e2)[only in dof_tdh]|TDH> """
        if matH.onesite:
            for istate_bra, istate_ket in itertools.product(
                range(matH.nstate), repeat=2
            ):
                statepair = (istate_bra, istate_ket)
                for idof in self.dofs_tdh:
                    matH_cas.coupleJ[istate_bra][istate_ket] += ints_spf[
                        statepair
                    ]["onesite"][idof][0, 0]

        if matH.general:
            for istate_bra, istate_ket in itertools.product(
                range(matH.nstate), repeat=2
            ):
                statepair = (istate_bra, istate_ket)
                terms_general_cas = {}
                for term_prod in matH.general[istate_bra][istate_ket]:
                    coef_cas = term_prod.coef
                    op_dofs_cas = []
                    op_keys_cas = []
                    """<TDH|<CAS|op[dof_tdh] op[dof_cas]|CAS>|TDH>"""
                    for idof, op_key in term_prod.mode_ops.items():
                        if idof in self.dofs_cas:
                            """<CAS|op[dof_cas]|CAS> => op"""
                            op_dofs_cas.append(self.dofs_cas.index(idof))
                            op_keys_cas.append(op_key)
                        else:
                            """<TDH|op[dof_tdh]|TDH> => coef"""
                            coef_cas *= ints_spf[statepair][op_key][idof][0, 0]
                    if len(op_dofs_cas) != 0:
                        key_cas = TermProductForm.convert_key(
                            op_dofs_cas, op_keys_cas
                        )
                        if key_cas in terms_general_cas:
                            terms_general_cas[key_cas].coef += coef_cas
                        else:
                            terms_general_cas[key_cas] = TermProductForm(
                                coef_cas, op_dofs_cas, op_keys_cas
                            )
                    else:
                        """matH_cas.coupleJ(e1,e2) += <TDH|general(e1,e2)[only in dof_tdh]|TDH> """
                        matH_cas.coupleJ[istate_bra][istate_ket] += coef_cas

                terms_general_cas_list = list(terms_general_cas.values())
                for term_prod in terms_general_cas_list:
                    term_prod.set_blockop_key(matH_cas.ndof)

                matH_cas.general[istate_bra][istate_ket] = (
                    terms_general_cas_list  # type: ignore
                )

        return matH_cas

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
        ints_spf_ovlp: dict | None = (
            None if "ovlp" not in ints_spf.op_keys() else ints_spf["ovlp"]
        )
        ints_spf_onesite: dict | None = (
            None
            if onesite_name not in ints_spf.op_keys()
            else ints_spf[onesite_name]
        )
        ints_spf_autocorr = (
            None if "auto" not in ints_spf.op_keys() else ints_spf["auto"]
        )
        lattice_info_states = self.lattice_info_states

        # integral transformation spf->site by reshape
        ints_site = {}
        for (istate_bra, lattice_info_bra), (
            istate_ket,
            lattice_info_ket,
        ) in itertools.product(enumerate(lattice_info_states), repeat=2):
            ints_site_ops = {}
            statepair = (istate_bra, istate_ket)
            isDiag = istate_bra == istate_ket

            """<n1|Op|n2> for each site"""
            if ints_spf_autocorr:
                if isDiag:
                    ints_spf_autocorr_cas = [
                        ints_spf_autocorr[statepair][idof]
                        for idof in self.dofs_cas
                    ]
                    ints_site_ops["auto"] = ints_spf2site_prod(
                        ints_spf_autocorr_cas,
                        lattice_info_bra.nspf_list_sites,
                        lattice_info_ket.nspf_list_sites,
                    )
            else:
                assert isinstance(ints_spf_ovlp, dict)
                ints_spf_ovlp_cas = [
                    ints_spf_ovlp[statepair][idof] for idof in self.dofs_cas
                ]
                ints_site_ops["ovlp"] = ints_spf2site_prod(
                    ints_spf_ovlp_cas,
                    lattice_info_bra.nspf_list_sites,
                    lattice_info_ket.nspf_list_sites,
                )

                assert isinstance(ints_spf_onesite, dict)
                if ints_spf_onesite[statepair] is not None:
                    assert isinstance(ints_spf_onesite[statepair], list)
                    ints_spf_onesite_cas = [
                        ints_spf_onesite[statepair][idof]
                        for idof in self.dofs_cas
                    ]
                    ints_site_ops["onesite"] = ints_spf2site_sum(
                        ints_spf_onesite_cas,
                        lattice_info_bra.nspf_list_sites,
                        lattice_info_ket.nspf_list_sites,
                    )

                if isDiag and self.with_matH_general:
                    for op_key in ints_spf.op_keys():
                        if op_key != "onesite":
                            if const.use_jax:
                                ints_site_ops[op_key] = [
                                    jnp.array(
                                        ints_spf[statepair][op_key][idof],
                                        dtype=jnp.complex128,
                                    )
                                    for idof in self.dofs_cas
                                ]
                            else:
                                ints_site_ops[op_key] = [
                                    ints_spf[statepair][op_key][idof]
                                    for idof in self.dofs_cas
                                ]

            if len(ints_site_ops) != 0:
                ints_site[statepair] = ints_site_ops

        return ints_site

    def construct_mfop_MPS(
        self,
        psite: int,
        op_sys: dict[tuple[int, int], dict[str, np.ndarray | jax.Array]],
        op_env: dict[tuple[int, int], dict[str, np.ndarray | jax.Array]],
        matH_cas: PolynomialHamiltonian,
        A_is_sys: bool,
        ints_spf: SPFInts | None = None,
        mps_coef_ket=None,
    ):
        if mps_coef_ket is None:
            mps_coef_ket = self

        def _mfop_site2spf_prod(
            mfop_site, ints_spf: SPFInts, nspf_list_bra, nspf_list_ket
        ):
            ndof_site = len(nspf_list_bra)
            mfop_spf_site = [
                np.zeros(
                    (nspf_list_bra[kdof_site], nspf_list_ket[kdof_site]),
                    dtype=complex,
                )
                for kdof_site in range(ndof_site)
            ]
            for (n_bra, J_bra), (n_ket, J_ket) in itertools.product(
                enumerate(
                    itertools.product(*[range(x) for x in nspf_list_bra])
                ),
                enumerate(
                    itertools.product(*[range(x) for x in nspf_list_ket])
                ),
            ):
                ints_spf_J = [
                    ints_spf[kdof_site][J_bra[kdof_site], J_ket[kdof_site]]
                    for kdof_site in range(ndof_site)
                ]
                ints_spf_J_reverse = ints_spf_J[::-1]
                # BGN:time consuming part of MFOP
                if const.verbose == 4:
                    helper.ElpTime.mfop_4 -= time()
                ints_cumprod_l = np.cumprod([1.0 + 0.0j] + ints_spf_J[:-1])
                ints_cumprod_r = np.cumprod(
                    [1.0 + 0.0j] + ints_spf_J_reverse[:-1]
                )[::-1]
                if const.verbose == 4:
                    helper.ElpTime.mfop_4 += time()
                # END:time consuming part of MFOP
                for kdof_site, (ispf_bra, ispf_ket) in enumerate(
                    zip(J_bra, J_ket, strict=True)
                ):
                    mfop_spf_site[kdof_site][ispf_bra, ispf_ket] += mfop_site[
                        n_bra, n_ket
                    ] * (
                        ints_cumprod_l[kdof_site] * ints_cumprod_r[kdof_site]
                    )  # prod
            return mfop_spf_site

        def _mfop_site2spf_summ(
            mfop_site, ints_spf: SPFInts, nspf_list_bra, nspf_list_ket
        ):
            ndof_site = len(nspf_list_bra)
            mfop_spf_site = [
                np.zeros(
                    (nspf_list_bra[kdof_site], nspf_list_ket[kdof_site]),
                    dtype=complex,
                )
                for kdof_site in range(ndof_site)
            ]
            for (n_bra, J_bra), (n_ket, J_ket) in itertools.product(
                enumerate(
                    itertools.product(*[range(x) for x in nspf_list_bra])
                ),
                enumerate(
                    itertools.product(*[range(x) for x in nspf_list_ket])
                ),
            ):
                ints_spf_J = [
                    ints_spf[kdof_site][J_bra[kdof_site], J_ket[kdof_site]]
                    for kdof_site in range(ndof_site)
                ]
                ints_spf_J_reverse = ints_spf_J[::-1]
                # BGN:time consuming part of MFOP
                if const.verbose == 4:
                    helper.ElpTime.mfop_4 -= time()
                ints_cumsum_l = np.cumsum([0.0] + ints_spf_J[:-1])
                ints_cumsum_r = np.cumsum([0.0] + ints_spf_J_reverse[:-1])[::-1]
                if const.verbose == 4:
                    helper.ElpTime.mfop_4 += time()
                # END:time consuming part of MFOP
                for kdof_site, (ispf_bra, ispf_ket) in enumerate(
                    zip(J_bra, J_ket, strict=True)
                ):
                    mfop_spf_site[kdof_site][ispf_bra, ispf_ket] += mfop_site[
                        n_bra, n_ket
                    ] * (
                        ints_cumsum_l[kdof_site] + ints_cumsum_r[kdof_site]
                    )  # sum
            return mfop_spf_site

        # BODY ######################################

        superblock_states = self.superblock_states
        superblock_states_ket = mps_coef_ket.superblock_states
        # lattice_info_states = self.lattice_info_states

        mfop_spf_ovlp_psite_states = {}
        mfop_spf_onesite_psite_states = {}
        mfop_spf_general_psite_states = {}
        for (istate_bra, superblock_bra), (
            istate_ket,
            superblock_ket,
        ) in itertools.product(
            enumerate(superblock_states), enumerate(superblock_states_ket)
        ):
            statepair = (istate_bra, istate_ket)
            isDiag = istate_bra == istate_ket
            isLVC = (
                "onesite" in op_sys[statepair]
                and "onesite" in op_env[statepair]
                and not isDiag
            )
            coupleJ = matH_cas.coupleJ[istate_bra][istate_ket]
            if isDiag or coupleJ != 0.0 or isLVC:
                matC_bra, matC_ket = (
                    superblock_bra[psite],
                    superblock_ket[psite],
                )
                # nspf_list_bra = lattice_info_states[istate_bra].nspf_list_sites[
                #    psite
                # ]
                # nspf_list_ket = lattice_info_states[istate_ket].nspf_list_sites[
                #    psite
                # ]
                # kdof_bgn = sum(self.ndof_per_sites[:psite])
                # kdof_end = sum(self.ndof_per_sites[: psite + 1])
                op_l_ovlp: np.ndarray | jax.Array | int
                op_r_ovlp: np.ndarray | jax.Array | int

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
                assert isinstance(op_l_ovlp, np.ndarray | jax.Array)
                if np.allclose(op_l_ovlp, np.eye(*op_l_ovlp.shape)):
                    op_l_ovlp = op_l_ovlp.shape[0]
                assert isinstance(op_r_ovlp, np.ndarray | jax.Array)
                if np.allclose(op_r_ovlp, np.eye(*op_r_ovlp.shape)):
                    op_r_ovlp = op_r_ovlp.shape[0]

                mfop_site_ovlp = mfop_site(
                    matC_bra, matC_ket, op_l_ovlp, op_r_ovlp
                )
                if self.site_is_dof:
                    mfop_spf_ovlp_psite = [mfop_site_ovlp]
                else:
                    raise NotImplementedError
                    # ints_spf_ovlp = ints_spf["ovlp"][(istate_bra, istate_ket)][
                    #     kdof_bgn:kdof_end
                    # ]
                    # mfop_spf_ovlp_psite = _mfop_site2spf_prod(
                    #     mfop_site_ovlp,
                    #     ints_spf_ovlp,
                    #     nspf_list_bra,
                    #     nspf_list_ket,
                    # )
                mfop_spf_ovlp_psite_states[(istate_bra, istate_ket)] = (
                    mfop_spf_ovlp_psite
                )

                if matH_cas.onesite:  # isDiag:
                    op_l_onesite = (
                        op_sys[statepair]["onesite"]
                        if A_is_sys
                        else op_env[statepair]["onesite"]
                    )
                    op_r_onesite = (
                        op_env[statepair]["onesite"]
                        if A_is_sys
                        else op_sys[statepair]["onesite"]
                    )

                    mfop_site_onesite = mfop_site(
                        matC_bra, matC_ket, op_l_onesite, op_r_ovlp
                    )
                    mfop_site_onesite += mfop_site(
                        matC_bra, matC_ket, op_l_ovlp, op_r_onesite
                    )
                    if self.site_is_dof:
                        mfop_spf_onesite_psite = [mfop_site_onesite]
                    else:
                        raise NotImplementedError
                        # ints_spf_onesite = ints_spf["onesite"][
                        #     (istate_bra, istate_ket)
                        # ][kdof_bgn:kdof_end]
                        # mfop_spf_onesite_psite_prod = _mfop_site2spf_prod(
                        #     mfop_site_onesite,
                        #     ints_spf_ovlp,
                        #     nspf_list_bra,
                        #     nspf_list_ket,
                        # )
                        # mfop_spf_onesite_psite_summ = _mfop_site2spf_summ(
                        #     mfop_site_ovlp,
                        #     ints_spf_onesite,
                        #     nspf_list_bra,
                        #     nspf_list_ket,
                        # )
                        # mfop_spf_onesite_psite = [
                        #     a + b
                        #     for a, b in zip(
                        #         mfop_spf_onesite_psite_prod,
                        #         mfop_spf_onesite_psite_summ,
                        #         strict=True,
                        #     )
                        # ]

                    mfop_spf_onesite_psite_states[(istate_bra, istate_ket)] = (
                        mfop_spf_onesite_psite
                    )

                if isDiag and matH_cas.general:
                    assert self.site_is_dof

                    nterms_general = len(
                        matH_cas.general[istate_bra][istate_ket]
                    )
                    op_l_general_concat_dict = defaultdict(
                        lambda: np.zeros(
                            [nterms_general] + [*op_l_onesite.shape],
                            dtype=complex,
                        )
                    )
                    op_r_general_concat_dict = defaultdict(
                        lambda: np.zeros(
                            [nterms_general] + [*op_r_onesite.shape],
                            dtype=complex,
                        )
                    )

                    if "enable_summed_op" in const.keys:
                        general_superH_byterm = matH_cas.general_new[
                            "forward" if A_is_sys else "backward"
                        ][psite].superH[istate_bra][istate_ket]
                    else:
                        general_superH_byterm = matH_cas.general[istate_bra][
                            istate_ket
                        ]

                    nterm_dict: dict[str, int] = defaultdict(int)
                    nterm_dict_mpi: dict[str, int] = defaultdict(int)
                    for term_prod in general_superH_byterm:
                        blockop_key_l = term_prod.blockop_key_sites["left"][
                            psite
                        ]
                        blockop_key_r = term_prod.blockop_key_sites["right"][
                            psite
                        ]
                        site_op_key = term_prod.blockop_key_sites["centr"][
                            psite
                        ]

                        nterm_dict_mpi[site_op_key] += 1
                        if (
                            nterm_dict_mpi[site_op_key] % const.mpi_size
                            != const.mpi_rank
                        ):
                            continue

                        op_l_general = (
                            op_sys[statepair][blockop_key_l]
                            if A_is_sys
                            else op_env[statepair][blockop_key_l]
                        )
                        op_r_general = (
                            op_env[statepair][blockop_key_r]
                            if A_is_sys
                            else op_sys[statepair][blockop_key_r]
                        )

                        op_l_general_concat_dict[site_op_key][
                            nterm_dict[site_op_key], :, :
                        ] += op_l_general
                        op_r_general_concat_dict[site_op_key][
                            nterm_dict[site_op_key], :, :
                        ] += op_r_general * term_prod.coef
                        nterm_dict[site_op_key] += 1

                    mfop_spf_general_psite = defaultdict(
                        lambda: np.zeros(
                            (matC_bra.shape[1], matC_ket.shape[1]),
                            dtype=complex,
                        )
                    )

                    for site_op_key in nterm_dict_mpi.keys():
                        nterm = nterm_dict[site_op_key]

                        op_left_concat = op_l_general_concat_dict[site_op_key][
                            :nterm, :, :
                        ]
                        op_right_concat = op_r_general_concat_dict[site_op_key][
                            :nterm, :, :
                        ]
                        if const.verbose == 4:
                            helper.ElpTime.zgemm -= time()
                        mfop_spf_general_psite_val_send = mfop_site_concat(
                            matC_bra, matC_ket, op_left_concat, op_right_concat
                        )
                        if const.verbose == 4:
                            helper.ElpTime.zgemm += time()
                        assert mfop_spf_general_psite_val_send.shape == (
                            matC_bra.shape[1],
                            matC_ket.shape[1],
                        )

                        if const.mpi_size > 1:
                            const.mpi_comm.Allreduce(
                                mfop_spf_general_psite_val_send,
                                mfop_spf_general_psite[site_op_key],
                                const.mpi_sum,
                            )
                        else:
                            mfop_spf_general_psite[site_op_key] = (
                                mfop_spf_general_psite_val_send
                            )

                    if (
                        "enable_summed_op" in const.keys
                    ):  # or 'debug' in const.keys:
                        op_l_general_sum = (
                            op_sys[statepair]["summed"]
                            if A_is_sys
                            else op_env[statepair]["summed"]
                        )
                        op_r_general_sum = (
                            op_env[statepair]["summed"]
                            if A_is_sys
                            else op_sys[statepair]["summed"]
                        )
                        mfop_spf_general_psite["ovlp"] += mfop_site(
                            matC_bra, matC_ket, op_l_ovlp, op_r_general_sum
                        )
                        mfop_spf_general_psite["ovlp"] += mfop_site(
                            matC_bra, matC_ket, op_l_general_sum, op_r_ovlp
                        )

                    mfop_spf_general_psite_states[(istate_bra, istate_ket)] = (
                        mfop_spf_general_psite
                    )
                    if const.verbose == 4:
                        helper.ElpTime.mfop_gen_new += time()

        return {
            "rho": mfop_spf_ovlp_psite_states,
            "onesite": mfop_spf_onesite_psite_states,
            "general": mfop_spf_general_psite_states,
        }

    def construct_mfop(self, ints_spf: SPFInts, matH):
        mps_copy = copy.deepcopy(self)
        ints_site = mps_copy.get_ints_site(ints_spf)
        matH_cas = (
            matH
            if "enable_tdh_dofs" not in const.keys
            else self.get_matH_cas(matH, ints_spf)
        )
        matH_sweep = self.get_matH_sweep(matH_cas)
        if (left_is_C := self.is_psite_canonical(0)) or self.is_psite_canonical(
            self.nsite - 1
        ):
            mfop_cas, op_block_cas = mps_copy.construct_mfop_along_sweep(
                ints_site, matH_sweep, A_is_sys=left_is_C
            )
        else:
            raise AssertionError("MPS is not canonicalized in terminal sites")
        # active-space:BGN>
        if "enable_tdh_dofs" not in const.keys:
            mfop = mfop_cas
        else:
            """.. need to factrize the codes below """
            matH_tdh = mps_copy.get_matH_tdh(matH, op_block_cas)

            mfop = {"rho": {}, "onesite": {}, "general": {}}
            for istate_bra, istate_ket in itertools.product(
                range(matH.nstate), repeat=2
            ):
                statepair = (istate_bra, istate_ket)

                mfop_rho_state_cas = mfop_cas["rho"][statepair]
                mfop_onesite_state_cas = mfop_cas["onesite"][statepair]
                mfop_general_state_cas = mfop_cas["general"][statepair]
                ints_spf_state = ints_spf[statepair]

                """rho"""
                mfop_rho_state = [None] * matH.ndof
                for idof in range(matH.ndof):
                    if idof in self.dofs_cas:
                        idof_cas = self.dofs_cas.index(idof)
                        mfop_rho_state[idof] = mfop_rho_state_cas[idof_cas]
                    else:
                        mfop_rho_state[idof] = np.eye(1, dtype=complex)
                mfop["rho"][statepair] = mfop_rho_state

                """onesite"""
                mfop_onesite_state = [None] * matH.ndof
                for idof in range(matH.ndof):
                    if idof in self.dofs_cas:
                        idof_cas = self.dofs_cas.index(idof)
                        mfop_onesite_state[idof] = mfop_onesite_state_cas[
                            idof_cas
                        ]
                        mfop_onesite_state[idof] += mfop_rho_state_cas[
                            idof_cas
                        ] * np.prod(
                            [
                                ints_spf_state["onesite"][kdof]
                                for kdof in self.dofs_tdh
                            ]
                        )
                    else:
                        idof_tdh = self.dofs_tdh.index(idof)
                        mfop_onesite_state[idof] = np.prod(
                            [
                                ints_spf_state["onesite"][kdof]
                                for kdof in self.dofs_tdh
                            ]
                        )
                        mfop_onesite_state[idof] += op_block_cas[statepair][
                            "onesite"
                        ]
                mfop["onesite"][statepair] = mfop_onesite_state

                """general"""
                mfop_general_state = defaultdict(lambda: defaultdict(complex))
                "(1) idof in dofs_cas"
                for op_key, data_dofs in mfop_general_state_cas.items():
                    for idof_cas, data in data_dofs.items():
                        mfop_general_state[op_key][self.dofs_cas[idof_cas]] = (
                            data
                        )
                "(2) idof in dofs_tdh"
                for term_tdh in matH_tdh.general[istate_bra][istate_ket]:
                    for idof_tdh, op_key_idof in term_tdh.mode_ops.items():
                        dum = 1.0
                        for kdof_tdh, op_key_kdof in term_tdh.mode_ops.items():
                            if kdof_tdh != idof_tdh:
                                dum *= ints_spf_state[op_key_kdof][
                                    self.dofs_tdh[kdof_tdh]
                                ]
                        mfop_general_state[op_key_idof][
                            self.dofs_tdh[idof_tdh]
                        ] += term_tdh.coef * dum

                mfop["general"][statepair] = mfop_general_state
        # active-space:END>

        if const.verbose == 4:
            helper.ElpTime.mfop_3 -= time()
        mfop["rhoinv"] = {}
        for key_ij, mfop_ovlp_ij in mfop["rho"].items():
            if key_ij[0] == key_ij[1]:
                mfop["rhoinv"][key_ij] = [
                    helper.matrix_regularized_inverse(x, const.epsrho)
                    for x in mfop_ovlp_ij
                ]
        if const.verbose == 4:
            helper.ElpTime.mfop_3 += time()

        return mfop

    def construct_mfop_along_sweep(
        self,
        ints_site: dict[
            tuple[int, int], dict[str, list[np.ndarray] | list[jax.Array]]
        ],
        matH_cas,
        *,
        A_is_sys: bool,
    ):
        regularize_MPS = False  # True#

        if const.verbose == 4:
            helper.ElpTime.mfop_0 -= time()
        superblock_states = self.superblock_states
        nsite = self.nsite

        mfop_spf = {"rho": {}, "onesite": {}, "general": {}}

        psites_sweep_forward = (
            list(range(nsite)) if A_is_sys else list(range(nsite))[::-1]
        )
        psites_sweep_backward = (
            list(range(nsite)) if not A_is_sys else list(range(nsite))[::-1]
        )

        for psite in psites_sweep_forward[:-1]:
            superblock_trans_APsiB_psite(
                psite, superblock_states, A_is_sys, regularize=regularize_MPS
            )
        op_sys_sites = self.construct_op_sites(
            superblock_states, ints_site, A_is_sys, matH_cas
        )[::-1]

        for psite in psites_sweep_backward[:-1]:
            superblock_trans_APsiB_psite(
                psite,
                superblock_states,
                not A_is_sys,
                regularize=regularize_MPS,
            )
        op_env_sites = self.construct_op_sites(
            superblock_states, ints_site, not A_is_sys, matH_cas
        )

        if const.verbose == 4:
            helper.ElpTime.mfop_0 += time()
        for psite in psites_sweep_forward:
            if const.verbose == 4:
                helper.ElpTime.mfop_1 -= time()
            op_sys = op_sys_sites.pop()
            op_env = op_env_sites.pop()
            mfop_spf_psite_states = self.construct_mfop_MPS(
                psite,
                op_sys,
                op_env,
                matH_cas,
                A_is_sys,
            )
            if const.verbose == 4:
                helper.ElpTime.mfop_1 += time()
                helper.ElpTime.mfop_2 -= time()
            """swap dimension: mfop_spf(statepair)[mfop_type]_psite
                            => mfop_spf[mfop_type][statepair]_psite
            .. obviously, the codes below needs to be factrized"""
            for mfop_type in mfop_spf_psite_states.keys():
                if mfop_type == "general":
                    for statepair, mfop_general in mfop_spf_psite_states[
                        mfop_type
                    ].items():
                        if statepair not in mfop_spf["general"]:
                            mfop_spf[mfop_type][statepair] = defaultdict(dict)
                        for op_key, val_dofs in mfop_general.items():
                            mfop_spf[mfop_type][statepair][op_key][psite] = (
                                val_dofs
                            )
                else:
                    for statepair, data_site in mfop_spf_psite_states[
                        mfop_type
                    ].items():
                        if statepair not in mfop_spf[mfop_type]:
                            mfop_spf[mfop_type][statepair] = []

                        mfop_spf[mfop_type][statepair].extend(data_site)

            if psite != psites_sweep_forward[-1]:
                superblock_trans_APsiB_psite(
                    psite,
                    superblock_states,
                    A_is_sys,
                    regularize=regularize_MPS,
                )
            else:
                _svalues, op_block_cas = self.trans_next_psite_AsigmaB(
                    psite,
                    superblock_states,
                    op_sys,
                    ints_site,
                    matH_cas,
                    A_is_sys,
                    regularize=regularize_MPS,
                )
                # Not sure whethere have to be real> assert np.allclose(_svalues[0], np.eye(1,dtype=complex))
                if const.use_jax:
                    norm = abs(jnp.linalg.norm(_svalues))
                else:
                    norm = abs(np.linalg.norm(_svalues))
                if not np.allclose(norm, np.eye(1, dtype=complex)):
                    print(
                        f"Norm must be 1, but {norm} in mfop sweep at {psite} site"
                    )

            if const.verbose == 4:
                helper.ElpTime.mfop_2 += time()

        if len(mfop_spf["onesite"]) == 0:
            del mfop_spf["onesite"]
        if len(mfop_spf["general"]) == 0:
            del mfop_spf["general"]

        return (mfop_spf, op_block_cas)

    def construct_mfop_along_sweep_TEMP4DIPOLE(
        self,
        ints_site: dict[
            tuple[int, int], dict[str, list[np.ndarray] | list[jax.Array]]
        ],
        matO_cas,
        *,
        A_is_sys: bool,
        mps_coef_ket=None,
    ):
        if mps_coef_ket is None:
            mps_coef_ket = self
            raise NotImplementedError

        superblock_states_bra = self.superblock_states
        superblock_states_ket = mps_coef_ket.superblock_states

        mfop_spf = {"rho": {}, "onesite": {}, "general": {}}

        op_sys = self.construct_op_zerosite(superblock_states_bra, matO_cas)
        op_env_sites = self.construct_op_sites(
            superblock_states_bra,
            ints_site,
            not A_is_sys,
            matO_cas,
            superblock_states_ket,
        )

        psites_sweep = (
            list(range(self.nsite))
            if A_is_sys
            else list(range(self.nsite))[::-1]
        )
        for psite in psites_sweep:
            op_env = op_env_sites.pop()
            mfop_spf_psite_states = self.construct_mfop_MPS(
                psite,
                op_sys,
                op_env,
                matO_cas,
                A_is_sys,
                mps_coef_ket=mps_coef_ket,
            )
            for mfop_type in mfop_spf_psite_states.keys():
                if mfop_type == "general":
                    for statepair, mfop_general in mfop_spf_psite_states[
                        mfop_type
                    ].items():
                        if statepair not in mfop_spf["general"]:
                            mfop_spf[mfop_type][statepair] = defaultdict(dict)
                        for op_key, data_site in mfop_general.items():
                            mfop_spf[mfop_type][statepair][op_key][psite] = (
                                data_site
                            )
                else:
                    for statepair, data_site in mfop_spf_psite_states[
                        mfop_type
                    ].items():
                        if statepair not in mfop_spf[mfop_type]:
                            mfop_spf[mfop_type][statepair] = []
                        mfop_spf[mfop_type][statepair].extend(data_site)

            if psite != psites_sweep[-1]:
                op_sys = self.trans_next_psite_APsiB(
                    psite,
                    superblock_states_bra,
                    op_sys,  # type: ignore
                    ints_site,
                    matO_cas,
                    A_is_sys,
                    superblock_states_ket,
                )

        if len(mfop_spf["onesite"]) == 0:
            del mfop_spf["onesite"]
        if len(mfop_spf["general"]) == 0:
            del mfop_spf["general"]

        return mfop_spf

    def construct_op_zerosite(
        self,
        superblock_states: list[list[SiteCoef]],
        matH_cas: PolynomialHamiltonian | None = None,
    ) -> dict[tuple[int, int], dict[str, np.ndarray | jax.Array]]:
        """initialize op_block_psites

        Args:
            superblock_states (List[List[SiteCoef]]) : Super Blocks (Tensor Cores) of each electronic states
            matH_cas (Optional[PolynomialHamiltonian]) : Operator (such as Hamiltonian). Defaults to None.

        Returns:
            Dict[Tuple[int,int], Dict[str, np.ndarray | jax.Array]] : block operator. \
                key is 'auto', 'ovlp', 'onesite', 'summed' or (bra_state, ket_state)

        """
        autocorr_only = matH_cas is None

        op_block_states = {}
        for istate_bra, istate_ket in itertools.product(
            range(len(superblock_states)), repeat=2
        ):
            op_block_ops: dict[str, np.ndarray | jax.Array] = {}
            statepair = (istate_bra, istate_ket)
            isDiag = istate_bra == istate_ket
            coupleJ = (
                matH_cas.coupleJ[istate_bra][istate_ket] if matH_cas else 0.0
            )

            if autocorr_only:
                if isDiag:
                    if const.use_jax:
                        op_block_ops["auto"] = jnp.ones(
                            (1, 1), dtype=jnp.complex128
                        )
                    else:
                        op_block_ops["auto"] = np.ones((1, 1), dtype=complex)
            else:
                assert isinstance(matH_cas, PolynomialHamiltonian)
                if isDiag or coupleJ != 0.0 or matH_cas.onesite:
                    if const.use_jax:
                        op_block_ops["ovlp"] = jnp.ones(
                            (1, 1), dtype=jnp.complex128
                        )
                    else:
                        op_block_ops["ovlp"] = np.ones((1, 1), dtype=complex)

                if matH_cas.onesite:
                    if const.use_jax:
                        op_block_ops["onesite"] = jnp.zeros(
                            (1, 1), dtype=jnp.complex128
                        )
                    else:
                        op_block_ops["onesite"] = np.zeros(
                            (1, 1), dtype=complex
                        )

                if "enable_summed_op" in const.keys:
                    if const.use_jax:
                        op_block_ops["summed"] = jnp.zeros(
                            (1, 1), dtype=jnp.complex128
                        )
                    else:
                        op_block_ops["summed"] = np.zeros((1, 1), dtype=complex)

            if len(op_block_ops) != 0:
                op_block_states[statepair] = op_block_ops

        return op_block_states

    def renormalize_auto_psite(
        self,
        op_block_auto: int | np.ndarray | jax.Array,
        op_psite_auto: np.ndarray | jax.Array,
        matLorR_bra: SiteCoef,
        matLorR_ket: SiteCoef,
    ) -> np.ndarray | jax.Array:
        return contract_with_site(
            matLorR_bra.conj(),
            matLorR_ket,
            op_block_auto,
            op_psite_auto,
        )

    # @profile
    def renormalize_op_psite(
        self,
        psite: int,
        superblock_states: list[list[SiteCoef]],
        op_block_states: dict[
            tuple[int, int],
            dict[str, _block_type],
        ],
        ints_site: dict[
            tuple[int, int], dict[str, list[np.ndarray] | list[jax.Array]]
        ],
        matH_cas: PolynomialHamiltonian,
        A_is_sys: bool,
        superblock_states_unperturb=None,
    ) -> dict[tuple[int, int], dict[str, np.ndarray | jax.Array]]:
        superblock_states_bra = superblock_states
        superblock_states_ket = (
            superblock_states
            if superblock_states_unperturb is None
            else superblock_states_unperturb
        )

        op_block_next = {}
        for (istate_bra, superblock_bra), (
            istate_ket,
            superblock_ket,
        ) in itertools.product(
            enumerate(superblock_states_bra), enumerate(superblock_states_ket)
        ):
            op_block_next_ops = {}
            statepair = (istate_bra, istate_ket)
            isDiag = istate_bra == istate_ket

            matLorR_bra = superblock_bra[psite]
            matLorR_ket = superblock_ket[psite]

            if statepair not in op_block_states:
                continue

            op_block_statepair = op_block_states[statepair]
            ints_site_statepair = ints_site[statepair]
            if "auto" in op_block_statepair:
                if isDiag:
                    op_block_next_ops["auto"] = self.renormalize_auto_psite(
                        op_block_auto=op_block_statepair["auto"],
                        op_psite_auto=ints_site_statepair["auto"][psite],
                        matLorR_bra=matLorR_bra,
                        matLorR_ket=matLorR_ket,
                    )

            elif "ovlp" in op_block_statepair:
                op_block_ovlp: _block_type = op_block_statepair["ovlp"]
                op_psite_ovlp: _block_type = ints_site_statepair["ovlp"][psite]
                op_block_next_ops["ovlp"] = contract_with_site(
                    matLorR_bra, matLorR_ket, op_block_ovlp, op_psite_ovlp
                )
                assert isinstance(op_block_ovlp, np.ndarray | jax.Array)
                if np.allclose(op_block_ovlp, np.eye(*op_block_ovlp.shape)):
                    op_block_ovlp = op_block_ovlp.shape[0]
                assert isinstance(op_psite_ovlp, np.ndarray | jax.Array)
                if np.allclose(op_psite_ovlp, np.eye(*op_psite_ovlp.shape)):
                    op_psite_ovlp = op_psite_ovlp.shape[0]

                if matH_cas and matH_cas.onesite:
                    op_block_onesite = op_block_statepair["onesite"]
                    op_psite_onesite = ints_site_statepair["onesite"][psite]
                    op_block_next_ops["onesite"] = contract_with_site(
                        matLorR_bra,
                        matLorR_ket,
                        op_block_onesite,
                        op_psite_ovlp,
                    ) + contract_with_site(
                        matLorR_bra,
                        matLorR_ket,
                        op_block_ovlp,
                        op_psite_onesite,
                    )

                if isDiag and matH_cas and matH_cas.general:
                    # When we implement 2nd order vibronic coupling, this part should be modified.
                    key_sys = "left" if A_is_sys else "right"
                    psite_next = psite + 1 if A_is_sys else psite - 1

                    nterms_general = len(
                        matH_cas.general[istate_bra][istate_ket]
                    )
                    op_block_general_concat: np.ndarray | jax.Array
                    op_psite_general_concat: np.ndarray | jax.Array
                    assert isinstance(op_block_onesite, np.ndarray | jax.Array)
                    op_block_general_concat = np.zeros(
                        [nterms_general] + [*op_block_onesite.shape],
                        dtype=complex,
                    )
                    op_psite_general_concat = np.zeros(
                        [nterms_general] + [*op_psite_onesite.shape],
                        dtype=complex,
                    )
                    index_dict = {}

                    if "enable_summed_op" in const.keys:
                        general_renorm_byterm = matH_cas.general_new[
                            "forward" if A_is_sys else "backward"
                        ][psite].renorm[istate_bra][istate_ket]
                    else:
                        general_renorm_byterm = matH_cas.general[istate_bra][
                            istate_ket
                        ]

                    nterm_concat = 0
                    for term_prod in general_renorm_byterm:
                        try:
                            blockop_key_next = term_prod.blockop_key_sites[
                                key_sys
                            ][psite_next]
                        except IndexError:  # when psite is an edge, the psite_next is out of range(nsite)
                            blockop_key_next = term_prod.term_key()
                        if blockop_key_next not in index_dict:
                            blockop_key = term_prod.blockop_key_sites[key_sys][
                                psite
                            ]
                            site_op_key = term_prod.blockop_key_sites["centr"][
                                psite
                            ]
                            op_block_general_concat[nterm_concat, :, :] = (
                                op_block_statepair[blockop_key]
                            )
                            op_psite_general_concat[nterm_concat, :, :] = (
                                ints_site_statepair[site_op_key][psite]
                            )
                            index_dict[blockop_key_next] = nterm_concat
                            nterm_concat += 1

                    if const.use_jax:
                        op_block_general_concat = jnp.array(
                            op_block_general_concat[:nterm_concat, :, :]
                        )
                        op_psite_general_concat = jnp.array(
                            op_psite_general_concat[:nterm_concat, :, :]
                        )
                    else:
                        op_block_general_concat = op_block_general_concat[
                            :nterm_concat, :, :
                        ]
                        op_psite_general_concat = op_psite_general_concat[
                            :nterm_concat, :, :
                        ]

                    op_block_general_next_concat = contract_with_site_concat(
                        matLorR_bra,
                        matLorR_ket,
                        op_block_general_concat,
                        op_psite_general_concat,
                    )

                    for blockop_key_next, i in index_dict.items():
                        op_block_next_ops[blockop_key_next] = (
                            op_block_general_next_concat[i, :, :]
                        )

                    if "enable_summed_op" in const.keys:
                        op_block_next_ops_summed_send: np.ndarray | jax.Array
                        if const.use_jax:
                            assert isinstance(
                                op_block_statepair["ovlp"], jax.Array
                            )
                            op_block_next_ops_summed_send = jnp.zeros(
                                op_block_statepair["ovlp"].shape,
                                dtype=jnp.complex128,
                            )
                        else:
                            assert isinstance(
                                op_block_next_ops["ovlp"], np.ndarray
                            )
                            op_block_next_ops_summed_send = np.zeros(
                                op_block_next_ops["ovlp"].shape, dtype=complex
                            )
                        for iwork, term_prod in enumerate(
                            matH_cas.general_new[
                                "forward" if A_is_sys else "backward"
                            ][psite].renorm_summed[istate_bra][istate_ket]
                        ):
                            if iwork % const.mpi_size != const.mpi_rank:
                                continue
                            blockop_key = term_prod.blockop_key_sites[key_sys][
                                psite
                            ]
                            site_op_key = term_prod.blockop_key_sites["centr"][
                                psite
                            ]
                            if site_op_key != "ovlp":
                                """<sys_next|summed|sys_next> += <sys|Os(key)|sys> <dot|Od(!=ovlp)|dot>"""
                                op_block_sys = op_block_statepair[blockop_key]
                                op_psite = ints_site_statepair[site_op_key][
                                    psite
                                ]
                                op_block_next_ops_summed_send += (
                                    term_prod.coef
                                    * contract_with_site(
                                        matLorR_bra,
                                        matLorR_ket,
                                        op_block_sys,
                                        op_psite,
                                    )
                                )

                        op_block_next_ops["summed"] = (
                            op_block_next_ops_summed_send
                        )

                        """<sys_next|summed|sys_next> += <sys|'summed'|sys> <dot|Od(==ovlp)|dot>"""
                        op_block_sys = op_block_statepair["summed"]
                        op_psite = ints_site_statepair["ovlp"][psite]
                        op_block_next_ops["summed"] += contract_with_site(
                            matLorR_bra, matLorR_ket, op_block_sys, op_psite
                        )
            else:
                raise ValueError(
                    f"Either 'auto' or 'ovlp' should be in {op_block_statepair}"
                )

            if len(op_block_next_ops) != 0:
                op_block_next[statepair] = op_block_next_ops

        return op_block_next

    def operators_for_superH(
        self,
        psite: int,
        op_sys: dict[tuple[int, int], dict[str, np.ndarray | jax.Array]],
        op_env: dict[tuple[int, int], dict[str, np.ndarray | jax.Array]],
        ints_site: dict[
            tuple[int, int], dict[str, list[np.ndarray] | list[jax.Array]]
        ],
        matH_cas: PolynomialHamiltonian,
        A_is_sys: bool,
    ) -> list[
        list[
            dict[
                str,
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
            psite (int): site index on "C"
            op_sys (Dict[Tuple[int,int], Dict[str, np.ndarray | jax.Array]]): System operator
            op_env (Dict[Tuple[int,int], Dict[str, np.ndarray | jax.Array]]): Environment operator
            ints_site (Dict[Tuple[int,int],Dict[str, np.ndarray]]): Site integral
            matH_cas (PolynomialHamiltonian) : Hamiltonian
            A_is_sys (bool): Whether left block is System

        Returns:
            List[List[Dict[str, Tuple[_block_type, _block_type, _block_type]]]]: \
                [i-bra-state][j-ket-state]['q'] =  (op_l, op_c, op_r)
        """
        nstate = len(matH_cas.coupleJ)
        op_lcr: list[
            list[
                dict[
                    str,
                    tuple[
                        _block_type,
                        _block_type,
                        _block_type,
                    ],
                ]
            ]
        ]
        op_lcr = [[None for j in range(nstate)] for i in range(nstate)]  # type: ignore
        op_l_ovlp: _block_type
        op_r_ovlp: _block_type
        op_c_ovlp: _block_type
        for istate_bra, istate_ket in itertools.product(
            list(range(nstate)), repeat=2
        ):
            coupleJ = matH_cas.coupleJ[istate_bra][istate_ket]
            statepair = (istate_bra, istate_ket)
            isDiag = istate_bra == istate_ket
            isLVC = (
                "onesite" in op_sys[statepair]
                and "onesite" in op_env[statepair]
                and not isDiag
            )

            if not isDiag and coupleJ == 0.0 and not isLVC:
                # If diabatic coupling <α|H|β> = 0.0 or LVC <α|(dH/dQ)|β><i|Q|j> = 0.0,
                # off-diagonal elements of Hamiltonian are no need to be calculated.
                continue

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
            op_c_ovlp = ints_site[statepair]["ovlp"][psite]

            # type(op) is 'int' if it is a unit-matrix --> already applied bra != ket spfs.
            assert isinstance(op_l_ovlp, np.ndarray | jax.Array)
            if np.allclose(op_l_ovlp, np.eye(*op_l_ovlp.shape)):
                op_l_ovlp = op_l_ovlp.shape[0]
            assert isinstance(op_c_ovlp, np.ndarray | jax.Array)
            if np.allclose(op_c_ovlp, np.eye(*op_c_ovlp.shape)):
                op_c_ovlp = op_c_ovlp.shape[0]
            assert isinstance(op_r_ovlp, np.ndarray | jax.Array)
            if np.allclose(op_r_ovlp, np.eye(*op_r_ovlp.shape)):
                op_r_ovlp = op_r_ovlp.shape[0]

            op_lcr[istate_bra][istate_ket] = {
                "ovlp": (op_l_ovlp, op_c_ovlp, op_r_ovlp)
            }

            if matH_cas.onesite:
                op_l_onesite = (
                    op_sys[statepair]["onesite"]
                    if A_is_sys
                    else op_env[statepair]["onesite"]
                )
                op_r_onesite = (
                    op_env[statepair]["onesite"]
                    if A_is_sys
                    else op_sys[statepair]["onesite"]
                )
                op_c_onesite = ints_site[statepair]["onesite"][psite]

                op_lcr[istate_bra][istate_ket]["onesite"] = (
                    op_l_onesite,
                    op_c_onesite,
                    op_r_onesite,
                )

            if isDiag and matH_cas.general:
                # When we implement 2nd order coupling term <α|d^2H/dQ_pdQ_q|β><i|Q_p|j><k|Q_l|l>,
                # this part should be modified.
                nterms_general = len(matH_cas.general[istate_bra][istate_ket])
                op_l_general_concat = np.zeros(
                    [nterms_general] + [*op_l_onesite.shape], dtype=complex
                )
                op_r_general_concat = np.zeros(
                    [nterms_general] + [*op_r_onesite.shape], dtype=complex
                )
                op_c_general_concat = np.zeros(
                    [nterms_general] + [*op_c_onesite.shape], dtype=complex
                )
                nterm_concat = 0

                if "enable_summed_op" in const.keys:
                    general_superH_byterm = matH_cas.general_new[
                        "forward" if A_is_sys else "backward"
                    ][psite].superH[istate_bra][istate_ket]
                else:
                    general_superH_byterm = matH_cas.general[istate_bra][
                        istate_ket
                    ]

                for iwork, term_prod in enumerate(general_superH_byterm):
                    if iwork % const.mpi_size != const.mpi_rank:
                        continue
                    blockop_key_l = term_prod.blockop_key_sites["left"][psite]
                    blockop_key_r = term_prod.blockop_key_sites["right"][psite]
                    site_op_key = term_prod.blockop_key_sites["centr"][psite]

                    op_l_general = (
                        op_sys[statepair][blockop_key_l]
                        if A_is_sys
                        else op_env[statepair][blockop_key_l]
                    )
                    op_r_general = (
                        op_env[statepair][blockop_key_r]
                        if A_is_sys
                        else op_sys[statepair][blockop_key_r]
                    )
                    op_c_general = ints_site[statepair][site_op_key][psite]

                    op_l_general_concat[nterm_concat, :, :] = op_l_general
                    op_r_general_concat[nterm_concat, :, :] = op_r_general
                    op_c_general_concat[nterm_concat, :, :] = (
                        op_c_general * term_prod.coef
                    )
                    nterm_concat += 1

                op_l_general_concat = op_l_general_concat[:nterm_concat, :, :]
                op_r_general_concat = op_r_general_concat[:nterm_concat, :, :]
                op_c_general_concat = op_c_general_concat[:nterm_concat, :, :]
                if const.use_jax:
                    op_lcr[istate_bra][istate_ket]["general_concat"] = (
                        jnp.array(op_l_general_concat, dtype=jnp.complex128),
                        jnp.array(op_c_general_concat, dtype=jnp.complex128),
                        jnp.array(op_r_general_concat, dtype=jnp.complex128),
                    )
                else:
                    op_lcr[istate_bra][istate_ket]["general_concat"] = (
                        op_l_general_concat,
                        op_c_general_concat,
                        op_r_general_concat,
                    )

                if "enable_summed_op" in const.keys:
                    op_l_general_sum = (
                        op_sys[statepair]["summed"]
                        if A_is_sys
                        else op_env[statepair]["summed"]
                    )
                    op_r_general_sum = (
                        op_env[statepair]["summed"]
                        if A_is_sys
                        else op_sys[statepair]["summed"]
                    )
                    op_lcr[istate_bra][istate_ket]["general_summ_l"] = (
                        op_l_general_sum,
                        op_c_ovlp,
                        op_r_ovlp,
                    )
                    op_lcr[istate_bra][istate_ket]["general_summ_r"] = (
                        op_l_ovlp,
                        op_c_ovlp,
                        op_r_general_sum,
                    )
        return op_lcr

    def operators_for_superK(
        self,
        psite: int,
        op_sys: dict[tuple[int, int], dict[str, np.ndarray | jax.Array]],
        op_env: dict[tuple[int, int], dict[str, np.ndarray | jax.Array]],
        matH_cas: PolynomialHamiltonian,
        A_is_sys: bool,
    ) -> list[
        list[
            dict[
                str,
                tuple[
                    _block_type,
                    _block_type,
                ],
            ]
        ]
    ]:
        """ LsR operator

        construct full-matrix Kamiltonian

        Args:
            psite (int): site index on "Psi"
            op_sys (Dict[Tuple[int,int], Dict[str, np.ndarray | jax.Array]]): System operator
            op_env (Dict[Tuple[int,int], Dict[str, np.ndarray | jax.Array]]): Environment operator
            matH_cas (PolynomialHamiltonian) : Hamiltonian
            A_is_sys (bool): Whether left block is System

        Returns:
            List[List[Dict[str, Tuple[_block_type, _block_type]]]]: \
                [i-bra-state][j-ket-state]['q'] =  (op_l, op_r)
        """
        nstate = len(matH_cas.coupleJ)
        op_lr: list[
            list[
                dict[
                    str,
                    tuple[
                        _block_type,
                        _block_type,
                    ],
                ]
            ]
        ]
        op_l_ovlp: _block_type
        op_r_ovlp: _block_type
        op_lr = [[None for j in range(nstate)] for i in range(nstate)]  # type: ignore
        for istate_bra, istate_ket in itertools.product(
            list(range(nstate)), repeat=2
        ):
            coupleJ = matH_cas.coupleJ[istate_bra][istate_ket]
            statepair = (istate_bra, istate_ket)
            isDiag = istate_bra == istate_ket
            isLVC = (
                "onesite" in op_sys[statepair]
                and "onesite" in op_env[statepair]
                and not isDiag
            )

            if not isDiag and coupleJ == 0 and not isLVC:
                # If diabatic coupling <α|H|β> = 0.0 or LVC <α|(dH/dQ)|β><i|Q|j> = 0.0,
                # off-diagonal elements of Hamiltonian are no need to be calculated.
                continue

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
            # type(op) is 'int' if it is a unit-matrix
            if np.allclose(op_l_ovlp, np.eye(*op_l_ovlp.shape)):
                op_l_ovlp = op_l_ovlp.shape[0]
            if np.allclose(op_r_ovlp, np.eye(*op_r_ovlp.shape)):
                op_r_ovlp = op_r_ovlp.shape[0]

            op_lr[istate_bra][istate_ket] = {"ovlp": (op_l_ovlp, op_r_ovlp)}

            if matH_cas.onesite:
                op_l_onesite = (
                    op_sys[statepair]["onesite"]
                    if A_is_sys
                    else op_env[statepair]["onesite"]
                )
                op_r_onesite = (
                    op_env[statepair]["onesite"]
                    if A_is_sys
                    else op_sys[statepair]["onesite"]
                )
                op_lr[istate_bra][istate_ket]["onesite"] = (
                    op_l_onesite,
                    op_r_onesite,
                )

            if isDiag and matH_cas.general:
                # When we implement 2nd order coupling term <α|d^2H/dQ_pdQ_q|β><i|Q_p|j><k|Q_l|l>,
                # this part should be modified.
                key_sys = "left" if A_is_sys else "right"
                key_env = "right" if A_is_sys else "left"
                psite_next = psite + 1 if A_is_sys else psite - 1

                nterms_general = len(matH_cas.general[istate_bra][istate_ket])
                op_l_general_concat = np.zeros(
                    [nterms_general] + list(op_l_onesite.shape), dtype=complex
                )
                op_r_general_concat = np.zeros(
                    [nterms_general] + list(op_r_onesite.shape), dtype=complex
                )
                nterm_concat = 0

                if "enable_summed_op" in const.keys:
                    general_superK_byterm = matH_cas.general_new[
                        "forward" if A_is_sys else "backward"
                    ][psite].superK[istate_bra][istate_ket]
                else:
                    general_superK_byterm = matH_cas.general[istate_bra][
                        istate_ket
                    ]

                for iwork, term_prod in enumerate(general_superK_byterm):
                    if iwork % const.mpi_size != const.mpi_rank:
                        continue

                    blockop_key_sys = term_prod.blockop_key_sites[key_sys][
                        psite_next
                    ]
                    blockop_key_env = term_prod.blockop_key_sites[key_env][
                        psite
                    ]

                    op_l_general = (
                        op_sys[statepair][blockop_key_sys]
                        if A_is_sys
                        else op_env[statepair][blockop_key_env]
                    )
                    op_r_general = (
                        op_env[statepair][blockop_key_env]
                        if A_is_sys
                        else op_sys[statepair][blockop_key_sys]
                    )

                    op_l_general_concat[nterm_concat, :, :] = op_l_general
                    op_r_general_concat[nterm_concat, :, :] = (
                        op_r_general * term_prod.coef
                    )
                    nterm_concat += 1

                op_l_general_concat = op_l_general_concat[:nterm_concat, :, :]
                op_r_general_concat = op_r_general_concat[:nterm_concat, :, :]
                if const.use_jax:
                    op_lr[istate_bra][istate_ket]["general_concat"] = (
                        jnp.array(op_l_general_concat, dtype=jnp.complex128),
                        jnp.array(op_r_general_concat, dtype=jnp.complex128),
                    )
                else:
                    op_lr[istate_bra][istate_ket]["general_concat"] = (
                        op_l_general_concat,
                        op_r_general_concat,
                    )

                if "enable_summed_op" in const.keys:
                    op_l_general_sum = (
                        op_sys[statepair]["summed"]
                        if A_is_sys
                        else op_env[statepair]["summed"]
                    )
                    op_r_general_sum = (
                        op_env[statepair]["summed"]
                        if A_is_sys
                        else op_sys[statepair]["summed"]
                    )
                    op_lr[istate_bra][istate_ket]["general_summ_l"] = (
                        op_l_general_sum,
                        op_r_ovlp,
                    )
                    op_lr[istate_bra][istate_ket]["general_summ_r"] = (
                        op_l_ovlp,
                        op_r_general_sum,
                    )
        return op_lr
