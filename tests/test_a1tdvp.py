import sys
from typing import Literal

import jax
import numpy as np
import pytest
from discvar import HarmonicOscillator as HO
from loguru import logger

import pytdscf
import pytdscf._const_cls
from pytdscf._contraction import (
    multiplyH_MPS_direct,
    multiplyH_MPS_direct_MPO,
    multiplyK_MPS_direct,
    multiplyK_MPS_direct_MPO,
)
from pytdscf._site_cls import SiteCoef
from pytdscf.basis import Exciton
from pytdscf.dvr_operator_cls import TensorOperator
from pytdscf.hamiltonian_cls import TensorHamiltonian
from pytdscf.model_cls import BasInfo, Model
from pytdscf.units import au_in_cm1

logger.remove()
logger.add(sys.stdout, level="DEBUG")


freqs_cm1 = [1000, 2000, 3000]
omega2 = [(freq / au_in_cm1) ** 2 for freq in freqs_cm1]
nspf = nprim = 8
prim_info = [HO(nprim, freq, units="cm-1") for freq in freqs_cm1] + [
    Exciton(nstate=2, names=["S0", "S1"])
]
backend = "numpy"


def get_model():
    """
    |Psi> = |HO1, HO2, HO3, E0>

    H =
    (Σ_i ω_i / 2 * (P_i^2 + Q_i^2) |i><i|) |S0><S0|
    + (ΔE + Σ_i ω_i / 2 * (P_i^2 + Q_i^2) + κ_i Q_i|i><i|) |S1><S1|
    + (J + Σ_i λ_i Q_i |i><i|) |S0><S1|
    + (J + Σ_i λ_i Q_i |i><i|) |S1><S0|

    """
    # dE = 0.007
    # J = -0.004
    # lamb = 0.0002
    # kappa = 0.0001

    dE = 0.01  # 0.27 eV
    J = 0.001  # 0.027 eV
    lamb = 0.0001
    kappa = 0.0001

    basinfo = BasInfo([prim_info])
    potential_mpo = []
    """
    Symbolic MPO constructed by PyMPO

    W0 = [[1, q0, ω_i^2/q0^2]]
    W1 = [[J+λq1, 1, κq1 + ω_i^2/2q1^2, ω_i^2/2q1^2],
          [    λ, 0,                 κ,           0],
          [    0, 0,                 1,           1]]
    W2 = [[                 0,           0,   1],
          [ΔE+κq2+ω_i^2/2q2^2, ω_i^2/2q2^2, λq2],
          [                 1,           0,   0],
          [                 0,           1,   0]]
    W3 = [[ a3†a3],
          [ a3a3†],
          [a3†+a3]]
    """
    W0 = np.zeros((1, nprim, 3), dtype=np.complex128)
    W1 = np.zeros((3, nprim, 4), dtype=np.complex128)
    W2 = np.zeros((4, nprim, 3), dtype=np.complex128)
    W3 = np.zeros((3, 2, 2, 1), dtype=np.complex128)

    q1_list = [np.array(ho.get_grids()) for ho in prim_info[:3]]
    q2_list = [q1 * q1 for q1 in q1_list]
    eye_list = [np.ones_like(q1) for q1 in q1_list]
    a = prim_info[3].get_annihilation_matrix()
    a_dag = prim_info[3].get_creation_matrix()

    W0[0, :, 0] = eye_list[0]
    W0[0, :, 1] = q1_list[0]
    W0[0, :, 2] = omega2[0] / 2 * q2_list[0]

    W1[0, :, 0] = J * eye_list[1] + lamb * q1_list[1]
    W1[0, :, 1] = eye_list[1]
    W1[0, :, 2] = kappa * q1_list[1] + omega2[1] ** 2 / 2 * q2_list[1]
    W1[0, :, 3] = omega2[1] / 2 * q2_list[1]
    W1[1, :, 0] = lamb * eye_list[1]
    W1[1, :, 2] = kappa * eye_list[1]
    W1[2, :, 2] = eye_list[1]
    W1[2, :, 3] = eye_list[1]

    W2[0, :, 2] = eye_list[2]
    W2[1, :, 0] = (
        dE * eye_list[2] + kappa * q1_list[2] + omega2[2] / 2 * q2_list[2]
    )
    W2[1, :, 1] = omega2[2] / 2 * q2_list[2]
    W2[1, :, 2] = lamb * q1_list[2]
    W2[2, :, 0] = eye_list[2]
    W2[3, :, 1] = eye_list[2]

    W3[0, :, :, 0] = a_dag @ a
    W3[1, :, :, 0] = a @ a_dag
    W3[2, :, :, 0] = a_dag + a

    potential_mpo = [W0, W1, W2, W3]

    diag_indices = np.arange(nprim)
    W0_diag = np.zeros((1, nprim, nprim, 3), dtype=np.complex128)
    W0_diag[:, diag_indices, diag_indices, :] = W0
    W1_diag = np.zeros((3, nprim, nprim, 4), dtype=np.complex128)
    W1_diag[:, diag_indices, diag_indices, :] = W1
    W2_diag = np.zeros((4, nprim, nprim, 3), dtype=np.complex128)
    W2_diag[:, diag_indices, diag_indices, :] = W2

    V = np.einsum(
        "abcd,defg,ghij,klmn->behlcfim", W0_diag, W1_diag, W2_diag, W3
    )
    V = V.reshape(nprim**3 * 2, nprim**3 * 2)
    np.testing.assert_allclose(V, V.conj().T)
    potential = [
        [
            {
                (0, 1, 2, (3, 3)): TensorOperator(
                    mpo=potential_mpo, legs=(0, 1, 2, 3, 3)
                )
            }
        ]
    ]
    kinetic_mpo = []
    """
    T = [0.5dq2 1] [[1 0.5dq2]  ... [[1     ]
                    [0      1]]      [0.5dq2]]

    """
    for idof in range(3):
        if idof == 0:
            core = np.zeros((1, nprim, nprim, 2), dtype=np.complex128)
            core[0, :, :, 0] = (
                prim_info[idof].get_2nd_derivative_matrix_dvr() / 2
            )
            core[0, :, :, 1] = np.eye(nprim)
        elif idof == 2:
            core = np.zeros((2, nprim, nprim, 1), dtype=np.complex128)
            core[0, :, :, 0] = np.eye(nprim)
            core[1, :, :, 0] = (
                prim_info[idof].get_2nd_derivative_matrix_dvr() / 2
            )
        else:
            core = np.zeros((2, nprim, nprim, 2), dtype=np.complex128)
            core[0, :, :, 0] = np.eye(nprim)
            core[1, :, :, 1] = np.eye(nprim)
            core[0, :, :, 1] = (
                prim_info[idof].get_2nd_derivative_matrix_dvr() / 2
            )
        kinetic_mpo.append(core)
    kinetic = [
        [
            {
                ((0, 0), (1, 1), (2, 2)): TensorOperator(
                    mpo=kinetic_mpo, legs=tuple([0, 0, 1, 1, 2, 2])
                )
            }
        ]
    ]
    hamiltonian = TensorHamiltonian(
        ndof=4, potential=potential, kinetic=kinetic, backend=backend
    )

    operators = {"hamiltonian": hamiltonian}

    model = Model(basinfo, operators)
    model.m_aux_max = 1
    model.init_HartreeProduct = [
        [ho.get_unitary()[0].tolist() for ho in prim_info[:3]]
        + [np.array([0.0, 1.0]).tolist()]
    ]
    return model


def get_superblock_full(superblock, delta_rank: int):
    superblock_full = []
    nsite = len(superblock)
    for i, core in enumerate(superblock):
        additional_rank = 0
        if core.gauge == "A":
            if i == nsite - 1:
                pass
            else:
                l1, c1, r1 = core.data.shape
                l2, c2, r2 = superblock[i + 1].data.shape
                assert l2 == r1
                additional_rank = max(
                    min(delta_rank, min(l1 * c1 - r1, c2 * r2 - l2)), 0
                )
        elif core.gauge == "B":
            if i == 0:
                pass
            else:
                l1, c1, r1 = core.data.shape
                l2, c2, r2 = superblock[i - 1].data.shape
                assert l1 == r2
                additional_rank = max(
                    min(delta_rank, min(c1 * r1 - l1, l2 * c2 - r2)), 0
                )
        if additional_rank == 0:
            superblock_full.append(core)
        else:
            superblock_full.append(
                core.thin_to_full(additional_rank=additional_rank)
            )

    return superblock_full


def get_op_block_full(
    mps,
    psite: int,
    superblock_states: list[list[SiteCoef]],
    superblock_states_full: list[list[SiteCoef]],
    op_block_previous,
    hamiltonian: TensorHamiltonian,
    construct_left: bool,
    mode: Literal["bra", "braket"],
):
    if len(superblock_states) != 1:
        raise NotImplementedError("only support single superblock")

    if construct_left:
        step = -1
    else:
        step = +1

    match mode:
        case "bra":
            superblock_states_bra = superblock_states_full
            superblock_states_ket = None
        case "braket":
            superblock_states = superblock_states_full
            superblock_states_bra = None
            superblock_states_ket = None

    block_full = mps.renormalize_op_psite(
        psite=psite + step,
        superblock_states=superblock_states,
        op_block_states=op_block_previous,
        ints_site=None,
        hamiltonian=hamiltonian,
        A_is_sys=construct_left,
        superblock_states_ket=superblock_states_ket,
        superblock_states_bra=superblock_states_bra,
    )
    return block_full


def test_a1tdvp_sweep():
    model = get_model()
    pytdscf._const_cls.const.set_runtype(
        use_jax=False, jobname="a1tdvp", adaptive=True
    )
    mps = pytdscf._mps_mpo.MPSCoefMPO.alloc_random(model)
    assert isinstance(mps, pytdscf._mps_mpo.MPSCoefMPO)
    superblock = mps.superblock_states[0]
    nsite = len(superblock)
    begin_site = 0
    end_site = nsite - 1
    step = 1
    A_is_sys = True
    dD = 3
    dt = 0.25
    # logger.debug(f"{superblock=}")
    pytdscf._mps_cls.canonicalize(superblock, orthogonal_center=0)
    superblock_full = get_superblock_full(superblock, delta_rank=dD)
    op_sys = mps.construct_op_zerosite([superblock], model.hamiltonian)
    mps.op_sys_sites = [op_sys]
    op_env_sites = mps.construct_op_sites(
        [superblock],
        None,
        begin_site=end_site,
        end_site=begin_site,
        matH_cas=model.hamiltonian,
        superblock_states_ket=None,
        superblock_states_bra=None,
    )
    # logger.debug(op_sys)
    # logger.debug(op_env_sites)
    psites_sweep = range(begin_site, end_site + step, step)
    for psite in psites_sweep:
        logger.debug(f"{psite=}")
        op_env_thin = op_env_sites.pop()
        if psite != end_site:
            assert len(op_env_sites) > 0
            op_env_previous = op_env_sites[-1]
            # logger.debug(f"{op_env_thin=}")
            # logger.debug(f"{op_env_previous=}")
            op_env_full_braket = get_op_block_full(
                mps=mps,
                psite=psite,
                superblock_states=[superblock],
                superblock_states_full=[superblock_full],
                op_block_previous=op_env_previous,
                hamiltonian=model.hamiltonian,
                construct_left=not A_is_sys,
                mode="braket",
            )
            # logger.debug(f"{op_env_full_braket=}")
            op_env_full_bra = get_op_block_full(
                mps=mps,
                psite=psite,
                superblock_states=[superblock],
                superblock_states_full=[superblock_full],
                op_block_previous=op_env_previous,
                hamiltonian=model.hamiltonian,
                construct_left=not A_is_sys,
                mode="bra",
            )
            op_sys_thin = mps.op_sys_sites[-1]
            psi_left, sigvec, psi_right, op_sys_full_bra = (
                get_psi_sigvec_psi_fullblock(
                    mps=mps,
                    psite=psite,
                    to="->",
                    op_block=op_sys_thin,
                    additional_rank=10,
                    hamiltonian=model.hamiltonian,
                )
            )
            psi_states_left = [psi_left]
            sigvec_states = [sigvec]
            psi_states_right = [psi_right]
            newD, error = calculate_projection_error(
                mps=mps,
                psite=psite,
                Dmax=100,
                p=1.0e-02,
                op_sys_full=op_sys_full_bra,
                op_sys_thin=op_sys_thin,
                op_env_full=op_env_full_bra,
                op_env_thin=op_env_previous,
                hamiltonian=model.hamiltonian,
                psi_states_left=psi_states_left,
                sigvec_states=sigvec_states,
                psi_states_right=psi_states_right,
                to="->",
            )
            logger.debug(f"{newD=}, {error=:3e}")
            op_env_D_bra = truncate_op_block(op_env_full_bra, newD, mode="bra")
            op_env_D_braket = truncate_op_block(
                op_env_full_braket, newD, mode="braket"
            )
            superblock[psite + step].data = superblock_full[psite + step].data[
                :newD, :, :
            ]
            R = newD
        else:
            op_env_D_bra = op_env_thin
            op_env_D_braket = op_env_thin
            R = 1
        l, c, r = superblock[psite].data.shape
        assert R <= r + dD, f"{R=}, {r=}, {dD=}"
        # logger.debug(f"{op_env_thin=}")
        # logger.debug(f"{op_env_full=}")
        original_data = superblock[psite].data.copy()
        op_lcr = mps.operators_for_superH(
            psite=psite,
            op_sys=op_sys,
            op_env=op_env_thin,
            ints_site=None,
            hamiltonian=model.hamiltonian,
            A_is_sys=A_is_sys,
        )
        # logger.debug(f"{op_lcr=}")
        mps.exp_superH_propagation_direct(
            psite=psite,
            superblock_states=[superblock],
            op_lcr=op_lcr,
            matH_cas=model.hamiltonian,
            stepsize=dt,
            tensor_shapes_out=None,
        )
        # logger.debug(superblock[psite].data-original_data)
        propagate_data = superblock[psite].data.copy()

        superblock[psite].data = original_data.copy()
        op_lcr = mps.operators_for_superH(
            psite=psite,
            op_sys=op_sys,
            op_env=op_env_D_bra,
            ints_site=None,
            hamiltonian=model.hamiltonian,
            A_is_sys=A_is_sys,
        )
        # logger.debug(f"{op_lcr=}")
        logger.debug(f"{l=}, {c=}, {R=}")
        logger.debug(f"{superblock[psite].data.shape=}")
        mps.exp_superH_propagation_direct(
            psite=psite,
            superblock_states=[superblock],
            op_lcr=op_lcr,
            matH_cas=model.hamiltonian,
            stepsize=dt,
            tensor_shapes_out=(l, c, R),
        )
        # confirm propagated data is close to original data
        np.testing.assert_allclose(
            superblock[psite].data[:l, :c, :r], propagate_data, atol=1e-03
        )
        if psite != end_site:
            svalues, op_sys = mps.trans_next_psite_AsigmaB(
                psite=psite,
                superblock_states=[superblock],
                op_sys=op_sys,
                ints_site=None,
                matH_cas=model.hamiltonian,
                PsiB2AB=A_is_sys,
            )
            assert (
                svalues[0].shape == (R, R)
            ), f"{svalues[0].shape=} not match {(R, R)=} where {R=} and {r=} and {dD=}"
            op_lr = mps.operators_for_superK(
                psite=psite,
                op_sys=op_sys,
                op_env=op_env_D_braket,
                hamiltonian=model.hamiltonian,
                A_is_sys=A_is_sys,
            )

            svalues = mps.exp_superK_propagation_direct(
                op_lr=op_lr,
                hamiltonian=model.hamiltonian,
                svalues=svalues,
                stepsize=dt,
            )
            logger.debug(f"{np.linalg.svd(svalues[0], compute_uv=False)=}")
            mps.trans_next_psite_APsiB(
                psite=psite,
                superblock_states=[superblock],
                svalues=svalues,
                A_is_sys=A_is_sys,
            )
            if isinstance(mps.op_sys_sites, list):
                mps.op_sys_sites.append(op_sys)

    superblock_full = []
    begin_site = nsite - 1
    end_site = 0
    step = -1
    A_is_sys = False
    superblock_full = get_superblock_full(superblock, delta_rank=dD)

    op_sys = mps.construct_op_zerosite([superblock], model.hamiltonian)
    op_env_sites = mps.op_sys_sites[:]
    mps.op_sys_sites = [op_sys]
    # logger.debug(op_sys)
    # logger.debug(op_env_sites)
    psites_sweep = range(begin_site, end_site + step, step)
    for psite in psites_sweep:
        logger.debug(f"{psite=}")
        op_env_thin = op_env_sites.pop()
        if psite != end_site:
            op_env_previous = op_env_sites[-1]
            # logger.debug(f"{op_env_thin=}")
            # logger.debug(f"{op_env_previous=}")
            op_env_full_braket = get_op_block_full(
                mps=mps,
                psite=psite,
                superblock_states=[superblock],
                superblock_states_full=[superblock_full],
                op_block_previous=op_env_previous,
                hamiltonian=model.hamiltonian,
                construct_left=not A_is_sys,
                mode="braket",
            )
            op_env_full_bra = get_op_block_full(
                mps=mps,
                psite=psite,
                superblock_states=[superblock],
                superblock_states_full=[superblock_full],
                op_block_previous=op_env_previous,
                hamiltonian=model.hamiltonian,
                construct_left=not A_is_sys,
                mode="bra",
            )
            op_sys_thin = mps.op_sys_sites[-1]
            psi_left, sigvec, psi_right, op_sys_full_bra = (
                get_psi_sigvec_psi_fullblock(
                    mps=mps,
                    psite=psite,
                    to="<-",
                    op_block=op_sys_thin,
                    additional_rank=10,
                    hamiltonian=model.hamiltonian,
                )
            )
            psi_states_left = [psi_left]
            sigvec_states = [sigvec]
            psi_states_right = [psi_right]
            newD, error = calculate_projection_error(
                mps=mps,
                psite=psite,
                Dmax=100,
                p=1.0e-02,
                op_sys_full=op_sys_full_bra,
                op_sys_thin=op_sys_thin,
                op_env_full=op_env_full_bra,
                op_env_thin=op_env_previous,
                hamiltonian=model.hamiltonian,
                psi_states_left=psi_states_left,
                sigvec_states=sigvec_states,
                psi_states_right=psi_states_right,
                to="<-",
            )
            logger.debug(f"{newD=}, {error=}")
            op_env_D_bra = truncate_op_block(op_env_full_bra, newD, mode="bra")
            op_env_D_braket = truncate_op_block(
                op_env_full_braket, newD, mode="braket"
            )
            superblock[psite + step].data = superblock_full[psite + step].data[
                :, :, :newD
            ]
            L = newD
        else:
            op_env_D_bra = op_env_thin
            op_env_D_braket = op_env_thin
            L = 1
        l, c, r = superblock[psite].data.shape
        assert L <= l + dD, f"{L=}, {l=}, {dD=}"
        # logger.debug(f"{op_env_thin=}")
        # logger.debug(f"{op_env_full=}")
        original_data = superblock[psite].data.copy()
        op_lcr = mps.operators_for_superH(
            psite=psite,
            op_sys=op_sys,
            op_env=op_env_thin,
            ints_site=None,
            hamiltonian=model.hamiltonian,
            A_is_sys=A_is_sys,
        )
        # logger.debug(f"{op_lcr=}")
        mps.exp_superH_propagation_direct(
            psite=psite,
            superblock_states=[superblock],
            op_lcr=op_lcr,
            matH_cas=model.hamiltonian,
            stepsize=dt,
            tensor_shapes_out=None,
        )
        # logger.debug(superblock[psite].data-original_data)
        propagate_data = superblock[psite].data.copy()
        superblock[psite].data = original_data.copy()
        op_lcr = mps.operators_for_superH(
            psite=psite,
            op_sys=op_sys,
            op_env=op_env_D_bra,
            ints_site=None,
            hamiltonian=model.hamiltonian,
            A_is_sys=A_is_sys,
        )
        # logger.debug(f"{op_lcr=}")
        logger.debug(f"{L=}, {c=}, {r=}")
        mps.exp_superH_propagation_direct(
            psite=psite,
            superblock_states=[superblock],
            op_lcr=op_lcr,
            matH_cas=model.hamiltonian,
            stepsize=dt,
            tensor_shapes_out=(L, c, r),
        )
        # confirm propagated data is close to original data
        np.testing.assert_allclose(
            superblock[psite].data[:l, :c, :r], propagate_data, atol=1e-03
        )
        if psite != end_site:
            svalues, op_sys = mps.trans_next_psite_AsigmaB(
                psite=psite,
                superblock_states=[superblock],
                op_sys=op_sys,
                ints_site=None,
                matH_cas=model.hamiltonian,
                PsiB2AB=A_is_sys,
            )
            assert (
                svalues[0].shape == (L, L)
            ), f"{svalues[0].shape=} not match {(L, L)=} where {L=} and {l=} and {dD=}"
            op_lr = mps.operators_for_superK(
                psite=psite,
                op_sys=op_sys,
                op_env=op_env_D_braket,
                hamiltonian=model.hamiltonian,
                A_is_sys=A_is_sys,
            )

            svalues = mps.exp_superK_propagation_direct(
                op_lr=op_lr,
                hamiltonian=model.hamiltonian,
                svalues=svalues,
                stepsize=dt,
            )
            logger.debug(f"{np.linalg.svd(svalues[0], compute_uv=False)=}")
            mps.trans_next_psite_APsiB(
                psite=psite,
                superblock_states=[superblock],
                svalues=svalues,
                A_is_sys=A_is_sys,
            )
            if isinstance(mps.op_sys_sites, list):
                mps.op_sys_sites.append(op_sys)


def truncate_op_block(op_block, D: int, mode: Literal["bra", "braket"] = "bra"):
    op_block_truncated = {}
    for state_key, op_block_state in op_block.items():
        op_block_state_truncated = {}
        for key, value in op_block_state.items():
            assert len(value.shape) == 3, f"{value.shape=} is not 3"
            if mode == "bra":
                op_block_state_truncated[key] = value[:D, :, :]
            elif mode == "braket":
                op_block_state_truncated[key] = value[:D, :, :D]
            else:
                raise ValueError(f"{mode=} is not valid")
        op_block_truncated[state_key] = op_block_state_truncated
    return op_block_truncated


def calculate_projection_error(
    mps,
    psite: int,
    Dmax: int,
    p: float,
    op_sys_full,
    op_sys_thin,
    op_env_full,
    op_env_thin,
    hamiltonian,
    psi_states_left: list[np.ndarray] | list[jax.Array],
    sigvec_states: list[np.ndarray] | list[jax.Array],
    psi_states_right: list[np.ndarray] | list[jax.Array],
    to: Literal["->", "<-"],
):
    """
    calculate f(D) = |H(D', D)Psi_left|^2 - |K(D)sig|^2 + |H(D, D')Psi_right|^2
    """
    Dmin, Dtarget = sigvec_states[0].shape
    assert Dmin == Dtarget, f"{Dmin=} != {Dtarget=}"
    Dleft, d_left, Dtarget = psi_states_left[0].shape
    assert Dmin == Dtarget, f"{Dmin=} != {Dtarget=}"
    Dtarget, d_right, Dright = psi_states_right[0].shape
    assert Dmin == Dtarget, f"{Dmin=} != {Dtarget=}"
    Dmax = min((Dmax, Dleft * d_left, Dright * d_right))
    assert Dmin <= Dmax, f"{Dmin=} <= {Dmax=}"
    logger.debug(f"{Dmin=}, {Dmax=}")
    for D in range(Dmin, Dmax + 1):
        op_env_D = truncate_op_block(op_env_full, D, mode="bra")
        op_sys_D = truncate_op_block(op_sys_full, D, mode="bra")

        if Dmin == Dmax:
            return Dmin, 0.0

        op_lcr1 = mps.operators_for_superH(
            psite=psite if to == "->" else psite - 1,
            op_sys=op_sys_thin,
            op_env=op_env_D,
            ints_site=None,
            hamiltonian=hamiltonian,
            A_is_sys=True,
        )
        op_lcr2 = mps.operators_for_superH(
            psite=psite + 1 if to == "->" else psite,
            op_sys=op_sys_D,
            op_env=op_env_thin,
            ints_site=None,
            hamiltonian=hamiltonian,
            A_is_sys=True,
        )
        op_lr = mps.operators_for_superK(
            psite=psite if to == "->" else psite - 1,
            op_sys=op_sys_D,
            op_env=op_env_D,
            hamiltonian=hamiltonian,
            A_is_sys=True,
        )
        if isinstance(hamiltonian, TensorHamiltonian):
            Heff_left = multiplyH_MPS_direct_MPO(
                op_lcr1,
                psi_states_left,
                hamiltonian,
                tensor_shapes_out=(
                    psi_states_left[0].shape[0],
                    psi_states_left[0].shape[1],
                    D,
                ),
            )
            Heff_right = multiplyH_MPS_direct_MPO(
                op_lcr2,
                psi_states_right,
                hamiltonian,
                tensor_shapes_out=(
                    D,
                    psi_states_right[0].shape[1],
                    psi_states_right[0].shape[2],
                ),
            )
            Keff = multiplyK_MPS_direct_MPO(
                op_lr,
                sigvec_states,
                hamiltonian,
                tensor_shapes_out=(D, D),
            )
        else:
            Heff_left = multiplyH_MPS_direct(
                op_lcr1,
                psi_states_left,
                hamiltonian,
                tensor_shapes_out=(
                    psi_states_left[0].shape[0],
                    psi_states_left[0].shape[1],
                    D,
                ),
            )
            Heff_right = multiplyH_MPS_direct(
                op_lcr2,
                psi_states_right,
                hamiltonian,
                tensor_shapes_out=(
                    D,
                    psi_states_right[0].shape[1],
                    psi_states_right[0].shape[2],
                ),
            )
            Keff = multiplyK_MPS_direct(
                op_lr,
                sigvec_states,
                hamiltonian,
                tensor_shapes_out=(D, D),
            )
        psi_states_left_in = psi_states_left
        psi_states_right_in = psi_states_right
        sigvec_states_in = sigvec_states
        psi_states_left_out = Heff_left.dot(psi_states_left_in)
        psi_states_left_vec = Heff_left.stack(psi_states_left_out, extend=False)
        psi_states_right_out = Heff_right.dot(psi_states_right_in)
        psi_states_right_vec = Heff_right.stack(
            psi_states_right_out, extend=False
        )
        sigvec_states_out = Keff.dot(sigvec_states_in)
        sigvec_states_vec = Keff.stack(sigvec_states_out, extend=False)

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
        logger.debug(f"{D=}, {total_error=:4e}")
        if D > Dmin:
            metric = (total_error - total_error_prev) / total_error
            logger.debug(f"{D=}, {metric=:4e}")
            if metric < p:
                return D - 1, metric
        # uv run ruff check --fix --unsafe-fixes may delete following line but it is necessary
        total_error_prev = total_error  # noqa: F841

    return max(Dmin, D), 0.0


def get_psi_sigvec_psi_fullblock(
    mps,
    psite: int,
    to: Literal["->", "<-"],
    op_block,
    additional_rank: int,
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
    psi_site = mps.superblock_states[0][psite].copy()
    match to:
        case "->":
            B_site = mps.superblock_states[0][psite + 1]
            A_site, sigvec = psi_site.gauge_trf(key="Psi2Asigma")
            psi_prime_site = np.tensordot(sigvec, B_site.data, axes=(1, 0))
            A_site_full = A_site.thin_to_full(additional_rank=additional_rank)
            superblock_states = [
                [None for _ in range(len(mps.superblock_states[0]))]
            ]
            superblock_states[0][psite] = A_site
            superblock_states_bra = [
                [None for _ in range(len(mps.superblock_states[0]))]
            ]
            superblock_states_bra[0][psite] = A_site_full
            op_block_A_full = mps.renormalize_op_psite(
                psite=psite,
                superblock_states=superblock_states,
                op_block_states=op_block,
                ints_site=None,
                hamiltonian=hamiltonian,
                A_is_sys=True,
                superblock_states_bra=superblock_states_bra,
                superblock_states_ket=None,
            )
            return psi_site.data, sigvec, psi_prime_site, op_block_A_full
        case "<-":
            A_site = mps.superblock_states[0][psite - 1]
            B_site, sigvec = psi_site.gauge_trf(key="Psi2sigmaB")
            psi_prime_site = np.tensordot(A_site.data, sigvec, axes=(2, 0))
            B_site_full = B_site.thin_to_full(additional_rank=additional_rank)
            superblock_states = [
                [None for _ in range(len(mps.superblock_states[0]))]
            ]
            superblock_states[0][psite] = B_site
            superblock_states_bra = [
                [None for _ in range(len(mps.superblock_states[0]))]
            ]
            superblock_states_bra[0][psite] = B_site_full
            op_block_B_full = mps.renormalize_op_psite(
                psite=psite,
                superblock_states=superblock_states,
                op_block_states=op_block,
                ints_site=None,
                hamiltonian=hamiltonian,
                A_is_sys=False,
                superblock_states_bra=superblock_states_bra,
                superblock_states_ket=None,
            )
            return psi_prime_site, sigvec, psi_site.data, op_block_B_full
        case _:
            raise ValueError(f"{to=} is not valid")


def test_a1tdvp_projection_error():
    model = get_model()
    mps = pytdscf._mps_mpo.MPSCoefMPO.alloc_random(model)
    assert isinstance(mps, pytdscf._mps_mpo.MPSCoefMPO)
    superblock = mps.superblock_states[0]
    superblock_full = get_superblock_full(superblock, delta_rank=10)
    nsite = len(superblock)
    psite = 0
    op_env_sites = mps.construct_op_sites(
        [superblock],
        None,
        begin_site=nsite - 1,
        end_site=0,
        matH_cas=model.hamiltonian,
    )
    op_env_thin = op_env_sites[-psite - 2]
    op_env_full_bra = get_op_block_full(
        mps=mps,
        psite=psite,
        superblock_states=[superblock],
        superblock_states_full=[superblock_full],
        op_block_previous=op_env_thin,
        hamiltonian=model.hamiltonian,
        construct_left=False,
        mode="bra",
    )
    op_sys_thin = mps.construct_op_zerosite(
        superblock_states=[superblock],
        operator=model.hamiltonian,
    )
    psi_left, sigvec, psi_right, op_sys_full_bra = get_psi_sigvec_psi_fullblock(
        mps=mps,
        psite=psite,
        to="->",
        op_block=op_sys_thin,
        additional_rank=10,
        hamiltonian=model.hamiltonian,
    )
    psi_states_left = [psi_left]
    sigvec_states = [sigvec]
    psi_states_right = [psi_right]
    newD, error = calculate_projection_error(
        mps=mps,
        psite=psite,
        Dmax=100,
        p=1.0e-02,
        op_sys_full=op_sys_full_bra,
        op_sys_thin=op_sys_thin,
        op_env_full=op_env_full_bra,
        op_env_thin=op_env_thin,
        hamiltonian=model.hamiltonian,
        psi_states_left=psi_states_left,
        sigvec_states=sigvec_states,
        psi_states_right=psi_states_right,
        to="->",
    )
    logger.debug(f"{newD=}, {error=:3e}")


if __name__ == "__main__":
    pytest.main([__file__])
