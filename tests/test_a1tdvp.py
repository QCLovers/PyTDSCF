import sys
from typing import Literal

import numpy as np
import pytest
from discvar import HarmonicOscillator as HO
from loguru import logger

import pytdscf
import pytdscf._const_cls
from pytdscf._mps_cls import get_superblock_full
from pytdscf._site_cls import (
    truncate_sigvec,
    validate_Atensor,
    validate_Btensor,
)
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
    lamb = 0.001
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


def sweep(
    mps,
    hamiltonian: TensorHamiltonian,
    to: Literal["->", "<-"],
    dt: float,
):
    from pytdscf._const_cls import const
    Dmax = const.Dmax
    dD = const.dD
    p_proj = const.p_proj
    p_svd = const.p_svd

    superblock = mps.superblock_states[0]
    nsite = len(superblock)
    match to:
        case "->":
            step = 1
            A_is_sys = True
            begin_site = 0
            end_site = nsite - 1
        case "<-":
            step = -1
            A_is_sys = False
            begin_site = nsite - 1
            end_site = 0
    superblock_states_full = [get_superblock_full(superblock, delta_rank=dD)]
    op_sys = mps.construct_op_zerosite([superblock], operator=hamiltonian)
    if mps.op_sys_sites is None:
        op_env_sites = mps.construct_op_sites(
            mps.superblock_states,
            ints_site=None,
            begin_site=end_site,
            end_site=begin_site,
            matH_cas=hamiltonian,
        )
    else:
        op_env_sites = mps.op_sys_sites[:]
    mps.op_sys_sites = [op_sys]
    psites_sweep = range(begin_site, end_site + step, step)
    for psite in psites_sweep:
        logger.debug(f"{psite=}")
        op_env = op_env_sites.pop()
        if psite != end_site:
            assert len(op_env_sites) > 0
            op_env_previous = op_env_sites[-1]
            newD, error, op_env_D_bra, op_env_D_braket = (
                mps.get_adaptive_rank_and_block(
                    psite=psite,
                    superblock_states=mps.superblock_states,
                    superblock_states_full=superblock_states_full,
                    op_env_previous=op_env_previous,
                    hamiltonian=hamiltonian,
                    to=to,
                )
            )
            logger.debug(f"{newD=}, {error=}")
            if to == "->":
                R = newD
            else:
                L = newD
        else:
            op_env_D_bra = op_env
            op_env_D_braket = op_env
            if to == "->":
                R = 1
            else:
                L = 1
        l, c, r = superblock[psite].data.shape
        if to == "->":
            L, C, _ = superblock[psite].data.shape
            assert R <= r + dD, f"{R=}, {r=}, {dD=}"
        else:
            _, C, R = superblock[psite].data.shape
            assert L <= l + dD, f"{L=}, {l=}, {dD=}"
        original_data = superblock[psite].data.copy()
        op_lcr = mps.operators_for_superH(
            psite=psite,
            op_sys=op_sys,
            op_env=op_env,
            ints_site=None,
            hamiltonian=hamiltonian,
            A_is_sys=A_is_sys,
        )
        mps.exp_superH_propagation_direct(
            psite=psite,
            superblock_states=mps.superblock_states,
            op_lcr=op_lcr,
            matH_cas=hamiltonian,
            stepsize=dt,
            tensor_shapes_out=None,
        )
        propagate_data = superblock[psite].data.copy()
        superblock[psite].data = original_data.copy()
        op_lcr = mps.operators_for_superH(
            psite=psite,
            op_sys=op_sys,
            op_env=op_env_D_bra,
            ints_site=None,
            hamiltonian=hamiltonian,
            A_is_sys=A_is_sys,
        )
        mps.exp_superH_propagation_direct(
            psite=psite,
            superblock_states=mps.superblock_states,
            op_lcr=op_lcr,
            matH_cas=hamiltonian,
            stepsize=dt,
            tensor_shapes_out=(L, C, R),
        )
        # confirm propagated data is close to original data
        np.testing.assert_allclose(
            superblock[psite].data[:l, :c, :r], propagate_data, atol=1e-03
        )
        if psite != end_site:
            svalues, op_sys = mps.trans_next_psite_AsigmaB(
                psite=psite,
                superblock_states=mps.superblock_states,
                op_sys=op_sys,
                ints_site=None,
                matH_cas=hamiltonian,
                PsiB2AB=A_is_sys,
            )
            assert (
                svalues[0].shape == (newD, newD)
            ), f"{svalues[0].shape=} not match {(newD, newD)=} where {newD=} and {r=} and {dD=}"
            op_lr = mps.operators_for_superK(
                psite=psite,
                op_sys=op_sys,
                op_env=op_env_D_braket,
                hamiltonian=hamiltonian,
                A_is_sys=A_is_sys,
            )

            svalues = mps.exp_superK_propagation_direct(
                op_lr=op_lr,
                hamiltonian=hamiltonian,
                svalues=svalues,
                stepsize=dt,
            )
            # logger.debug(f"{np.linalg.svd(svalues[0], compute_uv=False)=}")
            if to == "->":
                Asite, svalues[0], Bsite = truncate_sigvec(
                    superblock[psite], svalues[0], superblock[psite + 1], p_svd
                )
            else:
                Asite, svalues[0], Bsite = truncate_sigvec(
                    superblock[psite - 1], svalues[0], superblock[psite], p_svd
                )
            validate_Atensor(Asite)
            validate_Btensor(Bsite)
            if psite == begin_site:
                op_sys_prev = mps.construct_op_zerosite(
                    [superblock], operator=hamiltonian
                )
            else:
                op_sys_prev = mps.op_sys_sites[-1]
            op_sys = mps.renormalize_op_psite(
                psite=psite,
                superblock_states=mps.superblock_states,
                op_block_states=op_sys_prev,
                ints_site=None,
                hamiltonian=hamiltonian,
                A_is_sys=A_is_sys,
            )
            logger.debug(f"{np.linalg.svd(svalues[0], compute_uv=False)=}")
            mps.trans_next_psite_APsiB(
                psite=psite,
                superblock_states=mps.superblock_states,
                svalues=svalues,
                A_is_sys=A_is_sys,
            )
            if isinstance(mps.op_sys_sites, list):
                mps.op_sys_sites.append(op_sys)


@pytest.mark.parametrize("use_class_method", [True, False])
def test_a1tdvp_sweep(use_class_method: bool):
    model = get_model()
    pytdscf._const_cls.const.set_runtype(
        use_jax=False, jobname="a1tdvp", adaptive=True, adaptive_p_proj=1.0e-05, adaptive_p_svd=1.0e-07, adaptive_Dmax=100, adaptive_dD=30
    )
    mps = pytdscf._mps_mpo.MPSCoefMPO.alloc_random(model)
    assert isinstance(mps, pytdscf._mps_mpo.MPSCoefMPO)
    superblock = mps.superblock_states[0]
    nsite = len(superblock)
    dt = 0.1
    # logger.debug(f"{superblock=}")
    pytdscf._mps_cls.canonicalize(superblock, orthogonal_center=0)
    for i_iter in range(5):
        if use_class_method:
            mps.propagate_along_sweep(
                None,
                model.hamiltonian,
                dt,
                begin_site=0,
                end_site=nsite - 1,
            )
            mps.propagate_along_sweep(
                None,
                model.hamiltonian,
                dt,
                begin_site=nsite - 1,
                end_site=0,
            )
        else:
            sweep(
                mps,
                model.hamiltonian,
                "->",
                dt,
            )
            sweep(
                mps,
                model.hamiltonian,
                "<-",
                dt,
            )
        logger.info(
            f"{i_iter=}, {[core.shape[2] for core in mps.superblock_states[0][:-1]]}"
        )

if __name__ == "__main__":
    test_a1tdvp_sweep()
