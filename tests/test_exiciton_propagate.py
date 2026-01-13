"""Test for Linear Vibronic Coupling model"""

import numpy as np
import pytest
from discvar import HarmonicOscillator as HO

from pytdscf.basis import Exciton
from pytdscf.dvr_operator_cls import TensorOperator
from pytdscf.hamiltonian_cls import TensorHamiltonian
from pytdscf.model_cls import Model
from pytdscf.simulator_cls import Simulator
from pytdscf.units import au_in_cm1
from pytdscf.util import read_nc

freqs_cm1 = [1000, 2000, 3000]
omega2 = [(freq / au_in_cm1) ** 2 for freq in freqs_cm1]
nspf = nprim = 8
prim_info = [HO(nprim, freq, units="cm-1") for freq in freqs_cm1] + [
    Exciton(nstate=2, names=["S0", "S1"])
]


@pytest.mark.filterwarnings("ignore:DeprecationWarning")
@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_exiciton_propagate(backend):
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

    model = Model(prim_info, operators, bond_dim=2)
    model.init_HartreeProduct = [
        [ho.get_unitary()[0].tolist() for ho in prim_info[:3]]
        + [np.array([0.0, 1.0]).tolist()]
    ]
    # Starts from the S1 state

    jobname = "LVC_Exciton_test"
    # const.regularize_site = False
    simulator = Simulator(jobname, model, backend=backend)
    ener_calc, wf = simulator.propagate(
        stepsize=0.1, maxstep=20, reduced_density=([(3, 3), (0, 0), (0, 0, 3, 3)], 1)
    )
    assert pytest.approx(ener_calc) == 0.010000180312707298
    rdm = read_nc(f"{jobname}_prop/reduced_density.nc", [(3, 3)])
    np.testing.assert_allclose(rdm[(3, 3)][-1],
    np.array(
        [[1.86417721e-02+1.60379680e-20j, 2.87367863e-02-6.91095824e-02j],
        [2.87367863e-02+6.91095824e-02j, 9.81358228e-01-7.40721885e-18j]]
    ), atol = 1e-09)

if __name__ == "__main__":
    test_exiciton_propagate(backend="numpy")
