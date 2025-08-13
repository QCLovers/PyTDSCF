from functools import lru_cache
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import os
import pytest
from typing import Literal
import netCDF4 as nc
from pympo import (
    AssignManager,
    OpSite,
    SumOfProducts,
    get_eye_site,
)
from scipy.linalg import expm

from pytdscf import BasInfo, Exciton, Model, Simulator, units
from pytdscf.dvr_operator_cls import TensorOperator
from pytdscf.hamiltonian_cls import TensorHamiltonian

# Consider system consits of 1 central system spin and two bath spins.
# aligned: bath(i=0) - spin(i=1) - bath(i=2)
# H = H_sys + H_int
# H_sys = Bz * S0z
# H_int = J_01 (S0x S1x + S0y S1y + S0z S1z) + J_12 (S1x S2x + S1y S2y + S1z S2z)
# Include Haberkorn relaxation for system
# Include Lindblad relaxation for system


J_01 = 10.0
J_12 = 5.0
Bx = 10.0
By = 10.0
Bz = 10.0
k_Haberkorn = 1.0
k_Lindblad = 10.0

Sx = np.array([[0, 1], [1, 0]]) / 2
Sy = np.array([[0, -1j], [1j, 0]]) / 2
Sz = np.array([[1, 0], [0, -1]]) / 2
E = np.eye(2)

dt = 0.01
n_steps = 31

# Lindblad jump: from |1> to |0>
L0 = np.array([[0, 1], [0, 0]]) * np.sqrt(k_Lindblad)

def kron_three(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    return np.kron(np.kron(A, B), C)

def plot_rdms(rdms: np.ndarray, name: str):
    if "PYTEST_CURRENT_TEST" in os.environ:
        return
    plt.plot(rdms[:, 0, 0].real, label='00', marker='^', markevery=10)
    plt.plot(rdms[:, 0, 1].real, label='01', marker='x', markevery=10)
    plt.plot(rdms[:, 1, 0].real, label='10', marker='o', markevery=10)
    plt.plot(rdms[:, 1, 1].real, label='11', marker='v', markevery=10)
    plt.plot(rdms[:, 0, 0].real + rdms[:, 1, 1].real, label='00 + 11', marker='D', markevery=10)
    plt.legend()
    plt.savefig(Path(__file__).parent / 'build' / f'rdms-{name}.png')
    plt.show()
    plt.clf()

# Do not recalculate the exact solution. Use the cached one.
@lru_cache(maxsize=2)
def exact_solution(Lindblad=True):
    H = np.zeros((2**3, 2**3), dtype=np.complex128)
    H += Bx * kron_three(E, Sx, E)
    H += By * kron_three(E, Sy, E)
    H += Bz * kron_three(E, Sz, E)
    H += J_01 * kron_three(Sx, Sx, E)
    H += J_01 * kron_three(Sy, Sy, E)
    H += J_01 * kron_three(Sz, Sz, E)
    H += J_12 * kron_three(E, Sx, Sx)
    H += J_12 * kron_three(E, Sy, Sy)
    H += J_12 * kron_three(E, Sz, Sz)

    # Liouvillian can be vectorised.
    Liouville = (np.kron(H, np.eye(2**3)) - np.kron(np.eye(2**3), H.T)) / 1.0j

    Liouville -= k_Haberkorn * np.eye(2**6)
    if Lindblad:
        Lj = kron_three(E, L0, E)
        Liouville += np.kron(Lj, Lj.conj()) - 0.5 * (np.kron(Lj.conj().T @ Lj, np.eye(2**3)) + np.kron(np.eye(2**3), Lj.T @ Lj.conj()))

    # rho(0) = 1 otimes |0> <0| otimes 1 / Z
    dm = kron_three(E / 2, np.array([[1, 0], [0, 0]]), E / 2)
    dm = dm.reshape(-1)

    propagator = expm(Liouville * dt)
    # reduced density matrix
    rdms = []
    for _ in range(n_steps):
        rdms.append(np.einsum('abcadc->bd', dm.reshape(2, 2, 2, 2, 2, 2)))
        dm = propagator @ dm
    rdms = np.array(rdms)
    plot_rdms(rdms, 'exact')

    return rdms

@pytest.mark.parametrize("backend", ['numpy', 'jax'])
def test_sum_wavefunction_trajectory(backend: Literal['numpy', 'jax']):
    rdms_exact = exact_solution(Lindblad=False)
    sx0 = OpSite("sx0", 0, value=Sx)
    sy0 = OpSite("sy0", 0, value=Sy)
    sz0 = OpSite("sz0", 0, value=Sz)
    sx1 = OpSite("sx1", 1, value=Sx)
    sy1 = OpSite("sy1", 1, value=Sy)
    sz1 = OpSite("sz1", 1, value=Sz)
    sx2 = OpSite("sx2", 2, value=Sx)
    sy2 = OpSite("sy2", 2, value=Sy)
    sz2 = OpSite("sz2", 2, value=Sz)
    E1 = OpSite("E1", 1, value=E)

    sop = SumOfProducts()
    sop += Bx * sx1
    sop += By * sy1
    sop += Bz * sz1
    sop += J_01 * (sx0 * sx1 + sy0 * sy1 + sz0 * sz1)
    sop += J_12 * (sx1 * sx2 + sy1 * sy2 + sz1 * sz2)
    sop += -1.0j * k_Haberkorn / 2 * E1
    sop = sop.simplify()
    am = AssignManager(sop)
    am.assign()
    mpo = am.numerical_mpo()

    delta_t = dt * units.au_in_fs
    basis = [Exciton(nstate=2) for _ in range(3)]
    basinfo = BasInfo([basis], spf_info=None)

    op_dict ={
        ((0, 0), (1, 1), (2, 2)): TensorOperator(mpo=mpo)
    }
    H = TensorHamiltonian(
        3, potential=[[op_dict]], kinetic=None, backend=backend
    )

    model = Model(basinfo=basinfo, operators={"hamiltonian": H})

    hps = [
        [[1, 0], [1, 0], [1, 0]], # |↑↑↑>
        [[1, 0], [1, 0], [0, 1]], # |↑↑↓>
        [[0, 1], [1, 0], [1, 0]], # |↓↑↑>
        [[0, 1], [1, 0], [0, 1]], # |↓↑↓>
    ]
    density_sums = []
    for i, hp in enumerate(hps):
        model.init_HartreeProduct = [hp]
        model.m_aux_max = 64 # no compression
        jobname = f"traj_{backend}_{i}"
        simulator = Simulator(
            jobname=jobname,
            model=model,
            backend=backend,
            verbose=0
        )
        simulator.propagate(
            reduced_density=([(1, 1)], 1),
            maxstep=n_steps,
            stepsize=delta_t,
            autocorr=False,
            energy=False,
            norm=False,
            populations=False,
            conserve_norm=False, # Since Haberkorn relaxation is included
            integrator='lanczos', # Since H is still skew-Hermitian
        )
        with nc.Dataset(f"{jobname}_prop/reduced_density.nc", "r") as file:
            density_data_real = file.variables[f"rho_({1}, {1})_0"][
                :
            ]["real"]
            density_data_imag = file.variables[f"rho_({1}, {1})_0"][
                :
            ]["imag"]
            time_data = file.variables["time"][:]
        shutil.rmtree(f"{jobname}_prop", ignore_errors=True)
        os.remove(f"wf_{jobname}.pkl")

        density_data = np.array(density_data_real) + 1.0j * np.array(
            density_data_imag
        )
        time_data = np.array(time_data)
        if i == 0:
            density_sums = density_data
        else:
            density_sums += density_data

    density_sums /= len(hps)
    plot_rdms(density_sums, 'sum_wavefunction_trajectory')
    np.testing.assert_allclose(rdms_exact[0, :, :], density_sums[0, :, :], atol=1e-6)
    np.testing.assert_allclose(rdms_exact[30, :, :], density_sums[30, :, :], atol=1e-6)


@pytest.mark.parametrize("backend,supergate,scale", [
    ('numpy', False, 1),
    ('numpy', True, 1),
    ('jax', False, 1),
    ('jax', True, 1),
])
def test_vectorised_density_matrix(
        backend: Literal['numpy', 'jax'],
        supergate: bool,
        scale: int
    ):
    rdms_exact = exact_solution(Lindblad=True)
    SxE, ESx = np.kron(Sx, E), np.kron(E, Sx.T)
    SyE, ESy = np.kron(Sy, E), np.kron(E, Sy.T)
    SzE, ESz = np.kron(Sz, E), np.kron(E, Sz.T)

    sxE0, Esx0 = OpSite("sxE0", 0, value=SxE), OpSite("Esx0", 0, value=ESx)
    syE0, Esy0 = OpSite("syE0", 0, value=SyE), OpSite("Esy0", 0, value=ESy)
    szE0, Esz0 = OpSite("szE0", 0, value=SzE), OpSite("Esz0", 0, value=ESz)
    sxE1, ESx1 = OpSite("sxE1", 1, value=SxE), OpSite("ESx1", 1, value=ESx)
    syE1, ESy1 = OpSite("syE1", 1, value=SyE), OpSite("ESy1", 1, value=ESy)
    szE1, ESz1 = OpSite("szE1", 1, value=SzE), OpSite("ESz1", 1, value=ESz)
    sxE2, ESx2 = OpSite("sxE2", 2, value=SxE), OpSite("ESx2", 2, value=ESx)
    syE2, ESy2 = OpSite("syE2", 2, value=SyE), OpSite("ESy2", 2, value=ESy)
    szE2, ESz2 = OpSite("szE2", 2, value=SzE), OpSite("ESz2", 2, value=ESz)
    EE1 = OpSite("EE1", 1, value=np.kron(E, E))
    LL = OpSite("LL", 1, value=np.kron(L0, L0.conj()))
    LLE, ELL = OpSite("LLE", 1, value=np.kron(L0.conj().T@L0, E)), OpSite("ELL", 1, value=np.kron(E, L0.T@L0.conj()))


    sop = SumOfProducts()
    sop += Bx * (sxE1 - ESx1)
    sop += By * (syE1 - ESy1)
    sop += Bz * (szE1 - ESz1)
    sop += J_01 * (
        sxE0 * sxE1 + syE0 * syE1 + szE0 * szE1
        - Esx0 * ESx1 - Esy0 * ESy1 - Esz0 * ESz1
    )
    sop += J_12 * (
        sxE1 * sxE2 + syE1 * syE2 + szE1 * szE2
        - ESx1 * ESx2 - ESy1 * ESy2 - ESz1 * ESz2
    )
    sop += -1.0j * k_Haberkorn / 2 * (EE1 + EE1)

    if supergate:
        D = np.kron(L0, L0.conj()) - 0.5 * (
            np.kron(L0.conj().T @ L0, np.eye(2))
            + np.kron(np.eye(2), L0.T @ L0.conj())
        )
        print(D.shape)
        op_dict = {
            ((1,1),) : TensorOperator(
                mpo=[expm(D * dt/scale)[None, :, :, None]],
                legs=(1,1)
            )
        }
        expDt = TensorHamiltonian(
            3, potential=[[op_dict]], kinetic=None, backend=backend
        )
    else:
        sop += 1.0j * LL - 1.0j/2 * (LLE + ELL)

    sop = sop.simplify()
    am = AssignManager(sop)
    am.assign()
    mpo = am.numerical_mpo()

    delta_t = dt * units.au_in_fs
    basis = [Exciton(nstate=2**2) for _ in range(3)]
    basinfo = BasInfo([basis], spf_info=None)

    op_dict ={
        ((0, 0), (1, 1), (2, 2)): TensorOperator(mpo=mpo)
    }
    H = TensorHamiltonian(
        3, potential=[[op_dict]], kinetic=None, backend=backend
    )

    model = Model(
        basinfo=basinfo,
        operators={"hamiltonian": H},
        space="Liouville",
        one_gate_to_apply=expDt if supergate else None,
    )

    model.init_HartreeProduct = [
        [
            np.eye(2).reshape(-1),
            np.array([[1, 0], [0, 0]]).reshape(-1),
            np.eye(2).reshape(-1),
        ]
    ]
    model.m_aux_max = 64 # no compression
    jobname = f"vec_liouville_{backend}_{supergate}_{scale}"
    simulator = Simulator(
        jobname=jobname,
        model=model,
        backend=backend,
        verbose=0
    )
    simulator.propagate(
        reduced_density=([(1, 1)], 1),
        maxstep=n_steps*scale,
        stepsize=delta_t / scale,
        autocorr=False,
        energy=False,
        norm=False,
        populations=False,
        conserve_norm=False, # Since Haberkorn relaxation is included
        integrator='lanczos',
    )
    with nc.Dataset(f"{jobname}_prop/reduced_density.nc", "r") as file:
        density_data_real = file.variables[f"rho_({1}, {1})_0"][
            :
        ]["real"]
        density_data_imag = file.variables[f"rho_({1}, {1})_0"][
            :
        ]["imag"]
        time_data = file.variables["time"][:]
    shutil.rmtree(f"{jobname}_prop", ignore_errors=True)
    os.remove(f"wf_{jobname}.pkl")

    rdms = np.array(density_data_real) + 1.0j * np.array(
        density_data_imag
    )
    plot_rdms(rdms, 'vectorised_density_matrix')
    np.testing.assert_allclose(rdms_exact[0, :, :], rdms[0, :, :], atol=1e-6)
    np.testing.assert_allclose(rdms_exact[30, :, :], rdms[30*scale, :, :], atol=1e-3 / scale**2 if supergate else 1e-6)

    # remove output files
    shutil.rmtree(f"{jobname}_prop", ignore_errors=True)

@pytest.mark.parametrize("backend", ['numpy', 'jax'])
def test_purified_mps(backend: Literal['numpy', 'jax']):
    rdms_exact = exact_solution(Lindblad=False)
    sx0 = OpSite("sx0", 1, value=Sx)
    sy0 = OpSite("sy0", 1, value=Sy)
    sz0 = OpSite("sz0", 1, value=Sz)
    sx1 = OpSite("sx1", 2, value=Sx)
    sy1 = OpSite("sy1", 2, value=Sy)
    sz1 = OpSite("sz1", 2, value=Sz)
    sx2 = OpSite("sx2", 3, value=Sx)
    sy2 = OpSite("sy2", 3, value=Sy)
    sz2 = OpSite("sz2", 3, value=Sz)
    E1 = OpSite("E1", 2, value=E)

    sop = SumOfProducts()
    sop += Bx * sx1
    sop += By * sy1
    sop += Bz * sz1
    sop += J_01 * (sx0 * sx1 + sy0 * sy1 + sz0 * sz1)
    sop += J_12 * (sx1 * sx2 + sy1 * sy2 + sz1 * sz2)
    sop += -1.0j * k_Haberkorn / 2 * E1

    eye_sites = [
        get_eye_site(i, n_basis=2) for i in range(5)
    ]
    dummy = 1
    for eye_site in eye_sites:
        dummy *= eye_site
    sop += 0.0 * dummy

    sop = sop.simplify()
    am = AssignManager(sop)
    am.assign()
    mpo = am.numerical_mpo()

    delta_t = dt * units.au_in_fs
    basis = [Exciton(nstate=2) for _ in range(5)]
    basinfo = BasInfo([basis], spf_info=None)

    op_dict ={
        ((0,), (1, 1), (2, 2), (3, 3), (4,)): TensorOperator(mpo=mpo)
    }
    H = TensorHamiltonian(
        5, potential=[[op_dict]], kinetic=None, backend=backend
    )

    model = Model(basinfo=basinfo, operators={"hamiltonian": H})

    anc_0 = np.zeros((1, 2, 2))
    anc_0[0, 0, 0] = 1
    anc_0[0, 1, 1] = 1
    phys_1 = np.zeros((2, 2, 1))
    phys_1[0, 0, 0] = 1
    phys_1[1, 1, 0] = 1
    phys_2 = np.zeros((1, 2, 1))
    phys_2[0, 0, 0] = 1
    phys_3 = np.zeros((1, 2, 2))
    phys_3[0, 0, 0] = 1
    phys_3[0, 1, 1] = 1
    anc_4 = np.zeros((2, 2, 1))
    anc_4[0, 0, 0] = 1
    anc_4[1, 1, 0] = 1
    model.init_HartreeProduct = [[anc_0, phys_1, phys_2, phys_3, anc_4]]
    model.m_aux_max = 64 # no compression
    jobname = f"purified_mps_{backend}"
    simulator = Simulator(
        jobname=jobname,
        model=model,
        backend=backend,
        verbose=0
    )
    simulator.propagate(
        reduced_density=([(2, 2)], 1),
        maxstep=n_steps,
        stepsize=delta_t,
        autocorr=False,
        energy=False,
        norm=False,
        populations=False,
        conserve_norm=False, # Since Haberkorn relaxation is included
        integrator='lanczos', # Since H is still skew-Hermitian
    )
    with nc.Dataset(f"{jobname}_prop/reduced_density.nc", "r") as file:
        density_data_real = file.variables[f"rho_({2}, {2})_0"][
            :
        ]["real"]
        density_data_imag = file.variables[f"rho_({2}, {2})_0"][
            :
        ]["imag"]
        time_data = file.variables["time"][:]
    shutil.rmtree(f"{jobname}_prop", ignore_errors=True)
    os.remove(f"wf_{jobname}.pkl")

    rdms = np.array(density_data_real) + 1.0j * np.array(
        density_data_imag
    )
    time_data = np.array(time_data)

    plot_rdms(rdms, 'purified_mps')
    np.testing.assert_allclose(rdms_exact[0, :, :], rdms[0, :, :], atol=1e-6)
    np.testing.assert_allclose(rdms_exact[30, :, :], rdms[30, :, :], atol=1e-6)

    # remove output files
    shutil.rmtree(f"{jobname}_prop", ignore_errors=True)


if __name__ == '__main__':
    #exact_solution(Lindblad=True)
    # exact_solution(Lindblad=False)
    # test_sum_wavefunction_trajectory(backend='numpy')
    # test_sum_wavefunction_trajectory(backend='jax')
    # test_vectorised_density_matrix(backend='numpy')
    # test_vectorised_density_matrix(backend='numpy', supergate=True, scale=1)
    # test_vectorised_density_matrix(backend='jax', supergate=False)
    test_purified_mps(backend='jax')
