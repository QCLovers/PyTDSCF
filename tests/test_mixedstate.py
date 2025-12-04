import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pytest
from pympo import (
    AssignManager,
    OpSite,
    SumOfProducts,
    get_eye_site,
)
from scipy.linalg import expm

from pytdscf import Exciton, Model, Simulator, units
from pytdscf.dvr_operator_cls import TensorOperator
from pytdscf.hamiltonian_cls import TensorHamiltonian

# Consider system consits of 1 central system spin and two bath spins.
# aligned: bath(i=0, S=1/2) - spin(i=1, S=1) - bath(i=2, S=1/2)
# H = H_sys + H_int
# H_sys = Bz * S0z
# H_int = J_01 (S0x S1x + S0y S1y + S0z S1z) + J_12 (S1x S2x + S1y S2y + S1z S2z)
# Include Haberkorn relaxation for system
# Include Lindblad relaxation for system


J_01 = 10.0 * 0.1
J_12 = 5.0 * 0.1
Bx = 10.0 * 0.1
By = 10.0 * 0.1
Bz = 10.0 * 0.1
k_Haberkorn = 0.1
k_L_amp, k_L_deph = 6.0, 9.0

Sx = np.array([[0, 1], [1, 0]]) / 2
Sy = np.array([[0, -1j], [1j, 0]]) / 2
Sz = np.array([[1, 0], [0, -1]]) / 2
# spin 1 operators
Iz = np.diag([1, 0, -1]) / 2
# ladder operators
Ip = (
    np.array(
        [[0, np.sqrt(2), 0], [0, 0, np.sqrt(2)], [0, 0, 0]], dtype=np.complex128
    )
    / 2
)
Im = (
    np.array(
        [[0, 0, 0], [np.sqrt(2), 0, 0], [0, np.sqrt(2), 0]], dtype=np.complex128
    )
    / 2
)
# spin 1 operators
Ix = 0.5 * (Ip + Im)
Iy = -0.5j * (Ip - Im)
# bath 0 operators
Hdim = 2 * 3 * 2

E2 = np.eye(2)
E3 = np.eye(3)

dt = 0.1
n_steps = 11

# --- Lindblad jumps ---
L_amp_middle = np.array(
    [[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=complex
) * np.sqrt(k_L_amp)
L_deph_middle = Iz * np.sqrt(k_L_deph)


def kron_three(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    assert A.shape == C.shape == (2, 2) or A.shape == C.shape == (4, 4)
    assert B.shape == (3, 3) or B.shape == (9, 9)
    return np.kron(np.kron(A, B), C)


def plot_rdms(rdms: np.ndarray, name: str):
    if "PYTEST_CURRENT_TEST" in os.environ:
        return
    plt.plot(rdms[:, 0, 0].real, label="00", marker="^", markevery=10)
    plt.plot(rdms[:, 0, 1].real, label="01", marker="x", markevery=10)
    plt.plot(rdms[:, 1, 0].real, label="10", marker="o", markevery=10)
    plt.plot(rdms[:, 1, 1].real, label="11", marker="v", markevery=10)
    plt.plot(rdms[:, 2, 0].real, label="20", marker="s", markevery=10)
    plt.plot(rdms[:, 2, 1].real, label="21", marker="d", markevery=10)
    plt.plot(rdms[:, 2, 2].real, label="22", marker="^", markevery=10)
    plt.plot(
        np.einsum("taa->t", rdms).real, label="trace", marker="D", markevery=10
    )
    plt.legend()
    plt.savefig(Path(__file__).parent / "build" / f"rdms-{name}.png")
    plt.show()
    plt.clf()


# Do not recalculate the exact solution. Use the cached one.
@lru_cache(maxsize=3)
def exact_solution(ini_diag=(0, 0, 1), Lindblad=True, krylov=False):
    Pini = np.diag(ini_diag).astype(np.complex128)
    Pini /= np.linalg.trace(Pini)
    H = np.zeros((Hdim, Hdim), dtype=np.complex128)
    H += Bx * kron_three(E2, Ix, E2)
    H += By * kron_three(E2, Iy, E2)
    H += Bz * kron_three(E2, Iz, E2)
    H += J_01 * kron_three(Sx, Ix, E2)
    H += J_01 * kron_three(Sy, Iy, E2)
    H += J_01 * kron_three(Sz, Iz, E2)
    H += J_12 * kron_three(E2, Ix, Sx)
    H += J_12 * kron_three(E2, Iy, Sy)
    H += J_12 * kron_three(E2, Iz, Sz)

    # Liouvillian can be vectorised.
    Liouville = (np.kron(H, np.eye(Hdim)) - np.kron(np.eye(Hdim), H.T)) / 1.0j

    Liouville -= k_Haberkorn * np.eye(Hdim**2)
    if Lindblad:
        for _Lj in [L_amp_middle, L_deph_middle]:
            Lj = kron_three(E2, _Lj, E2)
            Liouville += np.kron(Lj, Lj.conj()) - 0.5 * (
                np.kron(Lj.conj().T @ Lj, np.eye(Hdim))
                + np.kron(np.eye(Hdim), Lj.T @ Lj.conj())
            )

    # rho(0) = 1 otimes |0> <0| otimes 1 / Z
    dm = kron_three(E2 / 2, Pini, E2 / 2)
    dm = dm.reshape(-1)

    propagator = expm(Liouville * dt)

    def lanczos_propagate(dm, L):
        # Use Hermitian Lanczos on A_eff = -i * (L + k I); final scaling e^{-k dt}
        Aeff = L / 1.0j
        # np.testing.assert_allclose(Aeff, Aeff.conj().T)
        maxdim = min(len(dm), 10)
        beta0 = np.linalg.norm(dm)
        if abs(beta0) < 1e-13:
            return dm
        v_prev = np.zeros_like(dm)
        v_curr = dm / beta0
        alphas = []
        betas = []
        basis_vectors = [v_curr]
        m_effective = maxdim
        for j in range(maxdim):
            w = Aeff @ v_curr
            alpha_j = np.vdot(v_curr, w)
            alphas.append(alpha_j)
            w = w - alpha_j * v_curr
            if j > 0:
                w = w - betas[-1] * v_prev
            beta_next = np.linalg.norm(w)
            if abs(beta_next) < 1e-12:
                m_effective = j + 1
                break
            betas.append(beta_next)
            v_prev, v_curr = v_curr, w / beta_next
            basis_vectors.append(v_curr)

        T = np.zeros((m_effective, m_effective), dtype=np.complex128)
        for i in range(m_effective):
            T[i, i] = alphas[i]
            if i + 1 < m_effective:
                T[i, i + 1] = betas[i]
                T[i + 1, i] = betas[i]
        print(alphas)
        Q = np.column_stack(basis_vectors[:m_effective])
        e1 = np.zeros((m_effective,), dtype=np.complex128)
        e1[0] = 1.0
        # exp(dt * L) = e^{-k dt} exp(-i * dt * T)
        y_small = expm(1.0j * dt * T) @ e1
        dm_next = beta0 * (Q @ y_small)
        return dm_next

    def arnoldi_propagate(dm, L):
        # short iterative Arnoldi to approximate exp(dt * Liouville) @ dm
        maxdim = min(len(dm), 10)
        beta = np.linalg.norm(dm)
        if beta == 0:
            return dm
        Q = np.zeros((dm.size, maxdim), dtype=np.complex128)
        Q[:, 0] = dm / beta
        H = np.zeros((maxdim, maxdim), dtype=np.complex128)
        m_effective = maxdim
        for j in range(maxdim - 1):
            w = L @ Q[:, j]
            for i in range(j + 1):
                H[i, j] = np.vdot(Q[:, i], w)
                w = w - H[i, j] * Q[:, i]
            h_next = np.linalg.norm(w)
            if h_next < 1e-14:
                m_effective = j + 1
                H = H[:m_effective, :m_effective]
                Q = Q[:, :m_effective]
                break
            H[j + 1, j] = h_next
            Q[:, j + 1] = w / h_next
        else:
            H = H[:m_effective, :m_effective]

        # If the projected matrix is tridiagonal, Lanczos would coincide numerically in this case.

        e1 = np.zeros((m_effective,), dtype=np.complex128)
        e1[0] = 1.0
        y_small = expm(dt * H) @ e1
        dm_next = beta * (Q @ y_small)
        return dm_next

    # reduced density matrix
    dm_arnoldi = dm.copy()
    dm_lanczos = dm.copy()
    rdms = []
    for _ in range(n_steps):
        rdms.append(np.einsum("abcadc->bd", dm.reshape(2, 3, 2, 2, 3, 2)))
        if krylov:
            # dm = propagate(dm)
            dm_arnoldi = arnoldi_propagate(dm_arnoldi, Liouville)
            dm_lanczos = lanczos_propagate(dm_lanczos, Liouville)
            dm = propagator @ dm
            print(
                f"diff(arn-exact) = {np.linalg.norm(dm - dm_arnoldi) / np.linalg.norm(dm)}"
            )
            print(
                f"diff(lan-exact) = {np.linalg.norm(dm - dm_lanczos) / np.linalg.norm(dm)}"
            )
        else:
            dm = propagator @ dm
    rdms = np.array(rdms)
    plot_rdms(rdms, "exact")

    return rdms


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_sum_wavefunction_trajectory(backend: Literal["numpy", "jax"], scale=1):
    rdms_exact = exact_solution(Lindblad=False)
    sx0 = OpSite("sx0", 0, value=Sx)
    sy0 = OpSite("sy0", 0, value=Sy)
    sz0 = OpSite("sz0", 0, value=Sz)
    sx1 = OpSite("sx1", 1, value=Ix)
    sy1 = OpSite("sy1", 1, value=Iy)
    sz1 = OpSite("sz1", 1, value=Iz)
    sx2 = OpSite("sx2", 2, value=Sx)
    sy2 = OpSite("sy2", 2, value=Sy)
    sz2 = OpSite("sz2", 2, value=Sz)
    E1 = OpSite("E1", 1, value=E3)

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
    basis = [Exciton(nstate=2), Exciton(nstate=3), Exciton(nstate=2)]
    model = Model(basis, operators={"hamiltonian": mpo}, bond_dim=64)

    hps = [
        [[1, 0], [0, 0, 1], [1, 0]],  # |↑-1↑>
        [[1, 0], [0, 0, 1], [0, 1]],  # |↑-1↓>
        [[0, 1], [0, 0, 1], [1, 0]],  # |↓-1↑>
        [[0, 1], [0, 0, 1], [0, 1]],  # |↓-1↓>
    ]
    density_sums = []
    for i, hp in enumerate(hps):
        model.init_HartreeProduct = [hp]
        jobname = f"traj_{backend}_{i}"
        simulator = Simulator(
            jobname=jobname, model=model, backend=backend, verbose=0
        )
        simulator.propagate(
            reduced_density=([(1, 1)], 1),
            maxstep=n_steps * scale,
            stepsize=delta_t / scale,
            autocorr=True,
            energy=False,
            norm=False,
            populations=False,
            conserve_norm=False,  # Since Haberkorn relaxation is included
            integrator="arnoldi",  # Since H is Hermitian and P = 1 for Haberkorn relaxation
        )
        with nc.Dataset(f"{jobname}_prop/reduced_density.nc", "r") as file:
            density_data_real = file.variables[f"rho_({1}, {1})_0"][:]["real"]
            density_data_imag = file.variables[f"rho_({1}, {1})_0"][:]["imag"]
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
    plot_rdms(density_sums, "sum_wavefunction_trajectory")
    np.testing.assert_allclose(
        rdms_exact[0, :, :], density_sums[0, :, :], atol=1e-12
    )
    np.testing.assert_allclose(
        rdms_exact[n_steps - 1, :, :],
        density_sums[(n_steps - 1) * scale, :, :],
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "backend,supergate,scale",
    [
        ("numpy", False, 1),
        ("numpy", True, 2),
        ("jax", False, 1),
        ("jax", True, 1),
    ],
)
def test_vectorised_density_matrix(
    backend: Literal["numpy", "jax"], supergate: bool, scale: int
):
    ini_diag = (0, 0, 1)
    rdms_exact = exact_solution(Lindblad=True, ini_diag=ini_diag)
    SxE, ESx = np.kron(Sx, E2), np.kron(E2, Sx.T)
    SyE, ESy = np.kron(Sy, E2), np.kron(E2, Sy.T)
    SzE, ESz = np.kron(Sz, E2), np.kron(E2, Sz.T)
    IxE, EIx = np.kron(Ix, E3), np.kron(E3, Ix.T)
    IyE, EIy = np.kron(Iy, E3), np.kron(E3, Iy.T)
    IzE, EIz = np.kron(Iz, E3), np.kron(E3, Iz.T)

    sxE0, Esx0 = OpSite("sxE0", 0, value=SxE), OpSite("Esx0", 0, value=ESx)
    syE0, Esy0 = OpSite("syE0", 0, value=SyE), OpSite("Esy0", 0, value=ESy)
    szE0, Esz0 = OpSite("szE0", 0, value=SzE), OpSite("Esz0", 0, value=ESz)
    sxE1, ESx1 = OpSite("sxE1", 1, value=IxE), OpSite("ESx1", 1, value=EIx)
    syE1, ESy1 = OpSite("syE1", 1, value=IyE), OpSite("ESy1", 1, value=EIy)
    szE1, ESz1 = OpSite("szE1", 1, value=IzE), OpSite("ESz1", 1, value=EIz)
    sxE2, ESx2 = OpSite("sxE2", 2, value=SxE), OpSite("ESx2", 2, value=ESx)
    syE2, ESy2 = OpSite("syE2", 2, value=SyE), OpSite("ESy2", 2, value=ESy)
    szE2, ESz2 = OpSite("szE2", 2, value=SzE), OpSite("ESz2", 2, value=ESz)
    EE1 = OpSite("EE1", 1, value=np.kron(E3, E3))

    sop = SumOfProducts()
    sop += Bx * (sxE1 - ESx1)
    sop += By * (syE1 - ESy1)
    sop += Bz * (szE1 - ESz1)
    sop += J_01 * (
        sxE0 * sxE1
        + syE0 * syE1
        + szE0 * szE1
        - Esx0 * ESx1
        - Esy0 * ESy1
        - Esz0 * ESz1
    )
    sop += J_12 * (
        sxE1 * sxE2
        + syE1 * syE2
        + szE1 * szE2
        - ESx1 * ESx2
        - ESy1 * ESy2
        - ESz1 * ESz2
    )
    sop += -1.0j * k_Haberkorn / 2 * (EE1 + EE1)

    if supergate:
        D = np.zeros((9, 9), dtype=np.complex128)
        for _Lj in [L_amp_middle, L_deph_middle]:
            D += np.kron(_Lj, _Lj.conj()) - 0.5 * (
                np.kron(_Lj.conj().T @ _Lj, E3)
                + np.kron(E3, _Lj.T @ _Lj.conj())
            )
        op_dict = {
            ((1, 1),): TensorOperator(
                mpo=[expm(D * dt / scale)[None, :, :, None]], legs=(1, 1)
            )
        }
        expDt = TensorHamiltonian(
            3, potential=[[op_dict]], kinetic=None, backend=backend
        )
    else:
        for _Lj in [L_amp_middle, L_deph_middle]:
            LL = OpSite("LL", 1, value=np.kron(_Lj, _Lj.conj()))
            LLE, ELL = (
                OpSite("LLE", 1, value=np.kron(_Lj.conj().T @ _Lj, E3)),
                OpSite("ELL", 1, value=np.kron(E3, _Lj.T @ _Lj.conj())),
            )
            sop += 1.0j * LL - 1.0j / 2 * (LLE + ELL)

    sop = sop.simplify()
    am = AssignManager(sop)
    am.assign()
    mpo = am.numerical_mpo()

    delta_t = dt * units.au_in_fs
    basis = [Exciton(nstate=4), Exciton(nstate=9), Exciton(nstate=4)]

    model = Model(
        basis,
        operators={"hamiltonian": mpo},
        space="Liouville",
        one_gate_to_apply=expDt if supergate else None,
        bond_dim=64,
    )

    Pini = np.diag(ini_diag).astype(np.complex128)
    Pini /= np.linalg.trace(Pini)

    model.init_HartreeProduct = [
        [
            E2.reshape(-1),
            Pini.reshape(-1),
            E2.reshape(-1),
        ]
    ]
    jobname = f"vec_liouville_{backend}_{supergate}_{scale}"
    simulator = Simulator(
        jobname=jobname, model=model, backend=backend, verbose=0
    )
    simulator.propagate(
        reduced_density=([(1, 1)], 1),
        maxstep=n_steps * scale,
        stepsize=delta_t / scale,
        autocorr=False,
        energy=False,
        norm=False,
        populations=False,
        conserve_norm=False,  # Since Haberkorn relaxation is included
        integrator="arnoldi",
    )
    with nc.Dataset(f"{jobname}_prop/reduced_density.nc", "r") as file:
        density_data_real = file.variables[f"rho_({1}, {1})_0"][:]["real"]
        density_data_imag = file.variables[f"rho_({1}, {1})_0"][:]["imag"]
        time_data = file.variables["time"][:]
    shutil.rmtree(f"{jobname}_prop", ignore_errors=True)
    os.remove(f"wf_{jobname}.pkl")

    rdms = np.array(density_data_real) + 1.0j * np.array(density_data_imag)
    plot_rdms(rdms, "vectorised_density_matrix")
    np.testing.assert_allclose(rdms_exact[0, :, :], rdms[0, :, :], atol=1e-12)
    np.testing.assert_allclose(
        rdms_exact[n_steps - 1, :, :],
        rdms[(n_steps - 1) * scale, :, :],
        atol=1e-2 / scale**2 if supergate else 1e-12,
    )

    # remove output files
    shutil.rmtree(f"{jobname}_prop", ignore_errors=True)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_purified_mps(backend: Literal["numpy", "jax"], scale=1):
    rdms_exact = exact_solution(Lindblad=False)
    sx0 = OpSite("sx0", 1, value=Sx)
    sy0 = OpSite("sy0", 1, value=Sy)
    sz0 = OpSite("sz0", 1, value=Sz)
    sx1 = OpSite("sx1", 2, value=Ix)
    sy1 = OpSite("sy1", 2, value=Iy)
    sz1 = OpSite("sz1", 2, value=Iz)
    sx2 = OpSite("sx2", 3, value=Sx)
    sy2 = OpSite("sy2", 3, value=Sy)
    sz2 = OpSite("sz2", 3, value=Sz)
    E1 = OpSite("E1", 2, value=E3)

    sop = SumOfProducts()
    sop += Bx * sx1
    sop += By * sy1
    sop += Bz * sz1
    sop += J_01 * (sx0 * sx1 + sy0 * sy1 + sz0 * sz1)
    sop += J_12 * (sx1 * sx2 + sy1 * sy2 + sz1 * sz2)
    sop += -1.0j * k_Haberkorn / 2 * E1

    eye_sites = [
        get_eye_site(0, n_basis=2),
        get_eye_site(1, n_basis=2),
        get_eye_site(2, n_basis=3),
        get_eye_site(3, n_basis=2),
        get_eye_site(4, n_basis=2),
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
    basis = [
        Exciton(nstate=2),
        Exciton(nstate=2),
        Exciton(nstate=3),
        Exciton(nstate=2),
        Exciton(nstate=2),
    ]
    model = Model(basis, operators={"hamiltonian": mpo}, bond_dim=64)

    anc_0 = np.zeros((1, 2, 2))
    anc_0[0, 0, 0] = 1
    anc_0[0, 1, 1] = 1
    phys_1 = np.zeros((2, 2, 1))
    phys_1[0, 0, 0] = 1
    phys_1[1, 1, 0] = 1
    phys_2 = np.zeros((1, 3, 1))
    phys_2[0, 2, 0] = 1
    phys_3 = np.zeros((1, 2, 2))
    phys_3[0, 0, 0] = 1
    phys_3[0, 1, 1] = 1
    anc_4 = np.zeros((2, 2, 1))
    anc_4[0, 0, 0] = 1
    anc_4[1, 1, 0] = 1
    model.init_HartreeProduct = [[anc_0, phys_1, phys_2, phys_3, anc_4]]
    jobname = f"purified_mps_{backend}"
    simulator = Simulator(
        jobname=jobname, model=model, backend=backend, verbose=0
    )
    simulator.propagate(
        reduced_density=([(2, 2)], 1),
        maxstep=n_steps * scale,
        stepsize=delta_t / scale,
        autocorr=False,
        energy=False,
        norm=False,
        populations=False,
        conserve_norm=False,  # Since Haberkorn relaxation is included
        integrator="arnoldi",  # Since H is still skew-Hermitian
    )
    with nc.Dataset(f"{jobname}_prop/reduced_density.nc", "r") as file:
        density_data_real = file.variables[f"rho_({2}, {2})_0"][:]["real"]
        density_data_imag = file.variables[f"rho_({2}, {2})_0"][:]["imag"]
        time_data = file.variables["time"][:]
    shutil.rmtree(f"{jobname}_prop", ignore_errors=True)
    os.remove(f"wf_{jobname}.pkl")

    rdms = np.array(density_data_real) + 1.0j * np.array(density_data_imag)
    time_data = np.array(time_data)

    plot_rdms(rdms, "purified_mps")
    np.testing.assert_allclose(rdms_exact[0, :, :], rdms[0, :, :], atol=1e-12)
    np.testing.assert_allclose(
        rdms_exact[n_steps - 1, :, :],
        rdms[(n_steps - 1) * scale, :, :],
        atol=1e-12,
    )

    # remove output files
    shutil.rmtree(f"{jobname}_prop", ignore_errors=True)


@pytest.mark.parametrize(
    "backend,scale",
    [
        ("numpy", 2),
        ("jax", 1),
    ],
)
def test_purified_mps_kraus_single_site(
    backend: Literal["numpy", "jax"], scale: int
):
    rdms_exact = exact_solution(Lindblad=True)
    kraus_dim = 24
    Ek = np.eye(kraus_dim)
    sx0 = OpSite("sx0", 1, value=Sx)
    sy0 = OpSite("sy0", 1, value=Sy)
    sz0 = OpSite("sz0", 1, value=Sz)
    sx1 = OpSite(
        "sx1E", 2, value=np.kron(Ix, Ek)
    )  # phsycal index x kraus index
    sy1 = OpSite("sy1E", 2, value=np.kron(Iy, Ek))
    sz1 = OpSite("sz1E", 2, value=np.kron(Iz, Ek))
    sx2 = OpSite("sx2", 3, value=Sx)
    sy2 = OpSite("sy2", 3, value=Sy)
    sz2 = OpSite("sz2", 3, value=Sz)
    E1 = OpSite("E1", 2, value=np.kron(E3, Ek))

    sop = SumOfProducts()
    sop += Bx * sx1
    sop += By * sy1
    sop += Bz * sz1
    sop += J_01 * (sx0 * sx1 + sy0 * sy1 + sz0 * sz1)
    sop += J_12 * (sx1 * sx2 + sy1 * sy2 + sz1 * sz2)
    sop += -1.0j * k_Haberkorn / 2 * E1

    eye_sites = [
        get_eye_site(0, n_basis=2),
        get_eye_site(1, n_basis=2),
        get_eye_site(2, n_basis=3 * kraus_dim),
        get_eye_site(3, n_basis=2),
        get_eye_site(4, n_basis=2),
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
    basis = [
        Exciton(nstate=2),
        Exciton(nstate=2),
        Exciton(nstate=3 * kraus_dim),
        Exciton(nstate=2),
        Exciton(nstate=2),
    ]
    from pytdscf.kraus import lindblad_to_kraus, trace_kraus_dim

    Bs = lindblad_to_kraus(
        [L_amp_middle, L_deph_middle], dt / scale, backend=backend
    )
    print(Bs.shape)

    model = Model(
        basis, operators={"hamiltonian": mpo}, kraus_op={(2,): Bs}, bond_dim=64
    )

    anc_0 = np.zeros((1, 2, 2))
    anc_0[0, 0, 0] = 1
    anc_0[0, 1, 1] = 1
    phys_1 = np.zeros((2, 2, 1))
    phys_1[0, 0, 0] = 1
    phys_1[1, 1, 0] = 1
    phys_2 = np.zeros((1, 3 * kraus_dim, 1))
    phys_2[0, kraus_dim * 2, 0] = 1
    phys_3 = np.zeros((1, 2, 2))
    phys_3[0, 0, 0] = 1
    phys_3[0, 1, 1] = 1
    anc_4 = np.zeros((2, 2, 1))
    anc_4[0, 0, 0] = 1
    anc_4[1, 1, 0] = 1
    model.init_HartreeProduct = [[anc_0, phys_1, phys_2, phys_3, anc_4]]
    jobname = f"purified_mps_kraus_single_site_{backend}"
    simulator = Simulator(
        jobname=jobname, model=model, backend=backend, verbose=0
    )
    simulator.propagate(
        reduced_density=([(2, 2)], 1),
        maxstep=n_steps * scale,
        stepsize=delta_t / scale,
        autocorr=False,
        energy=False,
        norm=False,
        populations=False,
        conserve_norm=False,  # Since Haberkorn relaxation is included
        integrator="arnoldi",  # Since H is still skew-Hermitian
    )
    with nc.Dataset(f"{jobname}_prop/reduced_density.nc", "r") as file:
        density_data_real = file.variables[f"rho_({2}, {2})_0"][:]["real"]
        density_data_imag = file.variables[f"rho_({2}, {2})_0"][:]["imag"]
        time_data = file.variables["time"][:]
    shutil.rmtree(f"{jobname}_prop", ignore_errors=True)
    os.remove(f"wf_{jobname}.pkl")

    rdms = np.array(density_data_real) + 1.0j * np.array(density_data_imag)
    rdms = trace_kraus_dim(rdms, 3)
    time_data = np.array(time_data)

    plot_rdms(rdms, "purified_mps_kraus")
    np.testing.assert_allclose(rdms_exact[0, :, :], rdms[0, :, :], atol=1e-12)
    np.testing.assert_allclose(
        rdms_exact[n_steps - 1, :, :],
        rdms[(n_steps - 1) * scale, :, :],
        atol=1e-02 / scale**2,
    )

    # remove output files
    shutil.rmtree(f"{jobname}_prop", ignore_errors=True)


@pytest.mark.parametrize(
    "backend,scale",
    [
        ("numpy", 2),
        ("jax", 1),
    ],
)
def test_purified_mps_kraus_two_site(
    backend: Literal["numpy", "jax"], scale: int
):
    rdms_exact = exact_solution(Lindblad=True, ini_diag=(0, 1, 1))
    kraus_dim = 32
    sx0 = OpSite("sx0", 1, value=Sx)
    sy0 = OpSite("sy0", 1, value=Sy)
    sz0 = OpSite("sz0", 1, value=Sz)
    sx1 = OpSite("sx1E", 2, value=Ix)  # phsycal index x kraus index
    sy1 = OpSite("sy1E", 2, value=Iy)
    sz1 = OpSite("sz1E", 2, value=Iz)
    sx2 = OpSite("sx2", 4, value=Sx)
    sy2 = OpSite("sy2", 4, value=Sy)
    sz2 = OpSite("sz2", 4, value=Sz)
    E1 = OpSite("E1", 2, value=E3)

    sop = SumOfProducts()
    sop += Bx * sx1
    sop += By * sy1
    sop += Bz * sz1
    sop += J_01 * (sx0 * sx1 + sy0 * sy1 + sz0 * sz1)
    sop += J_12 * (sx1 * sx2 + sy1 * sy2 + sz1 * sz2)
    sop += -1.0j * k_Haberkorn / 2 * E1

    eye_sites = [
        get_eye_site(0, n_basis=2),
        get_eye_site(1, n_basis=2),
        get_eye_site(2, n_basis=3),
        get_eye_site(3, n_basis=kraus_dim),
        get_eye_site(4, n_basis=2),
        get_eye_site(5, n_basis=2),
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
    basis = [
        Exciton(nstate=2),
        Exciton(nstate=2),
        Exciton(nstate=3),
        Exciton(nstate=kraus_dim),
        Exciton(nstate=2),
        Exciton(nstate=2),
    ]

    from pytdscf.kraus import lindblad_to_kraus

    Bs = lindblad_to_kraus(
        [L_amp_middle, L_deph_middle], dt / scale, backend=backend
    )
    print(Bs.shape)

    model = Model(
        basis,
        operators={"hamiltonian": mpo},
        kraus_op={(2, 3): Bs},
        bond_dim=64,
    )

    anc_0 = np.zeros((1, 2, 2))
    anc_0[0, 0, 0] = 1
    anc_0[0, 1, 1] = 1
    phys_1 = np.zeros((2, 2, 1))
    phys_1[0, 0, 0] = 1
    phys_1[1, 1, 0] = 1
    phys_2 = np.zeros((1, 3, 2))
    phys_2[0, 2, 0] = 1
    phys_2[0, 1, 1] = 1
    anc_3 = np.zeros((2, kraus_dim, 1))
    anc_3[0, 0, 0] = 1
    anc_3[1, 1, 0] = 1
    phys_4 = np.zeros((1, 2, 2))
    phys_4[0, 0, 0] = 1
    phys_4[0, 1, 1] = 1
    anc_5 = np.zeros((2, 2, 1))
    anc_5[0, 0, 0] = 1
    anc_5[1, 1, 0] = 1
    model.init_HartreeProduct = [[anc_0, phys_1, phys_2, anc_3, phys_4, anc_5]]
    jobname = f"purified_mps_kraus_{backend}"
    simulator = Simulator(
        jobname=jobname, model=model, backend=backend, verbose=0
    )
    simulator.propagate(
        reduced_density=([(2, 2)], 1),
        maxstep=n_steps * scale,
        stepsize=delta_t / scale,
        autocorr=False,
        energy=False,
        norm=False,
        populations=False,
        conserve_norm=False,  # Since Haberkorn relaxation is included
        integrator="arnoldi",  # Since H is still skew-Hermitian
    )
    with nc.Dataset(f"{jobname}_prop/reduced_density.nc", "r") as file:
        density_data_real = file.variables[f"rho_({2}, {2})_0"][:]["real"]
        density_data_imag = file.variables[f"rho_({2}, {2})_0"][:]["imag"]
        time_data = file.variables["time"][:]
    shutil.rmtree(f"{jobname}_prop", ignore_errors=True)
    os.remove(f"wf_{jobname}.pkl")

    rdms = np.array(density_data_real) + 1.0j * np.array(density_data_imag)
    time_data = np.array(time_data)

    plot_rdms(rdms, "purified_mps_kraus")
    np.testing.assert_allclose(rdms_exact[0, :, :], rdms[0, :, :], atol=1e-12)
    np.testing.assert_allclose(
        rdms_exact[n_steps - 1, :, :],
        rdms[(n_steps - 1) * scale, :, :],
        atol=1e-02 / scale**2,
    )

    # remove output files
    shutil.rmtree(f"{jobname}_prop", ignore_errors=True)


if __name__ == "__main__":
    # test_purified_mps_kraus_single_site(backend='numpy', scale=8)
    test_purified_mps_kraus_two_site(backend="numpy", scale=2)
