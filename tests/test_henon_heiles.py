from typing import Callable, Dict, List, Tuple

import pytest
from discvar import HarmonicOscillator as HO

from pytdscf import units
from pytdscf.dvr_operator_cls import (
    TensorOperator,
    construct_kinetic_operator,
    construct_nMR_recursive,
)
from pytdscf.hamiltonian_cls import TensorHamiltonian
from pytdscf.model_cls import BasInfo, Model
from pytdscf.simulator_cls import Simulator

jobname = "henon_heiles"


@pytest.mark.filterwarnings("ignore:DeprecationWarning")
@pytest.mark.parametrize(
    "ω, λ, f, N, m, Δt, backend, ener",
    [
        [4000, 1.0e-05, 1, 5, 4, 0.01, "jax", 0.027338011517478895],
        [2000, 1.0e-03, 2, 5, 4, 0.001, "numpy", 0.018225341011652626],
    ],
)
def test_henon_heiles(ω, λ, f, N, m, Δt, backend, ener):
    """Test for Henon-Heiles potential MPS-SM propagation

    Henon-Heiles potential is given by:

    But PyTDSCF adopts mass-weighted coordinate, thus the Hamiltonian is given by

    H = 1/2 Σᵢ₌₁ᶠ ( - 𝜕²/𝜕Qᵢ² + ω² Qᵢ²) + λ ω^{3/2} ( Σᵢ₌₁ᶠ⁻¹ Qᵢ²Qᵢ₊₁ - 1/3 Qᵢ₊₁³)

    Args:
        ω (float): frequency in cm-1
        λ (float): coupling strength in a.u.
        f (int): degree of freedom
        N (int): number of grid points for each degree of freedom
        m (int): MPS bond dimension
        Δt (float): time step size in femtosecond. If spectrum norm of Hamiltonian is large, Δt should be smaller.
    """

    dvr_prims = [HO(N, ω) for _ in range(f)]
    basinfo = BasInfo([dvr_prims])

    ω_au = ω / units.au_in_cm1

    # Potential Function of each degree of freedom pair
    henon_heiles_func: Dict[Tuple[int, ...], Callable] = {}
    for idof in range(f):
        if idof == 0:
            henon_heiles_func[(0,)] = lambda Q1: pow(ω_au, 2) / 2 * Q1**2
            if f > 1:
                henon_heiles_func[(0, 1)] = (
                    lambda Q1, Q2: λ * pow(ω_au, 3 / 2) * (Q1**2 * Q2)
                )
        elif idof == f - 1:
            henon_heiles_func[(f - 1,)] = (
                lambda Qf: pow(ω_au, 2) / 2 * Qf**2
                - λ * pow(ω_au, 3 / 2) / 3 * Qf**3
            )
        else:
            henon_heiles_func[(idof,)] = (
                lambda Qi: pow(ω_au, 2) / 2 * Qi**2
                - λ * pow(ω_au, 3 / 2) / 3 * Qi**3
            )
            henon_heiles_func[(idof, idof + 1)] = (
                lambda Qi, Qi1: λ * pow(ω_au, 3 / 2) * (Qi**2 * Qi1)
            )

    mpo = construct_nMR_recursive(
        dvr_prims, nMR=2, func=henon_heiles_func, rate=0.99999999999
    )

    # MPO has legs on (0,1,2, ... ,f-1) sites. This legs are given by tuple key
    V = {tuple([idof for idof in range(f)]): TensorOperator(mpo=mpo)}
    # V₀₀ is given by
    potential = [[V]]

    # Kinetic energy operator is given by
    K = construct_kinetic_operator(dvr_prims)
    # K₀₀ is given by
    kinetic = [[K]]

    H = TensorHamiltonian(
        ndof=f, potential=potential, kinetic=kinetic, backend=backend
    )
    operators = {"hamiltonian": H}

    model = Model(basinfo=basinfo, operators=operators)
    model.m_aux_max = m
    vib_GS: List[float] = [1.0] + [0.0] * (N - 1)
    vib_ES: List[float] = [0.0] + [1.0] + [0.0] * (N - 2)
    model.init_weight_VIBSTATE: List[List[List[float]]] = [
        [vib_ES] + [vib_GS] * (f - 1)
    ]
    simulator = Simulator(jobname=jobname, model=model, backend=backend)
    ener_calc, wf = simulator.propagate(maxstep=3, stepsize=Δt)
    assert pytest.approx(ener_calc) == ener


if __name__ == "__main__":
    test_henon_heiles(ω=2000, λ=1.0e-07, f=10, N=10, m=20, Δt=0.1)
