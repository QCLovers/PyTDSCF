"""Test for Harmonic Potential 3-mode MPS-SM (DVR) propagation"""

import pytest


@pytest.mark.filterwarnings("ignore:DeprecationWarning")
def test_harmonic_dvr_func_full_mpssm_propagate_jax():
    from discvar import HarmonicOscillator

    from pytdscf import units
    from pytdscf.dvr_operator_cls import (
        construct_fulldimensional,
        construct_kinetic_operator,
    )
    from pytdscf.hamiltonian_cls import TensorHamiltonian
    from pytdscf.model_cls import BasInfo, Model
    from pytdscf.simulator_cls import Simulator

    prim_info = [
        [
            HarmonicOscillator(5, 1500, 0.0),
            HarmonicOscillator(5, 2000, 0.0),
            HarmonicOscillator(5, 2500, 0.0),
        ]
    ]

    basinfo = BasInfo(prim_info)

    def PES(q1, q2, q3):
        """V = Σ_i ω_i^2 q_i^2 / 2"""
        return (
            (1500 / units.au_in_cm1) ** 2 / 2 * q1**2
            + (2000 / units.au_in_cm1) ** 2 / 2 * q2**2
            + (2500 / units.au_in_cm1) ** 2 / 2 * q3**2
        )

    potential = [[construct_fulldimensional(dvr_prims=prim_info[0], func=PES)]]
    kinetic = [[construct_kinetic_operator(dvr_prims=prim_info[0])]]
    hamiltonian = TensorHamiltonian(
        ndof=3,
        potential=potential,
        kinetic=kinetic,
        decompose_type="SVD",
        rate=0.9999999,
        backend="jax",
    )

    operators = {"hamiltonian": hamiltonian}

    model = Model(basinfo, operators)
    model.m_aux_max = 4

    jobname = "harmonic_dvr_jax"
    simulator = Simulator(jobname, model, backend="jax")
    ener_calc, wf = simulator.propagate(maxstep=3, stepsize=0.1, restart=True)
    assert pytest.approx(ener_calc) == 0.02024362306067599


if __name__ == "__main__":
    test_harmonic_dvr_func_full_mpssm_propagate_jax()
