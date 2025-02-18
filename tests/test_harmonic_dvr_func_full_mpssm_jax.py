"""Test for Harmonic Potential 3-mode MPS-SM (DVR) relaxation"""

import pytest


def test_harmonic_dvr_func_full_mpssm_irelax_jax():
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
    # <- Wrapper should be consistent with Simulator

    operators = {"hamiltonian": hamiltonian}

    model = Model(basinfo, operators)
    model.m_aux_max = 4

    jobname = "harmonic_dvr_jax"
    simulator = Simulator(jobname, model, backend="jax")
    ener_calc, wf = simulator.relax(maxstep=3, stepsize=0.1)
    assert pytest.approx(ener_calc) == 0.013669005758739458


def test_harmonic_dvr_func_full_mpssm_operate_jax():
    from discvar import HarmonicOscillator

    from pytdscf.dvr_operator_cls import construct_fulldimensional
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

    def DMS(q1, q2, q3):
        """V = Σ_i ω_i^2 q_i^2 / 2"""
        return 0.1 * q1 + 0.1 * q2 + 0.1 * q3

    potential = [[construct_fulldimensional(dvr_prims=prim_info[0], func=DMS)]]
    hamiltonian = TensorHamiltonian(
        ndof=3,
        potential=potential,
        kinetic=None,
        decompose_type="SVD",
        rate=0.9999999,
        backend="jax",
    )
    hamiltonian.coupleJ = [[1.0]]  # scalar term

    operators = {"hamiltonian": hamiltonian}

    model = Model(basinfo, operators)
    model.m_aux_max = 4

    jobname = "harmonic_dvr_jax"
    simulator = Simulator(jobname, model, backend="jax")
    norm_calc, wf = simulator.operate(restart=True, maxstep=5)
    assert pytest.approx(norm_calc) == 1.6490051381599562


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
    assert pytest.approx(ener_calc) == 0.019185297685193108


if __name__ == "__main__":
    test_harmonic_dvr_func_full_mpssm_irelax_jax()
    test_harmonic_dvr_func_full_mpssm_operate_jax()
    test_harmonic_dvr_func_full_mpssm_propagate_jax()
