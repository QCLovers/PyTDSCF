"""Test for Harmonic Potential 3-mode MPS-SM (DVR) operate"""

import pytest


@pytest.mark.filterwarnings("ignore:DeprecationWarning")
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
    simulator.operate(restart=True, maxstep=5)
