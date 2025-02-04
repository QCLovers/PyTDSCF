"""Test for Anharmonic Potential 3-mode MPS-SM propagation"""

import math

import pytest


@pytest.mark.filterwarnings("ignore:DeprecationWarning")
def test_anharmonic_fbr_mpssm_propagate_jax():
    from pytdscf import units
    from pytdscf.basis._primints_cls import PrimBas_HO
    from pytdscf.hamiltonian_cls import read_potential_nMR
    from pytdscf.model_cls import BasInfo, Model
    from pytdscf.potentials.h2o_potential import k_orig
    from pytdscf.simulator_cls import Simulator

    prim_info = [
        [
            PrimBas_HO(0.0, math.sqrt(k_orig[(1, 1)]) * units.au_in_cm1, 6),
            PrimBas_HO(0.0, math.sqrt(k_orig[(2, 2)]) * units.au_in_cm1, 6),
            PrimBas_HO(0.0, math.sqrt(k_orig[(3, 3)]) * units.au_in_cm1, 6),
        ]
    ]

    basinfo = BasInfo(prim_info)

    hamiltonian = read_potential_nMR(k_orig)

    operators = {"hamiltonian": hamiltonian}

    model = Model(basinfo, operators)
    model.m_aux_max = 4

    jobname = "anharmonic_fbr_propagate_sm"
    simulator = Simulator(jobname, model, backend="jax")
    ener_calc, wf = simulator.propagate(maxstep=1)
    assert pytest.approx(ener_calc) == 0.021360262338234466


if __name__ == "__main__":
    test_anharmonic_fbr_mpssm_propagate_jax()
