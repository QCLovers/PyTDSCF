"""Test for Anharmonic Potential 3-mode MPS-MCTDH propagation"""

import math

import pytest


@pytest.mark.filterwarnings("ignore:DeprecationWarning")
def test_anharmonic_fbr_mpsmctdh_relax_numpy():
    from pytdscf import units
    from pytdscf.basis._primints_cls import PrimBas_HO
    from pytdscf.hamiltonian_cls import read_potential_nMR
    from pytdscf.model_cls import BasInfo, Model
    from pytdscf.potentials.h2o_potential import k_orig
    from pytdscf.simulator_cls import Simulator

    prim_info = [
        [
            PrimBas_HO(0.0, math.sqrt(k_orig[(1, 1)]) * units.au_in_cm1, 10),
            PrimBas_HO(0.0, math.sqrt(k_orig[(2, 2)]) * units.au_in_cm1, 10),
            PrimBas_HO(0.0, math.sqrt(k_orig[(3, 3)]) * units.au_in_cm1, 10),
        ]
    ]

    spf_info = [[5, 5, 5]]

    basinfo = BasInfo(prim_info, spf_info)

    hamiltonian = read_potential_nMR(k_orig)

    operators = {"hamiltonian": hamiltonian}

    model = Model(basinfo, operators)
    model.m_aux_max = 4

    jobname = "anharmonic_fbr_relax"
    simulator = Simulator(jobname, model, backend="numpy")
    ener, wf = simulator.relax(improved=False, maxstep=1)
    assert pytest.approx(ener) == 0.021360262338234466
