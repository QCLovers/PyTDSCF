"""Test for Anharmonic Potential 3-mode MPS-MCTDH propagation"""

import math

import pytest


@pytest.mark.filterwarnings("ignore:DeprecationWarning")
def test_anharmonic_fbr_mpsmctdh_irelax_numpy():
    from pytdscf import BasInfo, Model, Simulator, units
    from pytdscf.basis import PrimBas_HO
    from pytdscf.hamiltonian_cls import read_potential_nMR
    from pytdscf.potentials.h2o_dipole import mu
    from pytdscf.potentials.h2o_potential import k_orig

    prim_info = [
        [
            PrimBas_HO(0.0, math.sqrt(k_orig[(1, 1)]) * units.au_in_cm1, 10),
            PrimBas_HO(0.0, math.sqrt(k_orig[(2, 2)]) * units.au_in_cm1, 10),
            PrimBas_HO(0.0, math.sqrt(k_orig[(3, 3)]) * units.au_in_cm1, 10),
        ]
    ]

    spf_info = [[5, 5, 5]]

    basinfo = BasInfo(prim_info, spf_info)

    dipole = read_potential_nMR(potential_emu=None, dipole_emu=mu)
    operators = {"hamiltonian": dipole}

    model = Model(basinfo, operators)
    model.m_aux_max = 4

    jobname = "anharmonic_fbr_operate"
    simulator = Simulator(jobname, model, backend="numpy", verbose=4)
    simulator.operate(maxstep=10)


if __name__ == "__main__":
    test_anharmonic_fbr_mpsmctdh_irelax_numpy()
