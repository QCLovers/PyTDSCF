"""Test for Harmonic Potential 3-mode MPS-MCTDH propagation"""

import pytest


@pytest.mark.filterwarnings("ignore:DeprecationWarning")
def test_harmonic_fbr_mpsmctdh_propagate_numpy():
    from pytdscf.basis._primints_cls import PrimBas_HO
    from pytdscf.hamiltonian_cls import PolynomialHamiltonian
    from pytdscf.model_cls import BasInfo, Model
    from pytdscf.simulator_cls import Simulator

    prim_info = [
        [
            PrimBas_HO(0.0, 1500, 8),
            PrimBas_HO(0.0, 2000, 8),
            PrimBas_HO(0.0, 2500, 8),
        ]
    ]

    spf_info = [[5, 5, 5]]

    basinfo = BasInfo(prim_info, spf_info)

    hamiltonian = PolynomialHamiltonian(ndof=3)
    hamiltonian.set_HO_potential(basinfo)

    operators = {"hamiltonian": hamiltonian}

    model = Model(basinfo, operators)
    model.m_aux_max = 4

    jobname = "harmonic_fbr"
    simulator = Simulator(jobname, model, backend="numpy")
    simulator.propagate(maxstep=3)


if __name__ == "__main__":
    test_harmonic_fbr_mpsmctdh_propagate_numpy()
