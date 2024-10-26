"""Test for Harmonic Potential 3-mode MCTDH propagation"""

import pytest


@pytest.mark.filterwarnings("ignore:DeprecationWarning")
def test_harmonic_fbr_sm_propagate_numpy():
    from pytdscf.basis._primints_cls import PrimBas_HO
    from pytdscf.hamiltonian_cls import PolynomialHamiltonian
    from pytdscf.model_cls import BasInfo, Model
    from pytdscf.simulator_cls import Simulator

    prim_info = [[PrimBas_HO(0.0, 1500, 8), PrimBas_HO(0.0, 2000, 8)]]

    basinfo = BasInfo(prim_info)

    hamiltonian = PolynomialHamiltonian(ndof=2)
    hamiltonian.set_HO_potential(basinfo)

    operators = {"hamiltonian": hamiltonian}

    model = Model(basinfo, operators)

    jobname = "harmonic_fbr_sm"
    simulator = Simulator(
        jobname, model, ci_type="standard-method", backend="numpy"
    )
    simulator.propagate(maxstep=1)


if __name__ == "__main__":
    test_harmonic_fbr_sm_propagate_numpy()
