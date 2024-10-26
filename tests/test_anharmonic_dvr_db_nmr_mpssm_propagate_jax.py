"""Test for Harmonic Potential 3-mode MPS-SM (DVR) propagation"""

import os

import numpy as np
import pytest


@pytest.mark.filterwarnings("ignore:DeprecationWarning")
def test_anharmonic_dvr_func_nmr_mpssm_propagate_jax():
    # from pytdscf.basis import HarmonicOscillator
    from discvar import HarmonicOscillator

    from pytdscf.dvr_operator_cls import (
        TensorOperator,
        construct_kinetic_operator,
        construct_nMR_recursive,
    )
    from pytdscf.hamiltonian_cls import TensorHamiltonian
    from pytdscf.model_cls import BasInfo, Model
    from pytdscf.simulator_cls import Simulator

    prim_info = [
        [
            HarmonicOscillator(5, 1658.835, 0.0),
            HarmonicOscillator(5, 3750.332, 0.0),
            HarmonicOscillator(5, 3851.945, 0.0),
        ]
    ]

    basinfo = BasInfo(prim_info)

    mpo = construct_nMR_recursive(
        dvr_prims=prim_info[0],
        nMR=2,
        db=f"{os.path.dirname(os.path.abspath(__file__))}/../pytdscf/potentials/h2o_5grids.db",
        rate=0.999999999,
    )
    dum = None
    J = [2, 2, 2]
    for j, core in zip(J, mpo, strict=False):
        assert core.shape[1] == 5
        if dum is None:
            dum = core[:, j, :]
        else:
            dum = np.einsum("ij,jk->ik", dum, core[:, j, :])
    assert np.linalg.norm(dum) < 1.0e-03

    potential = [[{(0, 1, 2): TensorOperator(mpo=mpo)}]]

    kinetic = [[construct_kinetic_operator(dvr_prims=prim_info[0])]]
    hamiltonian = TensorHamiltonian(
        ndof=3, potential=potential, kinetic=kinetic, backend="jax"
    )

    operators = {"hamiltonian": hamiltonian}

    model = Model(basinfo, operators)
    model.m_aux_max = 4

    jobname = "anharmonic_dvr_nMR"
    simulator = Simulator(jobname, model, backend="jax")
    simulator.propagate(stepsize=0.1, maxstep=3)


if __name__ == "__main__":
    test_anharmonic_dvr_func_nmr_mpssm_propagate_jax()
