"""Test for Harmonic Potential 3-mode MPS-SM (DVR) propagation"""

import os

import numpy as np
import pytest


def test_anharmonic_dvr_func_nmr_mpssm_propagate_jax():
    # from pytdscf.basis import HarmonicOscillator
    from discvar import HarmonicOscillator

    from pytdscf.dvr_operator_cls import (
        construct_kinetic_mpo,
        construct_nMR_recursive,
    )
    from pytdscf.model_cls import Model
    from pytdscf.simulator_cls import Simulator

    prim_info = [
        HarmonicOscillator(5, 1658.835, 0.0),
        HarmonicOscillator(5, 3750.332, 0.0),
        HarmonicOscillator(5, 3851.945, 0.0),
    ]

    mpo = construct_nMR_recursive(
        dvr_prims=prim_info,
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

    kin_mpo = construct_kinetic_mpo(dvr_prims=prim_info)

    operators = {"potential": mpo, "kinetic": kin_mpo}

    model = Model(prim_info, operators, bond_dim=4)

    jobname = "anharmonic_dvr_nMR"
    simulator = Simulator(jobname, model, backend="jax")
    ener_calc, wf = simulator.propagate(
        stepsize=0.1, maxstep=10, autocorr=True, autocorr_per_step=1
    )
    assert pytest.approx(ener_calc) == 0.010549182771706139


if __name__ == "__main__":
    test_anharmonic_dvr_func_nmr_mpssm_propagate_jax()
