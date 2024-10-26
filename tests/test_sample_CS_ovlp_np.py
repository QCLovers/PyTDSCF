"""Test for MPS sampling based on Coherent state overlap for Numpy FBR"""

import pytest

from pytdscf import BasInfo, Model, Simulator
from pytdscf.basis import PrimBas_HO
from pytdscf.hamiltonian_cls import read_potential_nMR


@pytest.mark.filterwarnings("ignore:DeprecationWarning")
@pytest.mark.parametrize("p, q", [[0.1, 0.1]])
def test_sample_CS_ovlp_np(p: float, q: float):
    # restart=false means apply dipole to HO ground state

    prim_info = [
        [
            PrimBas_HO(0.0, 1500, 5),
            PrimBas_HO(0.0, 2000, 5),
            PrimBas_HO(0.0, 2500, 5),
        ]
    ]

    basinfo = BasInfo(prim_info)

    mu = {
        (0,): [1 / 30, 1 / 30, 1 / 30],
        (1,): [1 / 30, 1 / 30, 1 / 30],
        (2,): [1 / 30, 1 / 30, 1 / 30],
    }
    dipole = read_potential_nMR(potential_emu=None, dipole_emu=mu)
    operators = {"hamiltonian": dipole}

    model = Model(basinfo, operators)
    model.m_aux_max = 4

    jobname = "coherent_sample_FBR"
    simulator = Simulator(jobname, model, backend="numpy")
    norm, wf_applied_dipole = simulator.operate(maxstep=10, restart=False)

    trans_arrays = [
        HO.todvr().get_ovi_CS_HO(p, q, type="FBR") for HO in prim_info[0]
    ]
    contracted_value = wf_applied_dipole.ci_coef.get_CI_coef_state(
        trans_arrays=trans_arrays
    )
    print(contracted_value)


if __name__ == "__main__":
    test_sample_CS_ovlp_np(p=0.1, q=0.1)
