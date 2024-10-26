"""Test for MPS sampling based on Coherent state overlap for TensorFlow DVR"""

import jax.numpy as jnp
import pytest
from discvar import HarmonicOscillator

from pytdscf import BasInfo, Model, Simulator
from pytdscf.dvr_operator_cls import TensorOperator, construct_nMR_recursive
from pytdscf.hamiltonian_cls import TensorHamiltonian


@pytest.mark.filterwarnings("ignore:DeprecationWarning")
@pytest.mark.parametrize("p, q", [[0.1, 0.1]])
def test_sample_CS_ovlp_jax(p: float, q: float):
    # restart=false means apply dipole to HO ground state

    prim_info = [
        [
            HarmonicOscillator(5, 1500, 0.0),
            HarmonicOscillator(5, 2000, 0.0),
            HarmonicOscillator(5, 2500, 0.0),
        ]
    ]

    basinfo = BasInfo(prim_info)

    dipole_funcs = {}
    dipole_funcs[(0,)] = lambda Q_0: 0.1 * Q_0
    dipole_funcs[(1,)] = lambda Q_1: 0.1 * Q_1
    dipole_funcs[(2,)] = lambda Q_2: 0.1 * Q_2

    mpo = construct_nMR_recursive(dvr_prims=prim_info[0], func=dipole_funcs)
    tensors = [[{(0, 1, 2): TensorOperator(mpo=mpo)}]]
    dipole = TensorHamiltonian(
        ndof=3,
        potential=tensors,
        kinetic=None,
        decompose_type="SVD",
        rate=0.9999999,
        backend="jax",
    )

    operators = {"hamiltonian": dipole}

    model = Model(basinfo, operators)
    model.m_aux_max = 4

    jobname = "coherent_sample_DVR"
    simulator = Simulator(jobname, model, backend="jax")
    norm, wf_applied_dipole = simulator.operate(maxstep=10, restart=False)

    # MPS is TensorFlow.Tensor, hence we should convert np.ndarray to Tensor in advance
    trans_arrays = [
        jnp.array(HO.get_ovi_CS_HO(p, q, type="DVR"), dtype=jnp.complex128)
        for HO in prim_info[0]
    ]
    contracted_value = wf_applied_dipole.ci_coef.get_CI_coef_state(
        trans_arrays=trans_arrays
    )
    print(contracted_value)


if __name__ == "__main__":
    test_sample_CS_ovlp_jax(p=0.1, q=0.1)
