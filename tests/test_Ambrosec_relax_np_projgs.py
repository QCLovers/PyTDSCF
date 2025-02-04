"""Test for Ambrosek 1D relaxation 2mol 4state 2mode"""

import itertools

import pytest

from pytdscf import units
from pytdscf.basis._primints_cls import PrimBas_HO
from pytdscf.hamiltonian_cls import PolynomialHamiltonian
from pytdscf.model_cls import BasInfo, Model
from pytdscf.simulator_cls import Simulator
from pytdscf.util.helper_input import matJ_1D_exciton

freqs_cm1 = [763.31, 1556.64]
disps = [0.317, 0.429]
nmol = 2
nspf = nprim = 5
s0 = [PrimBas_HO(0.0, freq, nprim) for freq in freqs_cm1]
s1 = [
    PrimBas_HO(disp, freq, nprim)
    for freq, disp in zip(freqs_cm1, disps, strict=False)
]
ener = [
    0.03929851595695371,
    0.010570469969995883
]


@pytest.mark.filterwarnings("ignore:DeprecationWarning")
@pytest.mark.parametrize(
    "coupleJ, bonddim, proj_gs, ener", [[-0.04, 5, True, ener[0]], [0.0, 4, False, ener[1]]]
)
def test_Ambrosec_relax_np_projgs(coupleJ, bonddim, proj_gs, ener):
    coupleJ /= units.au_in_eV
    deltaE = 0.0 / units.au_in_eV

    prim_info, spf_info, _, matJ = matJ_1D_exciton(
        nmol, nspf, s0, s1, coupleJ, deltaE=deltaE
    )
    my_basinfo = BasInfo(prim_info)

    my_hamiltonian = PolynomialHamiltonian(
        my_basinfo.get_ndof(), my_basinfo.get_nstate()
    )
    my_hamiltonian.coupleJ = matJ
    my_hamiltonian.set_HO_potential(my_basinfo)

    operators = {"hamiltonian": my_hamiltonian}
    model = Model(my_basinfo, operators)
    model.m_aux_max = bonddim
    model.ints_prim_file = None
    model.init_weight_ESTATE = [1.0] + [0.0] * (len(matJ) - 1)
    model.primbas_gs = list(
        itertools.chain.from_iterable([s0 for _ in range(nmol)])
    )

    jobname = "Ambrosek_test_projGS"
    simulator = Simulator(jobname, model, proj_gs=proj_gs, backend="numpy")
    ener_calc, wf = simulator.relax(maxstep=2, stepsize=0.05, improved=False)
    assert pytest.approx(ener_calc) == ener
