"""Test for Linear Vibronic Coupling model"""

import pytest

from pytdscf.basis._primints_cls import PrimBas_HO
from pytdscf.hamiltonian_cls import PolynomialHamiltonian
from pytdscf.model_cls import BasInfo, Model
from pytdscf.simulator_cls import Simulator

freqs_cm1 = [1000, 2000, 3000]
disps = [0.3, 0.4, 0.5]
nspf = nprim = 5
s0 = [PrimBas_HO(0.0, freq, nprim) for freq in freqs_cm1]
s1 = [
    PrimBas_HO(disp, freq, nprim)
    for freq, disp in zip(freqs_cm1, disps, strict=False)
]
prim_info = [s0, s1]


@pytest.mark.filterwarnings("ignore:DeprecationWarning")
@pytest.mark.parametrize(
    "coupleJ, bonddim, proj_gs, ener",
    [
        [-0.04, 5, True, 0.013669005758718421],
        [0.0, 4, False, 0.013669005758738601],
    ],
)
def test_LVC_propagate_np(coupleJ, bonddim, proj_gs, ener):
    deltaE = 0.007

    basinfo = BasInfo(prim_info)
    hamiltonian = PolynomialHamiltonian(
        basinfo.get_ndof(), basinfo.get_nstate()
    )
    hamiltonian.coupleJ = [[0, coupleJ], [coupleJ, deltaE]]

    lam = {
        (0, 1): {0: 0.002, 1: 0.002, 2: 0.002},
        (1, 0): {0: 0.002, 1: 0.002, 2: 0.002},
    }
    hamiltonian.set_LVC(basinfo, lam)

    operators = {"hamiltonian": hamiltonian}
    model = Model(basinfo, operators)
    model.m_aux_max = bonddim
    model.ints_prim_file = None
    model.init_weight_ESTATE = [1.0, 0.0]
    model.init_weight_VIB_GS = 1.0
    model.primbas_gs = s0

    jobname = "LVC_test"
    simulator = Simulator(jobname, model, proj_gs=proj_gs, backend="numpy")
    ener_calc, wf = simulator.propagate(maxstep=3, stepsize=0.05)
    assert pytest.approx(ener_calc) == ener


if __name__ == "__main__":
    test_LVC_propagate_np(-0.04, 5, True, 0.013669005758718421)
