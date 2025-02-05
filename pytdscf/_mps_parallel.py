"""MPO-based real-space parallelMPS class"""

from pytdscf._mps_mpo import MPSCoefMPO
from pytdscf._spf_cls import SPFInts
from pytdscf.hamiltonian_cls import TensorHamiltonian


class MPSCoefParallel(MPSCoefMPO):
    """Parallel MPS Coefficient Class"""

    def propagate(
        self, stepsize: float, ints_spf: SPFInts | None, matH: TensorHamiltonian
    ):
        super().propagate(stepsize, ints_spf, matH)
