from jax import config

from . import (
    dvr_operator_cls,
    hamiltonian_cls,
    properties,
    spectra,
    units,
    util,
    wavefunction,
)
from .__version__ import __version__
from .basis import Exciton, Exponential, HarmonicOscillator, PrimBas_HO, Sine
from .dvr_operator_cls import (
    construct_fulldimensional,
    construct_kinetic_operator,
    construct_nMR_recursive,
)
from .hamiltonian_cls import (
    TensorHamiltonian,
    TensorOperator,
    read_potential_nMR,
)
from .model_cls import BasInfo, Model
from .simulator_cls import Simulator

# See also JAX's GitHub https://github.com/google/jax#current-gotchas
config.update("jax_enable_x64", True)

__all__ = [
    "__version__",
    "dvr_operator_cls",
    "hamiltonian_cls",
    "properties",
    "spectra",
    "units",
    "util",
    "wavefunction",
    "BasInfo",
    "Model",
    "Simulator",
    "HarmonicOscillator",
    "Sine",
    "Exponential",
    "Exciton",
    "PrimBas_HO",
    "TensorHamiltonian",
    "TensorOperator",
    "construct_kinetic_operator",
    "construct_nMR_recursive",
    "construct_fulldimensional",
    "read_potential_nMR",
]
