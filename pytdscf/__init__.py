from jax import config

from . import (
    dvr_operator_cls,
    hamiltonian_cls,
    property,
    spectra,
    units,
    util,
    wavefunction,
)
from .__version__ import __version__
from .basis import Exponential, HarmonicOscillator, PrimBas_HO, Sine
from .model_cls import BasInfo, Model
from .simulator_cls import Simulator

# See also JAX's GitHub https://github.com/google/jax#current-gotchas
config.update("jax_enable_x64", True)

__all__ = [
    "__version__",
    "dvr_operator_cls",
    "hamiltonian_cls",
    "property",
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
    "PrimBas_HO",
]
