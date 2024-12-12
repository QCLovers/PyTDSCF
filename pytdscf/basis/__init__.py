"""
Caution: The DVR implementation of this module is deprecated.
Use the `Discvar` library instead.
"""

from .boson import Boson
from .exciton import Exciton
from .exponential import Exponential
from .ho import HarmonicOscillator, PrimBas_HO
from .sin import Sine

__all__ = [
    "Boson",
    "Exponential",
    "HarmonicOscillator",
    "PrimBas_HO",
    "Sine",
    "Exciton",
]
