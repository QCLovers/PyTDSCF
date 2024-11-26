"""
Caution: The DVR implementation of this module is deprecated.
Use the `Discvar` library instead.
"""

from .exponential import Exponential
from .ho import HarmonicOscillator, PrimBas_HO
from .sin import Sine
from .exciton import Exciton

__all__ = ["Exponential", "HarmonicOscillator", "PrimBas_HO", "Sine", "Exciton"]
