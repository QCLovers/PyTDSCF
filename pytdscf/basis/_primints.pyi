# C++ bindings for the primints module
from __future__ import annotations

def ovi_HO_FBR_cpp(
    v0: int,
    v1: int,
    freq_cm1_bra: float,
    freq_cm1_ket: float,
    origin_bra: float,
    origin_ket: float,
) -> float: ...
def poly_HO_FBR_cpp(
    v0: int,
    v1: int,
    freq_cm1_bra: float,
    freq_cm1_ket: float,
    origin_bra: float,
    origin_ket: float,
    norder: int,
) -> float: ...
