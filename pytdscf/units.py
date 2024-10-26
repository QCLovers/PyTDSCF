"""
Unit Convert Class

xxx_in_yyy = z

means

1 [xxx] = z [yyy]

Example:
    >>> from pytdscf.units import au_in_angstrom
    >>> print(au_in_angstrom)
    0.529177210903

Units is based on SciPy constants (CODATA 2018)

Attributes:
    au_in_cm1 (float): 1 a.u. in cm^-1
    au_in_fs (float): 1 a.u. in fs
    au_in_eV (float): 1 a.u. in eV
    au_in_dalton (float): 1 a.u. in Dalton
    au_in_angstrom (float): 1 a.u. in Angstrom
    au_in_debye (float): 1 a.u. in Debye

"""

from scipy.constants import physical_constants as pc

# au_in_cm1 = 2.1947463136320 * 1.0e+5 #
au_in_cm1 = (
    pc["atomic unit of energy"][0]
    / (pc["speed of light in vacuum"][0] * 1.0e02)
    / pc["Planck constant"][0]
)
Hartree_in_cm1 = au_in_cm1
# au_in_fs  = 2.4188843265857 * 1.0e-17 / 1.0e-15
# https://physics.nist.gov/cgi-bin/cuu/Value?aut
au_in_fs = pc["atomic unit of time"][0] / 1.0e-15
# au_in_eV  = 27.211386245988
# https://physics.nist.gov/cgi-bin/cuu/Value?hrev
au_in_eV = pc["Hartree energy in eV"][0]
Has_in_eV = au_in_eV
# au_in_dalton = 1 / 1822.888486209
au_in_dalton = pc["electron mass"][0] / pc["atomic mass constant"][0]
au_in_AMU = au_in_dalton
# au_in_angstrom = 0.529177210903
au_in_angstrom = pc["Bohr radius"][0] / 1.0e-10
Bohr_in_angstrom = au_in_angstrom
# 1 Debye = 1 / (speed of light * 10^21) C.m
# 1 a.u. = electron chage * bohr radius C.m
au_in_debye = (
    pc["atomic unit of electric dipole mom."][0]
    * pc["speed of light in vacuum"][0]
    * 1.0e21
)

if __name__ == "__main__":
    print(f"au_in_cm1 = {au_in_cm1}")
    print(f"au_in_fs = {au_in_fs}")
    print(f"au_in_eV = {au_in_eV}")
    print(f"au_in_dalton = {au_in_dalton}")
    print(f"au_in_angstrom = {au_in_angstrom}")
    print(f"au_in_debye = {au_in_debye}")
