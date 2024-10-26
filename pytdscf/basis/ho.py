"""
The Harmonic Oscillator (HO) primitive basis functions and DVR functions.
"""

import math
from typing import Optional

import numpy as np
from scipy.special import hermite

from pytdscf import units as _units
from pytdscf.basis.abc import DVRPrimitivesMixin

FACTORIAL_SQRT_INV = [0.0] * 25
FACTORIAL_SQRT_INV[0] = 1.0
for i in range(1, 25):
    FACTORIAL_SQRT_INV[i] = FACTORIAL_SQRT_INV[i - 1] / math.sqrt(i)


class HarmonicOscillator(DVRPrimitivesMixin):
    r"""Harmonic Oscillator DVR functions

    See also  MCTDH review Phys.Rep. 324, 1 (2000) appendix B
    https://doi.org/10.1016/S0370-1573(99)00047-2

    Normalization factor :

    .. math::
       A_n = \frac{1}{\sqrt{n! 2^n}} \left(\frac{m\omega}{\pi\hbar}\right)^{\frac{1}{4}}\
           \xrightarrow[\rm impl]{} \frac{1}{\sqrt{n! 2^n}} \left(\frac{\omega_i}{\pi}\right)^{\frac{1}{4}}


    Dimensionless coordinate :

    .. math::
       \zeta = \sqrt{\frac{m\omega} {\hbar}}(x-a) \xrightarrow[\rm impl]{} \sqrt{\omega_i}q_i

    Primitive Function :

    .. math::
       \varphi_n = A_n H_n(\zeta) \exp\left(- \frac{\zeta^2}{2}\right)\quad (n=0,2,\ldots,N-1)

    Attributes:
        ngrid (int) : # of grid
        nprim (int) : # of primitive function. same as ``ngrid``
        omega (float) : frequency in a.u.
        q_eq (float) : eq_point in mass-weighted coordinate
        dimensionless (bool) : Input is dimensionless coordinate or not.
        doAnalytical (bool) : Use analytical integral or diagonalization.

    """

    def __init__(
        self,
        ngrid: int,
        omega: float,
        q_eq: Optional[float] = 0.0,
        units: Optional[str] = "cm-1",
        dimnsionless: Optional[bool] = False,
        doAnalytical: Optional[bool] = True,
    ):
        super().__init__(ngrid)

        if units.lower() in ["cm1", "cm-1", "kaiser"]:
            self.omega = omega / _units.au_in_cm1
            self.freq_cm1 = omega
        elif units.lower() in ["au", "hartree", "a.u."]:
            self.omega = omega
            self.freq_cm1 = omega * _units.au_in_cm1
        elif units.lower() == "ev":
            self.omega = omega / _units.au_in_eV
            self.freq_cm1 = omega / _units.au_in_eV * _units.au_in_cm1
        else:
            raise ValueError(f"{units} must be [cm1, au, eV]")

        self.q_eq = q_eq
        if dimnsionless:
            self.q_eq /= math.sqrt(self.omega)
        self.origin = q_eq

        self.sigma = 1 / math.sqrt(self.omega)
        self.lb = -ngrid * self.sigma
        self.ub = ngrid * self.sigma
        self.label = "HO"
        self.doAnalytical = doAnalytical

    def fbr_func(self, n: int, q):
        r"""Primitive functions :math:``\varphi_n(q)``

        Args:
            n (int) : index ``(0 <= n < ngrid)``

        Returns:
            float : :math:`\varphi_n(q)`

        """
        if 0 <= n < self.ngrid:
            return (
                self._normal_factor(n)
                * self._hermite_pol(n, q)
                * self._gaussian(q)
            )
        else:
            ValueError(f"n={n} must be in [0,ngrid)=[0,{self.ngrid})")

    def get_pos_rep_matrix(self):
        r"""HO has analytical formulation

        .. math::
           \left\langle\varphi_{j}|\hat{x}| \varphi_{k}\right\rangle=\
           \sqrt{\frac{j+1}{2 m \omega}} \delta_{j, k-1}+x_{\mathrm{eq}}\
           \delta_{j k}+\sqrt{\frac{j}{2 m \omega}} \delta_{j, k+1}

        """
        if self.doAnalytical:
            if not hasattr(self, "pos_rep_matrix"):
                diag = np.ones(self.ngrid, dtype=complex) * self.q_eq
                semidiag = np.sqrt(
                    np.arange(1, self.ngrid) / 2 / self.omega, dtype=complex
                )
                self.pos_rep_matrix = (
                    np.diag(diag)
                    + np.diag(semidiag, k=1)
                    + np.diag(semidiag, k=-1)
                )
            return self.pos_rep_matrix
        else:
            return super().get_pos_rep_matrix()

    def get_1st_derivative_matrix_fbr(self):
        r"""Analytical Formulation

        .. math::
           D_{j k}^{(1)}=\
           \left\langle\varphi_{j} | \frac{\mathrm{d}}{\mathrm{d} x} | \varphi_{k}\right\rangle=\
           -\sqrt{\frac{m \omega}{2}}\left(\sqrt{j+1} \delta_{j, k-1}-\sqrt{j} \delta_{j, k+1}\right)

        """
        if not hasattr(self, "first_derivative_matrix_fbr"):
            semidiag = -np.sqrt(self.omega * np.arange(1, self.ngrid) / 2)
            self.first_derivative_matrix_fbr = np.diag(semidiag, k=1) - np.diag(
                semidiag, k=-1
            )

        return self.first_derivative_matrix_fbr

    def get_1st_derivative_matrix_dvr(self):
        return super().get_1st_derivative_matrix_dvr()

    def get_2nd_derivative_matrix_fbr(self):
        r"""Analytical Formulation

        .. math::
           D_{j k}^{(2)}=\
           \frac{m\omega}{2}\left(\sqrt{(j-1)j}\delta_{j,k+2}-(2j+1)\delta_{j,k} + \sqrt{(j+2)(j+1)}\delta_{j,k-2}\right)
        """
        if not hasattr(self, "second_derivative_matrix_fbr"):
            diag = -self.omega / 2 * (2 * np.arange(self.ngrid) + 1)
            semisemidiag = (
                self.omega
                / 2
                * np.sqrt(
                    np.arange(1, self.ngrid - 1) * np.arange(2, self.ngrid)
                )
            )
            self.second_derivative_matrix_fbr = (
                np.diag(diag)
                + np.diag(semisemidiag, k=2)
                + np.diag(semisemidiag, k=-2)
            )

        return self.second_derivative_matrix_fbr

    def get_2nd_derivative_matrix_dvr(self):
        return super().get_2nd_derivative_matrix_dvr()

    def diagnalize_pos_rep_matrix(self):
        """Analytical formulation has not yet derived."""
        return super().diagnalize_pos_rep_matrix()

    def get_ovi_CS_HO(
        self, p: float, q: float, type: str = "DVR"
    ) -> np.ndarray:
        """Get overlap integral 1D-array between coherent state <p,q,ω,| and |HO(ω)>

        Args:
            p (float) : momentum of coherent state in mass-weighted a.u.
            q (float) : position of coherent state in mass-weighted a.u.
            type (str) : Whether "DVR" or "FBR". Default is "DVR".

        Returns:
            np.ndarray : overlap integral 1D-array between coherent state <p,q,ω,| and |HO(ω)>

        """
        z = math.sqrt(self.omega * 0.5) * (q + 1j * p / self.omega)
        expo = math.exp(-0.5 * abs(z) ** 2)
        zp = 1.0 + 0.0j
        ints = np.zeros(self.nprim, dtype=complex)
        for v in range(self.nprim):
            ints[v] = FACTORIAL_SQRT_INV[v] * zp * expo
            zp *= z
        if type == "DVR":
            return np.conjugate(self.get_unitary().T) @ ints
        elif type == "FBR":
            return ints
        else:
            raise ValueError(
                f"type argument must be 'DVR' or 'FBR', not {type}"
            )

    def _normal_factor(self, n: int) -> float:
        r"""

        .. math::
           A_n = \frac{1}{\sqrt{n! 2^n}} \left(\frac{\omega_i}{\pi}\right)^{\frac{1}{4}}

        Args:
            n (int) : index ``(0 <= n < ngrid)``

        Returns:
            float : :math:`A_n`
        """
        if 0 <= n < self.ngrid:
            return (
                1
                / math.sqrt(math.factorial(n) * pow(2, n))
                * pow(self.omega / math.pi, 0.25)
            )
        else:
            raise ValueError(f"{n} is not in [0,{self.ngrid})")

    def _hermite_pol(self, n: int, q):
        """Hermite Polynomial

        Args:
            n (int) : index ``(0 <= n < ngrid)``
            q (_ArrayLike0D) : mass-weighted coordinate array

        Returns:
            _ArrayLike0D : :math:`H_n(q)`

        """
        q = q - self.q_eq
        if 0 <= n < self.ngrid:
            return hermite(n)(math.sqrt(self.omega) * q)
        else:
            raise ValueError(f"{n} is not in [0,{self.ngrid})")

    def _gaussian(self, q):
        r"""Gaussian :math:`\exp\left(- \frac{\omega q^2}{2}\right)`"""
        q = q - self.q_eq
        return np.exp(-self.omega * q * q / 2)


class PrimBas_HO:
    r""" The Harmonic Oscillator eigenfunction primitive basis.

    This class holds information on Gauss Hermite type SPF basis functions.
    The `n`-th primitive represents

    Normalization factor :

    .. math::
       A_n = \frac{1}{\sqrt{n! 2^n}} \left(\frac{m\omega}{\pi\hbar}\right)^{\frac{1}{4}}

    Dimensionless coordinate :

    .. math::
       \zeta = \sqrt{\frac{m\omega} {\hbar}}(x-a)

    Primitive Function :

    .. math::
       \chi_n = A_n H_n(\zeta) \exp\left(- \frac{\zeta^2}{2}\right)

    Args:
        origin (float) : The center (equilibrium) dimensionless coordinate \
                :math:`\sqrt{\frac{m\omega}{\hbar}} a` of Hermite polynomial.
        freq_cm1 (float) : The frequency :math:`\omega` of Hermite polynomial. \
                The unit is cm-1.
        nprim (int) : The number (max order) of primitive basis on a certain SPF.
        origin_is_dimless (bool, optional) : If True, given ``self.origin`` is \
                dimensionless coordinate. Defaults to True.

    Attributes:
        origin (float) : The center (equilibrium) dimensionless coordinate \
                :math:`\sqrt{\frac{m\omega}{\hbar}} a` of Hermite polynomial.
        freq_cm1 (float) : The frequency :math:`\omega` of Hermite polynomial. \
                The unit is cm-1.
        nprim (int) : The number (max order) of primitive basis on a certain SPF.
        freq_au (float) : ``self.freq_cm1`` in a.u.
        origin_mwc (float) : Mass weighted coordinate ``self.origin`` in a.u.

    """

    def __init__(
        self,
        origin: float,
        freq_cm1: float,
        nprim: int,
        origin_is_dimless: bool = True,
    ):
        self.freq_cm1 = freq_cm1
        self.nprim = nprim
        self.freq_au = freq_cm1 / _units.au_in_cm1
        if origin_is_dimless:
            self.origin_mwc = origin / math.sqrt(self.freq_au)
            self.origin = origin
        else:
            """origin is mass-weghted"""
            self.origin_mwc = origin
            self.origin = origin * math.sqrt(self.freq_au)

    def __len__(self) -> int:
        return self.nprim

    def todvr(self):
        return HarmonicOscillator(
            ngrid=self.nprim, omega=self.freq_cm1, q_eq=self.origin
        )
