"""
Sine DVR module
"""

import itertools
import warnings
from math import cos, pi, sin, sqrt

import numpy as np

from pytdscf import units as _units
from pytdscf.basis.abc import DVRPrimitivesMixin


class Sine(DVRPrimitivesMixin):
    r"""Sine DVR functions
    Note that Sine DVR position matrix is not tridiagonal!
    Index starts from j=1, alpha=1

    Terminal points (x_{0} and x_{N+1}) do not belog to the grid.

    See also  MCTDH review Phys.Rep. 324, 1 (2000) appendix B
    https://doi.org/10.1016/S0370-1573(99)00047-2

    Primitive Function :

    .. math::
       \varphi_{j}(x)= \begin{cases}\sqrt{2 / L} \sin \left(j \pi\left(x-x_{0}\right) / L\right) \
       & \text { for } x_{0} \leq x \leq x_{N+1} \\ 0 & \text { else }\end{cases} \quad (j=1,2,\ldots,N)


    Attributes:
        ngrid (int) : Number of grid. Note that which does not contain terminal point.
        nprim (int) : Number of primitive function. sama as ``ngrid``.
        length (float) : Length in a.u.
        x0 (float) : start point in a.u.
        doAnalytical (bool) : Use analytical integral or diagonalization.
        include_terminal (bool) : Whether include terminal grid.

    """

    def __init__(
        self,
        ngrid: int,
        length: float,
        x0: float = 0.0,
        units: str = "angstrom",
        doAnalytical: bool = True,
        include_terminal: bool = True,
    ):
        super().__init__(ngrid)

        if units.lower() in ["angstrom", "å"]:
            self.L = length / _units.au_in_angstrom
            self.x0 = x0 / _units.au_in_angstrom
        elif units.lower() in ["bohr", "a.u.", "au"]:
            self.L = length
            self.x0 = x0
        else:
            raise NotImplementedError

        if include_terminal:
            delta_x_tmp = self.L / (ngrid - 1)
            self.x0 -= delta_x_tmp
            self.L = (ngrid + 1) * delta_x_tmp

        self.lb = x0
        self.ub = x0 + self.L
        self.label = "Sine"
        self.doAnalytical = doAnalytical
        self.deltax = self.L / (self.ngrid + 1)

    def fbr_func(self, n: int, x: float):
        r"""Primitive functions :math:``\varphi_n(x)``

        Args:
            n (int) : index ``(0 <= n < ngrid)``

        Returns:
            float : :math:`\varphi_n(x)`

        """
        if 0 <= n < self.ngrid:
            _step1 = 1 * (self.x0 <= x)
            _step2 = 1 * (x <= self.x0 + self.L)
            return (
                sqrt(2 / self.L)
                * np.sin((n + 1) * pi * (x - self.x0) / self.L)
                * _step1
                * _step2
            )
        else:
            ValueError(f"n={n} must be in [0,ngrid)=[0,{self.ngrid})")

    def _transformed_var(self, x):
        r""":math:`z=\cos \left(\pi\left(x-x_{0}\right) / L\right)`"""
        return np.cos(pi * (x - self.x0) / self.L)

    def get_pos_rep_matrix(self):
        r"""Sine position matrix

        .. math::
           Q_{j k}=\left\langle\varphi_{j}|\hat{z}| \
            \varphi_{k}\right\rangle=\frac{1}{2}\left(\delta_{j, k+1}+\delta_{j, k-1}\right)

        where transformed variable

        .. math::
           z=\cos \left(\pi\left(x-x_{0}\right) / L\right)

        is introduced.
        """

        if self.doAnalytical:
            if not hasattr(self, "pos_rep_matrix"):
                semidiag = 0.5 * np.ones(self.ngrid - 1)
                self.pos_rep_matrix = np.diag(semidiag, k=1) + np.diag(
                    semidiag, k=-1
                )

            return self.pos_rep_matrix
        else:
            warnings.warn(
                "Sine DVR position operator is somehow different from straightforward integral."
                + "See https://doi.org/10.1016/S0370-1573(99)00047-2 and Set doAnalytical=True",
                stacklevel=2,
            )
            return super().get_pos_rep_matrix()

    def get_1st_derivative_matrix_fbr(self):
        r"""Analytical Formulation

        .. math::
           D_{j k}^{(1)}=\
           \left\langle\varphi_{j} |\frac{\mathrm{d}}{\mathrm{d} x} | \varphi_{k}\right\rangle={\rm mod} (j-k, 2) \frac{4}{L} \frac{j k}{j^{2}-k^{2}},\
           \quad j \neq k

        """
        if not hasattr(self, "first_derivative_matrix_fbr"):
            self.first_derivative_matrix_fbr = np.zeros(
                (self.ngrid, self.ngrid)
            )
            for j in range(self.ngrid):
                for k in range(j + 1, self.ngrid, 2):
                    self.first_derivative_matrix_fbr[j, k] = (
                        4 / self.L * (j + 1) * (k + 1) / ((j - k) / (j + k + 2))
                    )
                    self.first_derivative_matrix_fbr[
                        k, j
                    ] = -self.first_derivative_matrix_fbr[j, k]

        return self.first_derivative_matrix_fbr

    def get_1st_derivative_matrix_dvr(self):
        return super().get_1st_derivative_matrix_dvr()

    def get_2nd_derivative_matrix_fbr(self):
        r"""Analytical Formulation (start from j=1 !)

        .. math::
           D_{j k}^{(1)}=\
           \left\langle\varphi_{j} |\frac{\mathrm{d}}{\mathrm{d} x} | \varphi_{k}\right\rangle={\rm mod} (j-k, 2) \frac{4}{L} \frac{j k}{j^{2}-k^{2}},\
           \quad j \neq k

        """
        if not hasattr(self, "second_derivative_matrix_fbr"):
            self.second_derivative_matrix_fbr = -(
                (pi / self.L * np.diag(np.arange(1, self.ngrid + 1))) ** 2
            )

        return self.second_derivative_matrix_fbr

    def get_2nd_derivative_matrix_dvr(self):
        r"""Analytical forumulation exists.

        .. math ::
           D_{\alpha \beta}^{(2), \mathrm{DVR}}=-\left(\frac{\pi}{\Delta x}\right)^{2}\left\{\begin{array}{l}
           -\frac{1}{3}+\frac{1}{6(N+1)^{2}}-\frac{1}{2(N+1)^{2} \sin ^{2}\left(\frac{\alpha \pi}{N+1}\right)}, \quad \alpha=\beta \\
           \frac{2(-1)^{\alpha-\beta}}{(N+1)^{2}} \frac{\sin \left(\frac{\alpha \pi}{N+1}\right) \sin \left(\frac{\beta \pi}{N+1}\right)}{\left(\cos \left(\frac{\alpha \pi}{N+1}\right)-\cos \left(\frac{\beta \pi}{N+1}\right)\right)^{2}},\quad \alpha \neq \beta
           \end{array} \right.

        """
        if self.doAnalytical:
            if not hasattr(self, "second_derivative_matrix_dvr"):
                self.second_derivative_matrix_dvr = np.zeros(
                    (self.ngrid, self.ngrid)
                )
                _N = self.ngrid + 1
                for alpha, beta in itertools.product(
                    range(1, self.ngrid + 1), repeat=2
                ):
                    ap_N = alpha * pi / _N
                    bp_N = beta * pi / _N
                    if alpha == beta:
                        # Beck's 2000 thesis analytical form is wrong. (sign of first term)
                        val = (
                            +1 / 3
                            + 1 / (6 * (_N**2))
                            - 1 / (2 * ((_N * sin(ap_N)) ** 2))
                        )
                    else:
                        val = (
                            2
                            * ((-1) ** (alpha - beta))
                            / (_N**2)
                            * sin(ap_N)
                            * sin(bp_N)
                            / (cos(ap_N) - cos(bp_N)) ** 2
                        )
                    self.second_derivative_matrix_dvr[alpha - 1, beta - 1] = val
                coef = -((pi / self.deltax) ** 2)
                self.second_derivative_matrix_dvr *= coef
            return self.second_derivative_matrix_dvr
        else:
            return super().get_2nd_derivative_matrix_dvr()

    def diagnalize_pos_rep_matrix(self):
        r"""
        Analytical diagonalization has been derived

        .. math::
           U_{j \alpha}=\sqrt{\frac{2}{N+1}} \sin \left(\frac{j \alpha \pi}{N+1}\right)

        .. math::
           z_{\alpha}=\cos \left(\frac{\alpha \pi}{N+1}\right)

        This leads to the DVR grid points

        .. math::
           x_{\alpha}=x_{0}+\frac{L}{\pi} \arccos \left(z_{\alpha}\right)\
            =x_{0}+\alpha \frac{L}{N+1}=x_{0}+\alpha \Delta x

        """
        if not hasattr(self, "grids"):
            if self.doAnalytical:
                self.unitary = np.zeros((self.ngrid, self.ngrid))
                for j in range(self.ngrid):
                    for alpha in range(self.ngrid):
                        self.unitary[j, alpha] = sin(
                            (j + 1) * (alpha + 1) * pi / (self.ngrid + 1)
                        )
                        self.unitary[alpha, j] = self.unitary[j, alpha]
                self.unitary *= sqrt(2 / (self.ngrid + 1))

                self.grids = [
                    self.x0 + alpha * self.deltax
                    for alpha in range(1, self.ngrid + 1)
                ]
                self.sqrt_weights = [
                    sqrt(self.deltax) for _ in range(self.ngrid)
                ]
            else:
                warnings.warn(
                    "Sine DVR position operator is somehow different from straightforward integral."
                    + "See https://doi.org/10.1016/S0370-1573(99)00047-2 and Set doAnalytical=True",
                    stacklevel=2,
                )
                super().diagnalize_pos_rep_matrix()
