"""
Exponential DVR module
"""

import numpy as np

from pytdscf.basis.abc import DVRPrimitivesMixin


class Exponential(DVRPrimitivesMixin):
    r"""

    Exponential DVR is equivalent to FFT. It is suitable for periodic system.

    See also

    - `MCTDH review Phys.Rep. 324, 1 (2000) appendix B.4.5 <https://doi.org/10.1016/S0370-1573(99)00047-2>`_
    - `D.T. Colbert, W.H. Miller, J. Chem. Phys. 96 (1992) 1982. <https://doi.org/10.1063/1.462100>`_
    - `R. Meyer, J. Chem. Phys. 52 (1969) 2053. <https://doi.org/10.1063/1.1673259>`_

    Primitive Function :

    .. math::
       \varphi_{j}(x)= \frac{1}{\sqrt{L}} \exp \left(\mathrm{i} \frac{2 \pi j (x-x_0)}{L}\right)
       \quad \left(-\frac{N-1}{2} \leq j \leq \frac{N-1}{2}\right)

    They are periodic with period :math:`L=x_{N}-x_{0}`:

    .. math::
       \varphi_{j}(x+L)=\varphi_{j}(x)

    Note that :math:`N`, i.e., \
    ``ngrid`` must be odd number.

    .. note::
       Naturally, the DVR function :math:`\chi_\alpha(x)` is given by the multiplication of \
       delta function at equidistant grid \
       :math:`\in \{x_0, x_0+\frac{L}{N}, x_0+\frac{2L}{N}, \cdots, x_0+\frac{(N-1)L}{N}\}` \
       point and primitive function :math:`\varphi_{j}(x)` not \
       by the Unitary transformation of the primitive function :math:`\varphi_{j}(x)`. \
       (The operator :math:`\hat{z}` introduced in \
       `MCTDH review Phys.Rep. 324, 1 (2000) appendix B.4.5 <https://doi.org/10.1016/S0370-1573(99)00047-2>`_ \
       might be wrong.)

    Args:
        ngrid (int) : Number of grid. Must be odd number. (The center grid is give by index of ``n=ngrid//2``)
        length (float) : Length (unit depends on your potential. e.g. radian, bohr, etc.)
        x0 (float) : left terminal point. Basis functions are defined in :math:`[x_0, x_0+L)`.
        doAnalytical (bool) : If ``True``, use analytical formula. At the moment, \
            numerical integration is not implemented.

    """

    def __init__(
        self,
        ngrid: int,
        length: float,
        x0: float = 0.0,
        doAnalytical: bool = True,
    ):
        if ngrid % 2 == 0:
            raise ValueError("ngrid must be odd number.")
        super().__init__(ngrid)
        self.x0 = x0
        self.L = length
        self.lb = x0
        self.ub = x0 + self.L
        self.label = "Exponential"
        self.doAnalytical = doAnalytical
        if not doAnalytical:
            raise NotImplementedError(
                "Numerical Integral of complex exponential is somehow difficult."
            )
        self.deltax = self.L / self.ngrid  # Different from SineDVR

    def fbr_func(self, n: int, x: float) -> complex:
        r"""Primitive functions :math:`\varphi_n(x)`

        Args:
            n (int) : index ``(0 <= n < ngrid)``
            x (float) : position in a.u.

        Returns:
            complex : :math:`\frac{1}{\sqrt{L}} \exp \left(\mathrm{i} \frac{2 \pi n (x-x_0)}{L}\right)`

        j = 0, ±1, ±2, ..., ±(N-1)/2
        """
        j = n - self.ngrid // 2
        return np.exp(1j * 2.0 * np.pi * j * (x - self.x0) / self.L) / np.sqrt(
            self.L
        )

    def get_pos_rep_matrix(self) -> np.ndarray:
        r"""Exponential position matrix

        I think this method is not necessary.

        """
        raise NotImplementedError

    def get_1st_derivative_matrix_dvr(self) -> np.ndarray:
        r"""Exponential 1st derivative matrix given by analytical formula

        .. math::
           D_{\alpha \beta}^{(1), \mathrm{DVR}} =
           \begin{cases}
           0 & \text { if } \alpha=\beta \\
           \frac{\pi}{L} \frac{(-1)^{\alpha-\beta}}{\sin \left(\frac{\pi(\alpha-\beta)}{N}\right)}
           & \text { if } \alpha \neq \beta
           \end{cases}
           \quad (\alpha, \beta=0, \cdots, N-1)


        """
        if not hasattr(self, "first_derivative_matrix_dvr"):
            self.first_derivative_matrix_dvr = np.zeros(
                (self.ngrid, self.ngrid)
            )
            for alpha in range(self.ngrid - 1):
                for beta in range(alpha + 1, self.ngrid):
                    self.first_derivative_matrix_dvr[
                        alpha, beta
                    ] = self.first_derivative_matrix_dvr[beta, alpha] = (
                        np.pi
                        / self.L
                        * (-1) ** (alpha - beta)
                        / np.sin(np.pi * (alpha - beta) / self.ngrid)
                    )

        return self.first_derivative_matrix_dvr

    def get_1st_derivative_matrix_fbr(self) -> np.ndarray:
        if not hasattr(self, "first_derivative_matrix_fbr"):
            self.first_derivative_matrix_fbr = (
                self.get_unitary()
                @ self.get_1st_derivative_matrix_dvr()
                @ self.get_unitary().T
            )
        return self.first_derivative_matrix_fbr

    def get_2nd_derivative_matrix_dvr(self) -> np.ndarray:
        r"""Exponential 2nd derivative matrix given by analytical formula

        .. math::
           D_{\alpha \beta}^{(2), \mathrm{DVR}} =
           \begin{cases}
           -\frac{\pi^{2}}{3 L^{2}}\left(N^{2}-1\right)
           & \text { if } \alpha=\beta \\
           -\frac{2\pi^{2}}{L^{2}}
           (-1)^{\alpha-\beta}
           \frac{\cos\left(\frac{\pi(\alpha-\beta)}{N}\right)}
           {\sin ^{2}\left(\frac{\pi(\alpha-\beta)}{N}\right)}
           & \text { if } \alpha \neq \beta
           \end{cases}
           \quad (\alpha, \beta=0, \cdots, N-1)

        """
        if not hasattr(self, "second_derivative_matrix_dvr"):
            self.second_derivative_matrix_dvr = np.zeros(
                (self.ngrid, self.ngrid)
            )
            for alpha in range(self.ngrid):
                for beta in range(alpha, self.ngrid):
                    if alpha == beta:
                        self.second_derivative_matrix_dvr[alpha, beta] = (
                            -(np.pi**2) / 3.0 / self.L**2 * (self.ngrid**2 - 1)
                        )
                    else:
                        self.second_derivative_matrix_dvr[
                            alpha, beta
                        ] = self.second_derivative_matrix_dvr[beta, alpha] = (
                            -2.0
                            * np.pi**2
                            / self.L**2
                            * (-1) ** (alpha - beta)
                            * np.cos(np.pi * (alpha - beta) / self.ngrid)
                            / np.sin(np.pi * (alpha - beta) / self.ngrid) ** 2
                        )
        return self.second_derivative_matrix_dvr

    def get_2nd_derivative_matrix_fbr(self) -> np.ndarray:
        if not hasattr(self, "second_derivative_matrix_fbr"):
            self.second_derivative_matrix_fbr = (
                self.get_unitary()
                @ self.get_2nd_derivative_matrix_dvr()
                @ self.get_unitary().T
            )
        return self.second_derivative_matrix_fbr

    def diagnalize_pos_rep_matrix(self) -> None:
        r"""

        Not diagonalizing the position representation matrix \
        but manually setting the grid points and transformation matrix.

        .. math::
           \chi_\alpha(x) = \sum_{j=1}^N \varphi_j(x) U_{j\alpha}
           = \sum_{j=1}^N \langle\varphi_j(x_\alpha)|\varphi_j(x)\rangle \varphi_j(x)

        """
        if not hasattr(self, "grids"):
            self.grids = [
                self.x0 + alpha * self.deltax for alpha in range(self.ngrid)
            ]
            self.sqrt_weights = [
                np.sqrt(self.deltax) for _ in range(self.ngrid)
            ]
            self.unitary = np.zeros((self.ngrid, self.ngrid), dtype=complex)
            for alpha in range(self.ngrid):
                for j in range(self.ngrid):
                    self.unitary[j, alpha] = self.fbr_func(
                        j, self.grids[alpha]
                    ).conjugate()


if __name__ == "__main__":
    exp = Exponential(5, 5.0)
    print(exp.get_1st_derivative_matrix_dvr())
    print(exp.get_2nd_derivative_matrix_dvr())
    print(exp.get_unitary())
    print(exp.get_grids())
    exp.plot_fbr()
    exp.plot_dvr()
