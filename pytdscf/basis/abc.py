"""
Abstract DVR primitive class module
"""

import logging
from abc import ABC, abstractmethod
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.linalg import eigh

logger = logging.getLogger(__name__)


class DVRPrimitivesMixin(ABC):
    """
    Abstract DVR primitive function class
    """

    def __init__(self, ngrid: int):
        logger.warning(
            "Built-In DVR will be deprecated in the near future. Use Discvar instead."
        )
        if type(ngrid) is not int:
            raise TypeError(
                "ngrid argument must be integer but {ngrid} is given."
            )
        self.ngrid = ngrid
        self.nprim = ngrid

    def __call__(self, n: int, q: float) -> float or complex:
        return self.dvr_func(n, q)

    def __iter__(self):
        for grid in self.get_grids():
            yield grid

    def __len__(self) -> int:
        return self.ngrid

    @abstractmethod
    def fbr_func(self, n: int, q: float) -> float or complex:
        r"""fbrinal Primitive function :math:`\varphi_n(q)`, such as HO, Sine, etc"""
        pass

    @abstractmethod
    def get_pos_rep_matrix(self) -> np.ndarray:
        r"""Numerical integral of :math:`\langle\varphi_j|\hat{q}|\varphi_k\rangle`

        If analytical integral is known, implemented in inheritance class

        """
        if not hasattr(self, "pos_rep_matrix"):
            self.pos_rep_matrix = np.zeros(
                (self.ngrid, self.ngrid), dtype=complex
            )
            avg_error = 0.0
            for j in range(self.ngrid):
                for k in range(j, self.ngrid):
                    integrand = (
                        lambda x: np.conjugate(self.fbr_func(j, x))
                        * x
                        * self.fbr_func(k, x)
                    )
                    self.pos_rep_matrix[j, k], error = integrate.quad(
                        integrand, self.lb, self.ub
                    )
                    avg_error += error
            avg_error /= self.ngrid * (self.ngrid - 1) // 2
            print(f"average numerical integral error : {avg_error}")
            self.pos_rep_matrix += np.conjugate(
                self.pos_rep_matrix.T
            ) - np.diag(np.diag(self.pos_rep_matrix))
        return self.pos_rep_matrix

    @abstractmethod
    def get_1st_derivative_matrix_fbr(self) -> np.ndarray:
        r"""Numerical integral of :math:`\langle\varphi_j|\frac{d}{dq}|\varphi_k\rangle`

        If analytical integral is known, implemented in inheritance class

        """
        raise NotImplementedError

    def get_1st_derivative_matrix_dvr(self) -> np.ndarray:
        r""":math:`\langle\chi_\alpha|\frac{d}{dq}|\chi_\beta\rangle`"""
        if not hasattr(self, "first_derivative_matrix_dvr"):
            self.first_derivative_matrix_dvr = (
                self.get_unitary().conj().T
                @ self.get_1st_derivative_matrix_fbr()
                @ self.get_unitary()
            )
        return self.first_derivative_matrix_dvr

    @abstractmethod
    def get_2nd_derivative_matrix_fbr(self) -> np.ndarray:
        r"""Numerical integral of :math:`\langle\varphi_j|\frac{d^2}{dq^2}|\varphi_k\rangle`

        If analytical integral is known, implemented in inheritance class

        """
        raise NotImplementedError

    def get_2nd_derivative_matrix_dvr(self) -> np.ndarray:
        r""":math:`\langle\chi_\alpha|\frac{d^2}{dq^2}|\chi_\beta\rangle`"""
        if not hasattr(self, "second_derivative_matrix_dvr"):
            self.second_derivative_matrix_dvr = (
                self.get_unitary().conj().T
                @ self.get_2nd_derivative_matrix_fbr()
                @ self.get_unitary()
            )
        return self.second_derivative_matrix_dvr

    @abstractmethod
    def diagnalize_pos_rep_matrix(self) -> None:
        """Numerical diagonalization of `pos_rep_matrix`.

        If analytical diagonalization is known, implemented in inheritance class

        """
        if not hasattr(self, "grids"):
            eig_val, eig_vec = eigh(self.get_pos_rep_matrix())
            self.grids = list(eig_val)
            self.unitary = eig_vec
            self.get_sqrt_weights()

    def get_sqrt_weights(self, k: int = 0) -> List[float]:
        r""":math:`\sqrt{w_\alpha}=U_{k\alpha}^{\ast}/\varphi_k(x_\alpha)`"""
        if not hasattr(self, "sqrt_weights"):
            self.sqrt_weights = [
                (
                    np.conjugate(self.get_unitary()[k, alpha])
                    / self.fbr_func(k, self.get_grids()[alpha])
                ).real
                for alpha in range(self.ngrid)
            ]
            for alpha in range(self.ngrid):
                if self.sqrt_weights[alpha].real < 0:
                    self.sqrt_weights[alpha] *= -1.0
                    self.unitary[:, alpha] *= -1.0

        return self.sqrt_weights

    def get_grids(self) -> List[float]:
        r"""grids :math:`x_\alpha` correspond to eigenvalue of `pos_rep_matrix`"""
        if not hasattr(self, "grids"):
            self.diagnalize_pos_rep_matrix()
        return self.grids

    def get_unitary(self) -> np.ndarray:
        r"""Get Unitary Matrix which diagonalize `pos_rep_matrix`

        Returns:
            np.ndarray : `u[alpha,j]` = :math:`(U_{j\alpha})^\dagger` = :math:`(U^\dagger)_{\alpha j}`

        where,

        .. math::
           \sum_{j,k}
           U_{j\alpha}\langle\varphi_j|\hat{q}|\varphi_k\rangle U_{k\beta}^\dagger
           = x_\alpha \delta_{\alpha\beta}

        """
        if not hasattr(self, "unitary"):
            self.diagnalize_pos_rep_matrix()
        return self.unitary

    def dvr_func(self, n: int, q: float) -> float:
        r"""DVR function

        .. math::
           \chi_\alpha=\sum_{j=0}^{N-1}\varphi_j(q)U_{j\alpha} \quad (\alpha=0,\ldots, N-1)

        In other words,


        .. math::
           |\chi_\alpha\rangle =U^\dagger |\varphi_j\rangle

        """
        if not (0 <= n < self.ngrid):
            ValueError
        dum = 0.0
        for j in range(self.ngrid):
            dum += self.fbr_func(j, q) * self.get_unitary()[j, n]
        return dum

    def plot_fbr(self, n: int = None, q: float = None) -> None:
        r"""Plot FBR :math:`\{\varphi_n(q)\}`"""
        plt.title(f"{self.label}-FBR funtions")
        self._plot(self.fbr_func, n, q, name="fbr-func")

    def plot_dvr(self, n: int = None, q: int = None) -> None:
        r"""Plot DVR functions :math:`\{\chi_n(q)\}`"""
        plt.title(f"{self.label}-DVR functions")
        self._plot(self.dvr_func, n, q, name="dvr-func")

    def _plot(
        self, func: Callable, n: int = None, q: float = None, name=None
    ) -> None:
        if q is None:
            q = np.linspace(self.lb, self.ub, 100)
        if n is None:
            for n in range(self.ngrid):
                array = func(n, q)
                # if imaginary part is small, plot real part
                if np.max(np.abs(array.imag)) < 1e-10:
                    plt.plot(q, array.real, label=f"{n}")
                else:
                    plt.plot(q, array.real, label=f"{n} real")
                    plt.plot(q, array.imag, label=f"{n} imag", linestyle="--")
        else:
            array = func(n, q)
            # if imaginary part is small, plot real part
            if np.max(np.abs(array.imag)) < 1e-10:
                plt.plot(q, array.real, label=f"{n}")
            else:
                plt.plot(q, array.real, label=f"{n} real")
                plt.plot(q, array.imag, label=f"{n} imag")
        plt.legend(loc="upper right")
        if type(name) is str:
            plt.savefig(f"{name}.pdf")
        plt.show()
