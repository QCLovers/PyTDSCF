"""
Core tensor (site coefficient) class for MPS & MPO
"""

from __future__ import annotations

import math
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scipy.linalg as linalg

from pytdscf._const_cls import const


class SiteCoef:
    r"""Site Coefficient (:math:`a_{\tau_{p-1}\tau_{p}}^{j_p}`) \
        Array Like Class

    Args:
        data (np.ndarray) : data (3 rank tensor)
        gauge (str) : gauge type

    """

    gauge: Literal["C", "A", "B", "Psi", "Gamma", "sigma", "Lambda", "V"]
    # "C" : Arbitrary 3-rank tensor
    # "A" : A tensor (left canonical) (A†A=I)
    # "B" : B tensor (right canonical) (BB†=I)
    # "Psi" : 3-rank orthogonal center tensor (Psi=A*Lambda=Lambda*B)
    # "Gamma" : 3-rank tensor introduced by Vidal (A=Lambda*Gamma, B=Gamma*Lambda)
    # "sigma" : Arbitrary 2-rank tensor
    # "Lambda" : Diagonal matrix defined by Lambda=diag(lambda_1,lambda_2,...)
    # "V" : 2-rank tensor introduced by Stoudenmire and White (V=Lambda^(-1))

    # Static index
    l_indx = 0
    n_indx = 1
    r_indx = 2

    def __init__(
        self,
        data: np.ndarray | jax.Array,
        gauge: Literal["C", "A", "B", "Psi", "Gamma", "sigma", "Lambda", "V"],
        isite: int,
    ):
        self.data = data
        self.gauge = gauge
        self.shape = data.shape
        self.isite = isite

    def copy(self):
        return SiteCoef(self.data.copy(), self.gauge, self.isite)

    def __array__(self, copy: bool | None = None):
        """
        Whatever `copy` is given, always return shallow copy of data
        in order to handle the warning,
        DeprecationWarning: __array__ implementation doesn't accept a copy keyword,
        so passing copy=False failed. __array__ must implement 'dtype' and 'copy' keyword arguments.
        """  # noqa: E501
        return self.data

    def __array_wrap__(self, out_arr, context=None, return_scalar=False):
        if return_scalar:
            raise ValueError("SiteCoef is always tensor. Cannot return scalar.")
        return out_arr

    def __repr__(self):
        # print like MPS core
        # for example
        # when self.gauge == "L" and self.data.shape == (l,c,r):
        # define picture as
        #    [c]
        #     |
        # [l]-L-[r]

        l, c, r = self.data.shape  # noqa: E741
        left_space = len(str(l))
        picture = "\n" + " " * (left_space + 2) + f"[{c}]\n"
        picture += " " * (left_space + 3) + "|\n"
        picture += f"[{l}]-{self.gauge}-[{r}]\n"
        return picture
        # return "SiteCoef:\n{0} {1}".format(self.gauge, self.data)

    def __add__(self, other):
        assert self.gauge == other.gauge
        assert self.isite == other.isite
        return SiteCoef(self.data + other.data, self.gauge, self.isite)

    def __sub__(self, other):
        assert self.gauge == other.gauge
        assert self.isite == other.isite
        return SiteCoef(self.data - other.data, self.gauge, self.isite)

    def __mul__(self, scale):
        return SiteCoef(self.data * scale, self.gauge, self.isite)

    def __rmul__(self, scale):
        return SiteCoef(self.data * scale, self.gauge, self.isite)

    def __truediv__(self, scale):
        return SiteCoef(self.data / scale, self.gauge, self.isite)

    def dot_conj(self, other):
        return np.inner(np.conj(other).flatten(), np.array(self).flatten())

    def size(self):
        return self.data.size

    def norm(self):
        if const.use_jax:
            return jnp.linalg.norm(self.data)
        else:
            return linalg.norm(np.array(self.data))

    def conj(self) -> SiteCoef:
        if isinstance(self.data, np.ndarray):
            data = np.conj(self.data)
        else:
            data = jnp.conj(self.data)
        return SiteCoef(data, self.gauge, self.isite)

    def gauge_trf(
        self,
        key: Literal["Psi2Asigma", "Psi2sigmaB", "C2Asigma", "C2sigmaB"],
        regularize: bool = False,
    ) -> tuple[SiteCoef, np.ndarray | jax.Array]:
        r""" Gauge transformation of MPS site

        .. math ::
           \Psi_{\tau_{p-1}\tau_p}^{j_p} \to \sum_{\tau_{p-1}^\prime}
           \sigma_{\tau_{p-1}\tau_{p-1}^\prime}B_{\tau_{p-1}^\prime\tau_p}^{j_p}

        or

        .. math ::
           \Psi_{\tau_{p-1}\tau_p}^{j_p} \to \sum_{\tau_{p}^\prime}
           A_{\tau_{p-1}\tau_p^\prime}^{j_p}\sigma_{\tau_p^\prime\tau_p}

        Args:
            key (str): "Psi2Asigma" or "Psi2sigmaB"
            regularize (bool, optional): Defaults to False.

        Returns:
            Tuple[SiteCoef, np.ndarray | jax.Array]: \
                (L or R site, :math:`\sigma`)
        """
        match key:
            case "Psi2Asigma":
                assert (
                    self.gauge == "Psi"
                ), f"Gauge must be Psi but got {self.gauge}"
                sys_indx, env_indx, dot_indx = (
                    self.l_indx,
                    self.r_indx,
                    self.n_indx,
                )
            case "C2Asigma":
                assert (
                    self.gauge == "C"
                ), f"Gauge must be C but got {self.gauge}"
                sys_indx, env_indx, dot_indx = (
                    self.l_indx,
                    self.r_indx,
                    self.n_indx,
                )
            case "Psi2sigmaB":
                assert (
                    self.gauge == "Psi"
                ), f"Gauge must be Psi but got {self.gauge}"
                sys_indx, env_indx, dot_indx = (
                    self.r_indx,
                    self.l_indx,
                    self.n_indx,
                )
            case "C2sigmaB":
                assert (
                    self.gauge == "C"
                ), f"Gauge must be C but got {self.gauge}"
                sys_indx, env_indx, dot_indx = (
                    self.r_indx,
                    self.l_indx,
                    self.n_indx,
                )
            case _:
                raise ValueError(f"Invalid key: {key}")
        if const.use_jax:
            matC = self.data
        else:
            matC = np.array(self)
        """still experimental and this makes linalg.eigh(proj_dens) slow...)"""
        if regularize:
            """regularize the site coefficients"""
            ldim, ndim, rdim = matC.shape
            sqrt_epsrho = math.sqrt(const.epsrho)
            sig_reg: jax.Array | np.ndarray
            if const.use_jax:
                U, sig, Vh = jsp.linalg.svd(
                    matC.transpose(0, 2, 1).reshape(-1, ndim),
                    full_matrices=False,
                )
                sig_reg = jnp.where(
                    sig > 64.0e0 * sqrt_epsrho,
                    sig,
                    sig + sqrt_epsrho * jnp.exp(-sig / sqrt_epsrho),
                )
                matC = (
                    jnp.dot(U, jnp.dot(jnp.diag(sig_reg), Vh))
                    .reshape(ldim, rdim, ndim)
                    .transpose(0, 2, 1)
                )
            else:
                U, sig, Vh = linalg.svd(
                    matC.transpose(0, 2, 1).reshape(-1, ndim),
                    full_matrices=False,
                )
                sig_reg = np.where(
                    sig > 64.0e0 * sqrt_epsrho,
                    sig,
                    sig + sqrt_epsrho * np.exp(-sig / sqrt_epsrho),
                )
                matC = (
                    np.dot(U, np.dot(np.diag(sig_reg), Vh))
                    .reshape(ldim, rdim, ndim)
                    .transpose(0, 2, 1)
                )
        m_aux_env = matC.shape[env_indx]
        m_aux_sys = matC.shape[sys_indx]
        nspf = matC.shape[dot_indx]
        ndim = nspf * m_aux_sys
        if env_indx == 0:
            if const.use_jax:
                sval, matR = gauge_trf_LQ(matC, m_aux_sys, nspf, m_aux_env)
            else:
                Q, R = linalg.qr(
                    matC.transpose((2, 1, 0)).reshape(ndim, -1), mode="economic"
                )
                sval = R.transpose()
                matR = Q.reshape(-1, nspf, m_aux_env).transpose((2, 1, 0))
            coef = SiteCoef(data=matR, gauge="B", isite=self.isite)
        else:
            if const.use_jax:
                sval, matL = gauge_trf_QR(matC, m_aux_sys, nspf, m_aux_env)
            else:
                Q, R = linalg.qr(matC.reshape(ndim, -1), mode="economic")
                sval = R
                matL = Q.reshape(-1, nspf, m_aux_env)
            coef = SiteCoef(data=matL, gauge="A", isite=self.isite)
        return (coef, sval)

    @classmethod
    def init_random(
        cls,
        isite: int,
        ndim: int,
        m_aux_l: int,
        m_aux_r: int,
        vibstate: list[float],
        is_lend: bool = False,
        is_rend: bool = False,
    ) -> SiteCoef:
        r"""Initialization randomly

        Args:
            ndim (int): Number of SPFs. :math:`j_p` =1,2,...,ndim
            m_aux_l (int): Left side bond dimension. \
                :math:`\tau_{p-1}` =1,2,...,m_aux_l
            m_aux_r (int): Right side bond dimension. \
                :math:`\tau_{p}` =1,2,...,m_aux_r
            vibstate (List[float]): Vibrational state.
            is_lend (bool, optional): \
                Whether site is left terminal. Defaults to False.
            is_rend (bool, optional): \
                Whether site is right terminal. Defaults to False.

        Returns:
            SiteCoef: Site Coefficient data \
                :math:`a_{\tau_{p-1}\tau_{p}}^{j_p}`
        """
        shape: list[int] = [None, None, None]  # type: ignore
        shape[SiteCoef.n_indx] = ndim
        shape[SiteCoef.l_indx] = m_aux_l if not is_lend else 1
        shape[SiteCoef.r_indx] = m_aux_r if not is_rend else 1

        data: np.ndarray | jax.Array
        data = np.zeros(shape, dtype="complex128")
        vibstate_array = np.array(vibstate, dtype="complex128")
        data[0, :, 0] = vibstate_array / np.linalg.norm(vibstate_array)

        if const.use_jax:
            data = jnp.array(data, dtype=jnp.complex128)

        obj = SiteCoef(data, "C", isite=isite)
        return obj / obj.norm()


@partial(jax.jit, static_argnums=(1, 2, 3))
def gauge_trf_LQ(
    matC: jax.Array, m_aux_sys: int, nspf: int, m_aux_env: int
) -> tuple[jax.Array, jax.Array]:
    Q, R = jsp.linalg.qr(
        matC.transpose((2, 1, 0)).reshape(m_aux_sys * nspf, m_aux_env),
        mode="economic",
    )
    return R.transpose(), Q.reshape(m_aux_sys, nspf, m_aux_env).transpose(
        (2, 1, 0)
    )


@partial(jax.jit, static_argnums=(1, 2, 3))
def gauge_trf_QR(
    matC: jax.Array, m_aux_sys: int, nspf: int, m_aux_env: int
) -> tuple[jax.Array, jax.Array]:
    Q, R = jsp.linalg.qr(
        matC.reshape(m_aux_sys * nspf, m_aux_env), mode="economic"
    )
    return R, Q.reshape(m_aux_sys, nspf, m_aux_env)


def validate_Btensor(coef: np.ndarray | jax.Array | SiteCoef) -> None:
    r"""Check if data is B tensor

    Args:
        coef (np.ndarray | jax.Array | SiteCoef): Data to check

    Returns:
        bool: Whether data is B tensor

    B tensor satisfies the following condition:

    sum_(sigma_p) B_p^(sigma_p) B_p^(sigma_p)† = I
    """
    subscript = "ipk,lpk->il"
    if isinstance(coef, SiteCoef):
        assert coef.gauge == "B", "Gauge must be B but got {coef.gauge}"
        data = coef.data
    else:
        data = coef
    assert data.ndim == 3, "Data must be 3 rank tensor but got {data.ndim}"
    if const.use_jax:
        _BB = jnp.einsum(subscript, data, jnp.conj(data))
        BB = np.array(_BB)
    else:
        BB = np.einsum(subscript, data, np.conj(data))
    assert BB.ndim == 2
    assert BB.shape[0] == BB.shape[1]
    np.testing.assert_allclose(BB, np.eye(BB.shape[0]), atol=1.0e-15)


def validate_Atensor(coef: np.ndarray | jax.Array | SiteCoef) -> None:
    r"""Check if data is A tensor

    Args:
        coef (np.ndarray | jax.Array | SiteCoef): Data to check

    Returns:
        bool: Whether data is A tensor

    A tensor satisfies the following condition:

    sum_(sigma_p) A_p^(sigma_p)† A_p^(sigma_p) = I
    """
    subscript = "ipj,ipk->jk"
    if isinstance(coef, SiteCoef):
        assert coef.gauge == "A", f"Gauge must be A but got {coef.gauge}"
        data = coef.data
    else:
        data = coef
    assert data.ndim == 3, f"Data must be 3 rank tensor but got {data.ndim}"
    if const.use_jax:
        _AA = jnp.einsum(subscript, jnp.conj(data), data)
        AA = np.array(_AA)
    else:
        AA = np.einsum(subscript, np.conj(data), data)
    assert AA.ndim == 2
    assert AA.shape[0] == AA.shape[1]
    np.testing.assert_allclose(AA, np.eye(AA.shape[0]), atol=1.0e-15)
