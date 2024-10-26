"""
Core tensor (site coefficient) class for MPS & MPO
"""

from __future__ import annotations

import math

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
        gauge (str) : Canonical gauge "L" or "C" or "R".

    """

    gauge = None
    l_indx = 0
    n_indx = 1
    r_indx = 2

    def __init__(self, data: np.ndarray | jax.Array, gauge: str | None):
        self.data = data
        self.gauge = gauge
        self.shape = data.shape

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
        return SiteCoef(self.data + other.data, self.gauge)

    def __sub__(self, other):
        assert self.gauge == other.gauge
        return SiteCoef(self.data - other.data, self.gauge)

    def __mul__(self, scale):
        return SiteCoef(self.data * scale, self.gauge)

    def __rmul__(self, scale):
        return SiteCoef(self.data * scale, self.gauge)

    def __truediv__(self, scale):
        return SiteCoef(self.data / scale, self.gauge)

    def dot_conj(self, other):
        return np.inner(np.conj(other).flatten(), np.array(self).flatten())

    def size(self):
        return self.data.size

    def norm(self):
        assert self.gauge == "C"
        if const.use_jax:
            return jnp.linalg.norm(self.data)
        else:
            return linalg.norm(np.array(self.data))

    def gauge_trf(
        self, key: str, regularize: bool = False
    ) -> tuple[SiteCoef, np.ndarray | jax.Array]:
        r""" Gauge transformation of MPS site

        .. math ::
           C_{\tau_{p-1}\tau_p}^{j_p} \to \sum_{\tau_{p-1}^\prime}
           \sigma_{\tau_{p-1}\tau_{p-1}^\prime}R_{\tau_{p-1}^\prime\tau_p}^{j_p}

        or

        .. math ::
           C_{\tau_{p-1}\tau_p}^{j_p} \to \sum_{\tau_{p}^\prime}
           L_{\tau_{p-1}\tau_p^\prime}^{j_p}\sigma_{\tau_p^\prime\tau_p}

        Args:
            key (str): "C2L" or "C2R"
            regularize (bool, optional): Defaults to False.

        Returns:
            Tuple[SiteCoef, np.ndarray | jax.Array]: \
                (L or R site, :math:`\sigma`)
        """
        if key == "C2L":
            sys_indx, env_indx, dot_indx = self.l_indx, self.r_indx, self.n_indx
        elif key == "C2R":
            sys_indx, env_indx, dot_indx = self.r_indx, self.l_indx, self.n_indx
        else:
            raise ValueError(f"Key must be 'C2L' or 'C2R', but got {key}")
        assert self.gauge == "C"
        if const.use_jax:
            matC = self.data
        else:
            matC = np.array(self)
        """still experimental and this makes linalg.eigh(proj_dens) slow...)"""
        if regularize:
            """regularize the site coefficients"""
            """Not Yet Tensorflow support"""
            if const.use_jax:
                raise NotImplementedError
            else:
                ldim, ndim, rdim = matC.shape
                U, sig, Vh = linalg.svd(
                    matC.transpose(0, 2, 1).reshape(-1, ndim),
                    full_matrices=False,
                )
                sqrt_epsrho = math.sqrt(const.epsrho)
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
                sval, matR = jax.jit(
                    gauge_trf_LQ,
                    static_argnums=(
                        1,
                        2,
                        3,
                    ),
                )(matC, m_aux_sys, nspf, m_aux_env)
            else:
                Q, R = linalg.qr(
                    matC.transpose((2, 1, 0)).reshape(ndim, -1), mode="economic"
                )
                sval = R.transpose()
                matR = Q.reshape(-1, nspf, m_aux_env).transpose((2, 1, 0))
            coef = SiteCoef(data=matR, gauge="R")
        else:
            if const.use_jax:
                sval, matL = jax.jit(
                    gauge_trf_QR,
                    static_argnums=(
                        1,
                        2,
                        3,
                    ),
                )(matC, m_aux_sys, nspf, m_aux_env)
            else:
                Q, R = linalg.qr(matC.reshape(ndim, -1), mode="economic")
                sval = R
                matL = Q.reshape(-1, nspf, m_aux_env)
            coef = SiteCoef(data=matL, gauge="L")
        return (coef, sval)

    @classmethod
    def init_random(
        cls,
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

        obj = SiteCoef(data, "C")
        return obj / obj.norm()


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


def gauge_trf_QR(
    matC: jax.Array, m_aux_sys: int, nspf: int, m_aux_env: int
) -> tuple[jax.Array, jax.Array]:
    Q, R = jsp.linalg.qr(
        matC.reshape(m_aux_sys * nspf, m_aux_env), mode="economic"
    )
    return R, Q.reshape(m_aux_sys, nspf, m_aux_env)
