"""
Core tensor (site coefficient) class for MPS & MPO
"""

from __future__ import annotations

from functools import partial
from math import isqrt
from typing import Literal, overload

import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger as _logger
from scipy.linalg import qr, svd

from pytdscf._const_cls import const

logger = _logger.bind(name="main")
rank_logger = _logger.bind(name="rank")

SQRT_EPSRHO = 1e-04  # const.p_svd
TIKHONOV_LAMBDA = 1.0e-08
RCOND = 1e-13  # const.p_svd


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
        self.isite = isite

    @property
    def shape(self):
        return self.data.shape

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
        return np.inner(np.conj(other).ravel(), np.array(self).ravel())

    def size(self):
        return self.data.size

    def norm(self):
        if const.use_jax:
            return jnp.linalg.norm(self.data)
        else:
            return np.linalg.norm(np.array(self.data))

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
                assert self.gauge == "Psi", (
                    f"Gauge must be Psi but got {self.gauge}"
                )
                sys_indx, env_indx, dot_indx = (
                    self.l_indx,
                    self.r_indx,
                    self.n_indx,
                )
            case "C2Asigma":
                assert self.gauge == "C", (
                    f"Gauge must be C but got {self.gauge}"
                )
                sys_indx, env_indx, dot_indx = (
                    self.l_indx,
                    self.r_indx,
                    self.n_indx,
                )
            case "Psi2sigmaB":
                assert self.gauge == "Psi", (
                    f"Gauge must be Psi but got {self.gauge}"
                )
                sys_indx, env_indx, dot_indx = (
                    self.r_indx,
                    self.l_indx,
                    self.n_indx,
                )
            case "C2sigmaB":
                assert self.gauge == "C", (
                    f"Gauge must be C but got {self.gauge}"
                )
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
            # raise NotImplementedError(
            #     "Regularization is not guaranteed to be correct."
            # )
            """regularize the site coefficients"""
            ldim, ndim, rdim = matC.shape
            # sqrt_epsrho = math.sqrt(const.epsrho)
            sqrt_epsrho = SQRT_EPSRHO
            sig_reg: jax.Array | np.ndarray
            if const.use_jax:
                U, sig, Vh = jnp.linalg.svd(
                    matC.transpose(0, 2, 1).reshape(-1, ndim),
                    full_matrices=False,
                )
                sig_reg = jnp.where(
                    sig > sqrt_epsrho,
                    sig,
                    sig + sqrt_epsrho * jnp.exp(-sig / sqrt_epsrho),
                )
                matC = (
                    jnp.dot(U, jnp.dot(jnp.diag(sig_reg), Vh))
                    .reshape(ldim, rdim, ndim)
                    .transpose(0, 2, 1)
                )
            else:
                # U, sig, Vh = np.linalg.svd(
                #     matC.transpose(0, 2, 1).reshape(-1, ndim),
                #     full_matrices=False,
                # )
                U, sig, Vh = svd(
                    np.ascontiguousarray(
                        matC.transpose(0, 2, 1).reshape(-1, ndim)
                    ),
                    full_matrices=False,
                    overwrite_a=True,
                )
                sig_reg = np.where(
                    sig > sqrt_epsrho,
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
                # Q, R = np.linalg.qr(
                #     matC.transpose((2, 1, 0)).reshape(ndim, -1), mode="reduced"
                # )
                Q, R = qr(
                    np.ascontiguousarray(
                        matC.transpose((2, 1, 0)).reshape(ndim, -1)
                    ),
                    mode="economic",
                    overwrite_a=True,
                )
                sval = R.transpose()
                matR = Q.reshape(m_aux_sys, nspf, -1).transpose((2, 1, 0))
            coef = SiteCoef(data=matR, gauge="B", isite=self.isite)
        else:
            if const.use_jax:
                sval, matL = gauge_trf_QR(matC, m_aux_sys, nspf, m_aux_env)
            else:
                # Q, R = np.linalg.qr(matC.reshape(ndim, -1), mode="reduced")
                Q, R = qr(
                    matC.reshape(ndim, -1), mode="economic", overwrite_a=True
                )
                sval = R
                try:
                    matL = Q.reshape(m_aux_sys, nspf, -1)
                except ValueError:
                    logger.error(Q.shape)
                    logger.error(R.shape)
                    logger.error(matC.shape)
                    logger.error(f"{m_aux_sys=}, {nspf=}, {m_aux_env=}")
                    raise
            coef = SiteCoef(data=matL, gauge="A", isite=self.isite)
        return (coef, sval)

    def thin_to_full(self, delta_rank: int = 1) -> SiteCoef:
        """
        QR decomposition is defined by
        [[Q1, Q2]] [[R1],
                    [ 0]]
        where Q1 and Q2 are orthogonal matrices, and R1 is an upper triangular matrix.
        The A or B tensor is usually Q1 or its transpose.
        For adaptive 1-site TDVP,
        the full_rank A = [Q1, Q2] or B = [Q1, Q2]^t is required.
        This function returns the full-rank A or B tensor.
        """
        l, c, r = self.data.shape  # noqa: E741
        if isinstance(self.data, jax.Array):
            raise NotImplementedError
        match self.gauge:
            case "A":
                assert l * c >= r
                dr = min(delta_rank, l * c - r)
                mat = self.data.reshape((l * c, r))
                # Q, _ = np.linalg.qr(mat, mode="complete")
                Q, _ = qr(mat, mode="full", overwrite_a=False)
                # to align sign of Q, calculate inner product of Q and Q_ref
                ip = mat.T.conj() @ Q
                unflip = np.sign(np.sign(np.diag(ip[:r, :r])) + 0.5)
                Q = Q[:, : r + dr]
                Q[:, :r] *= unflip[np.newaxis, :]
                if const.pytest_enabled:
                    assert self.data.shape == (l, c, r)
                    assert mat.shape == (l * c, r)
                    assert ip.shape == (r, l * c)
                    np.testing.assert_allclose(
                        mat.T.conj() @ mat, np.eye(r), atol=1.0e-14
                    )
                    np.testing.assert_allclose(
                        np.abs(ip[:r, :r]), np.eye(r), atol=1.0e-14
                    )
                    np.testing.assert_allclose(Q[:, :r], mat, atol=1.0e-14)
                    np.testing.assert_allclose(
                        (Q.T.conj() @ Q)[: r + dr, : r + dr],
                        np.eye(r + dr),
                        atol=1.0e-14,
                    )
                Q = Q.reshape(l, c, r + dr)
                return SiteCoef(data=Q, gauge="A", isite=self.isite)
            case "B":
                assert c * r >= l
                dl = min(delta_rank, c * r - l)
                mat = self.data.reshape((l, c * r)).transpose(1, 0)
                # Q, _ = np.linalg.qr(mat, mode="complete")
                Q, _ = qr(
                    np.ascontiguousarray(mat), mode="full", overwrite_a=False
                )
                # to align sign of Q, calculate inner product of Q and Q_ref
                ip = mat.T.conj() @ Q
                unflip = np.sign(np.sign(np.diag(ip[:l, :l])) + 0.5)
                Q = Q[:, : l + dl]
                Q[:, :l] *= unflip[np.newaxis, :]
                # confirm Q is orthogonal
                if const.pytest_enabled:
                    np.testing.assert_allclose(
                        mat.T.conj() @ mat, np.eye(l), atol=1.0e-14
                    )
                    assert mat.shape == (c * r, l)
                    assert self.data.shape == (l, c, r)
                    assert ip.shape == (l, c * r)
                    np.testing.assert_allclose(
                        np.abs(ip[:l, :l]), np.eye(l), atol=1.0e-14
                    )
                    np.testing.assert_allclose(
                        (Q.T.conj() @ Q)[: l + dl, : l + dl],
                        np.eye(l + dl),
                        atol=1.0e-14,
                    )
                Q = Q.transpose(1, 0)
                Q = Q.reshape(l + dl, c, r)
                return SiteCoef(data=Q, gauge="B", isite=self.isite)
            case _:
                raise ValueError(f"Invalid gauge: {self.gauge}")

    @classmethod
    def init_random(
        cls,
        isite: int,
        ndim: int,
        m_aux_l: int,
        m_aux_r: int,
        init_state: list[float] | np.ndarray,
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
            init_state (List[float] | np.ndarray): Initial core state.
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
        init_state_array = np.array(init_state, dtype="complex128")
        if init_state_array.ndim == 1:
            # Common API (define initial state as a Hartree product (m=1))
            data[0, :, 0] = init_state_array
            match const.space:
                case "hilbert":
                    # In Hilbert space, the norm of the wavefunction is 1.0
                    data[0, :, 0] /= np.linalg.norm(init_state_array)
                case "liouville":
                    # In Liouville space, the trace of density matrix is 1.0
                    trace = np.trace(
                        init_state_array.reshape(isqrt(ndim), isqrt(ndim))
                    )
                    data[0, :, 0] /= trace
        elif init_state_array.ndim == 3:
            # Lower API (define core state directly)
            i, j, k = init_state_array.shape
            try:
                data[:i, :j, :k] = init_state_array
            except Exception as e:
                msg = "Failed to assign init_state_array to data"
                msg += f"isite: {isite}"
                msg += f"Expected shape is smaller than {data.shape}"
                msg += f"tried shape: {init_state_array.shape}"
                msg += f"tried values = {init_state_array}"
                logger.error(msg)
                raise e

        if const.use_jax:
            data = jnp.array(data, dtype=jnp.complex128)

        return SiteCoef(data, "C", isite=isite)


@partial(jax.jit, static_argnums=(1, 2, 3))
def gauge_trf_LQ(
    matC: jax.Array, m_aux_sys: int, nspf: int, m_aux_env: int
) -> tuple[jax.Array, jax.Array]:
    Q, R = jnp.linalg.qr(
        matC.transpose((2, 1, 0)).reshape(m_aux_sys * nspf, m_aux_env),
        mode="reduced",
    )
    return R.transpose(), Q.reshape(m_aux_sys, nspf, m_aux_env).transpose(
        (2, 1, 0)
    )


@partial(jax.jit, static_argnums=(1, 2, 3))
def gauge_trf_QR(
    matC: jax.Array, m_aux_sys: int, nspf: int, m_aux_env: int
) -> tuple[jax.Array, jax.Array]:
    Q, R = jnp.linalg.qr(
        matC.reshape(m_aux_sys * nspf, m_aux_env), mode="reduced"
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


@overload
def truncate_sigvec(
    Asite: SiteCoef, sigvec: np.ndarray, Bsite: SiteCoef, p: float
) -> tuple[SiteCoef, np.ndarray, SiteCoef]: ...


@overload
def truncate_sigvec(
    Asite: None, sigvec: np.ndarray, Bsite: SiteCoef, p: float
) -> tuple[np.ndarray, np.ndarray, SiteCoef]: ...


@overload
def truncate_sigvec(
    Asite: SiteCoef, sigvec: np.ndarray, Bsite: None, p: float
) -> tuple[SiteCoef, np.ndarray, np.ndarray]: ...


@overload
def truncate_sigvec(
    Asite: None, sigvec: np.ndarray, Bsite: None, p: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


def truncate_sigvec(
    Asite: SiteCoef | None,
    sigvec: np.ndarray,
    Bsite: SiteCoef | None,
    p: float,
    regularize: bool = False,
    keepdim: bool = False,
) -> tuple[SiteCoef | np.ndarray, np.ndarray, SiteCoef | np.ndarray]:
    r"""Truncate singular vector

    Args:
        Asite (SiteCoef): A tensor
        sigvec (np.ndarray): Singular vector
        Bsite (SiteCoef): B tensor
        p (float): Truncation parameter

    Returns:
        tuple[SiteCoef, np.ndarray | jax.Array, SiteCoef]: \
            (Asite, sigvec, Bsite)

    Given a matrix in AσB form:
    1. Perform SVD decomposition on σ to get Uσ'Vh
    2. Truncate singular values based on their cumulative contribution
    3. Transform the tensors:
       A -> AU
       σ -> σ' (truncated)
       B -> VhB

    If Asite is None, return (U, σ', VhB)
    If Bsite is None, return (AU, σ', Vh)
    If both are None, return (U, σ', Vh)
    """
    # if regularize:
    #     raise ValueError("Use eval_PsiXpinvPsi instead")
    if isinstance(sigvec, jax.Array):
        raise NotImplementedError
    # U, sigvec2, Vh = np.linalg.svd(sigvec, full_matrices=False)
    U, sigvec2, Vh = svd(sigvec, full_matrices=False, overwrite_a=True)
    cumsum = np.cumsum(sigvec2.real)
    contribution = cumsum / cumsum[-1]
    idx = np.argmax(contribution >= (1 - p)) + 1
    sigvec2_thin = sigvec2[:idx]
    if not keepdim:
        U = U[:, :idx]
        Vh = Vh[:idx, :]
    if isinstance(Asite, SiteCoef):
        assert Asite.gauge == "A", "Asite must be A tensor"
        Asite.data = np.tensordot(Asite.data, U, axes=(2, 0))
        L = Asite
    else:
        L = U
    if isinstance(Bsite, SiteCoef):
        assert Bsite.gauge == "B", "Bsite must be B tensor"
        Bsite.data = np.tensordot(Vh, Bsite.data, axes=(1, 0))
        R = Bsite
    else:
        R = Vh
    if regularize and sigvec.shape != (1, 1):
        # sqrt_epsrho = math.sqrt(const.epsrho)
        sqrt_epsrho = SQRT_EPSRHO
        sigvec2_thin = np.where(
            sigvec2_thin > sqrt_epsrho,
            sigvec2_thin,
            sigvec2_thin + sqrt_epsrho * np.exp(-sigvec2_thin / sqrt_epsrho),
        )
    if keepdim:
        sigvec_full = np.zeros_like(sigvec2)
        sigvec_full[:idx] = sigvec2_thin
        sigvec = np.diag(sigvec_full)
    else:
        sigvec = np.diag(sigvec2_thin)
    sigvec /= np.sqrt(np.sum(np.diag(sigvec) ** 2))  # normalize
    return L, sigvec, R


@overload
def multiply_sigvec_pinv(
    X: np.ndarray,
    left_tensor: np.ndarray | None = None,
    right_tensor: np.ndarray | None = None,
) -> np.ndarray: ...


@overload
def multiply_sigvec_pinv(
    X: jax.Array,
    left_tensor: jax.Array | None = None,
    right_tensor: jax.Array | None = None,
) -> jax.Array: ...


def multiply_sigvec_pinv(
    X: np.ndarray | jax.Array,
    left_tensor: np.ndarray | jax.Array | None = None,
    right_tensor: np.ndarray | jax.Array | None = None,
):
    """
    Multiply X^{+} Y or Y X^{+}

    Args:
        X (np.ndarray): Matrix to multiply
        left_tensor (np.ndarray | None): Left tensor to multiply
        right_tensor (np.ndarray | None): Right tensor to multiply

    Returns:
        np.ndarray: Result of multiplication
    """
    # raise ValueError("Use eval_PsiXpinvPsi instead")
    if (
        isinstance(X, jax.Array)
        or isinstance(left_tensor, jax.Array)
        or isinstance(right_tensor, jax.Array)
    ):
        raise NotImplementedError
    # XTX_reg = X.T.conj() @ X + TIKHONOV_LAMBDA * np.eye(X.shape[0])
    # X_pinv_reg = np.linalg.solve(XTX_reg, X.T.conj())
    X_pinv_reg = np.linalg.pinv(X, rcond=RCOND)
    match (left_tensor, right_tensor):
        case (None, None):
            return X_pinv_reg
        case (None, _):
            assert isinstance(right_tensor, np.ndarray)
            return np.tensordot(X_pinv_reg, right_tensor, axes=(1, 0))
        case (_, None):
            assert isinstance(left_tensor, np.ndarray)
            left_axes = left_tensor.ndim - 1
            return np.tensordot(left_tensor, X_pinv_reg, axes=(left_axes, 0))
        case (_, _):
            assert isinstance(left_tensor, np.ndarray)
            assert isinstance(right_tensor, np.ndarray)
            left_axes = left_tensor.ndim - 1
            right_axes = right_tensor.ndim - 1
            return np.tensordot(
                np.tensordot(left_tensor, X_pinv_reg, axes=(left_axes, 0)),
                right_tensor,
                axes=(right_axes, 0),
            )
    # if const.mpi_rank % 2 == 0 and use_lstsq:
    #     joint_sigvec = np.linalg.lstsq(
    #         self.joint_sigvec_not_pinv, joint_sigvec
    #     )[0]
    # PsiBBBx+
    # Instead of calculating X=BA^+ where A^+ is the pseudo inverse of A,
    # solve ATXT=BT which gives XT=AT^+BT = (BA^+)^T
    # Btensor = superblock_data[-1]
    # l, c, r = Btensor.shape  # noqa: E741
    # Bmatrix = Btensor.reshape(l * c, r)
    # solution = np.linalg.lstsq(
    #     self.joint_sigvec_not_pinv.T, Bmatrix.T
    # )[0].T
    # superblock_data[-1] = solution.reshape(l, c, r)


def eval_PsiXpinvPsi(
    Psi_L: SiteCoef,
    X: np.ndarray | jax.Array,
    Psi_R: SiteCoef,
) -> tuple[SiteCoef, SiteCoef]:
    """
    Evaluate Psi_L X^{+} Psi_R = AZX^{+}YB = AWB = PsiB
    """
    if isinstance(X, jax.Array):
        raise NotImplementedError
    assert Psi_L.gauge == "Psi", (
        f"Psi_L must be Psi tensor but got {Psi_L.gauge}"
    )
    assert Psi_R.gauge == "Psi", (
        f"Psi_R must be Psi tensor but got {Psi_R.gauge}"
    )

    A, Z = Psi_L.gauge_trf(key="Psi2Asigma")
    B, Y = Psi_R.gauge_trf(key="Psi2sigmaB")
    norm_Z = np.linalg.norm(Z)
    norm_Y = np.linalg.norm(Y)
    if const.pytest_enabled:
        assert norm_Z - 1.0 < 1e-12, f"{norm_Z=}"
        assert norm_Y - 1.0 < 1e-12, f"{norm_Y=}"
    Z /= norm_Z
    Y /= norm_Y
    Q, R = qr_with_same_sign_diagonal(Q=A.data, R=Z, R_ref=X)
    A.data = Q
    Z = R
    Q, R = qr_with_same_sign_diagonal(Q=B.data.T, R=Y, R_ref=X)
    B.data = Q.T
    Y = R.T
    dZ = Z - X
    dY = Y - X
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    # U, S, Vh = svd(X, full_matrices=False, overwrite_a=False)
    Sinv = np.zeros_like(S)
    Sinv[S > SQRT_EPSRHO] = 1 / S[S > SQRT_EPSRHO]
    Sinv = np.diag(Sinv)
    Xpinv_tilde = Vh.T @ Sinv @ U.T
    W = X + dZ + dY + dZ @ Xpinv_tilde @ dY
    norm_W = np.linalg.norm(W)
    if abs(norm_W - 1.0) > 1e-01:
        rank_logger.warning(
            f"Time step might be too large to keep norm of adjoint site |W|={norm_W:.3e}"
        )
        # raise ValueError(f"Time step might be too large: {norm_W=}")
    W /= norm_W
    Psi_L = SiteCoef(A.data @ W, "Psi", Psi_L.isite)
    return Psi_L, B


def qr_with_same_sign_diagonal(Q, R, R_ref):
    assert Q.ndim == 3, f"{Q.shape=}"
    assert R.shape == R_ref.shape, f"{R.shape=} != {R_ref.shape=}"
    diag_R = np.diag(R.real)
    diag_R_ref = np.diag(R_ref.real)
    # Deal with value close to zero
    signs_R = np.sign(0.5 + np.sign(diag_R))
    signs_R_ref = np.sign(0.5 + np.sign(diag_R_ref))
    # Sign mismatch
    sign_correction = signs_R != signs_R_ref
    R[sign_correction, :] *= -1
    Q[..., sign_correction] *= -1
    return Q, R
