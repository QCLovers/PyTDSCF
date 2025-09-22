"""
Integrator module for MPS time evolution & diagonalization.
"""

from __future__ import annotations

import cmath
from functools import partial
from typing import overload

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
from loguru import logger as _logger

from pytdscf._const_cls import const
from pytdscf._contraction import SplitStack
from pytdscf._helper import _Debug

logger = _logger.bind(name="main")
EPS = 1e-12  # Threshold for whether Krylov subspace is exausted or not


@overload
def expectation_Op(
    bra_states: list[np.ndarray],
    multiplyOp,
    ket_states: list[np.ndarray],
) -> complex: ...


@overload
def expectation_Op(
    bra_states: list[jax.Array],
    multiplyOp,
    ket_states: list[jax.Array],
) -> complex: ...


def expectation_Op(
    bra_states: list[np.ndarray] | list[jax.Array],
    multiplyOp,
    ket_states: list[np.ndarray] | list[jax.Array],
) -> complex:
    """

    get expectation value of a certain operator for MPS.

    Args:
        bra_states (List[np.ndarray | jax.Array]) : \
            bra part of MPS site coefficient in each electronic states.\
            ``np.ndarray`` part shape is (tau_{p-1}, j_p, tau_p).
        multiplyOp (pytdscf._contraction.multiplyH_MPS_direct) : \
            operator in tensor style.
        ket_states (List[np.ndarray | jax.Array]) : \
            bra part of MPS site coefficient in each electronic states.\
            ``np.ndarray`` part shape is (tau_{p-1}, j_p, tau_p).

    Returns:
        complex : expectation value. If operator is hermitian and bra=ket,\
                complex part is very small.

    """
    psivec = multiplyOp.stack(bra_states)
    sigvec = multiplyOp.stack(multiplyOp.dot(ket_states))
    if isinstance(psivec, jax.Array) and isinstance(sigvec, jax.Array):
        retval = np.array(jnp.inner(jnp.conj(psivec), sigvec))
    else:
        retval = np.inner(np.conj(psivec), sigvec)
    return complex(retval)


def matrix_diagonalize_lanczos(multiplyOp, psi_states, root=0, thresh=1.0e-09):
    """

    Get Operator eigenvector of root states for MPS by Lanczos method.
    This method is mainly used for
    time-independent diagonalization, such as improved relaxation.

    Args:
        multiplyOp (mps_cls.multiplyH_MPS_direct) : \
            operator for diagonalization.
        psi_states (list[np.ndarray]) : \
            Initial eigenvector guess of Lanczos method.
                ``np.ndarray`` part shape is (tau_{p-1}, j_p, tau_p).
        root (int) : The root of eigenvector. \
            If you want minimum eigenvalue states, this is ``0`` (default).\
                If you want maximum eigenvalue states, this is ``-1`` .
        thresh (float) : The threshold of Lanczos convergence tolerance.\
             Defaults to ``1.0e-9``.

    Returns:
        list[np.ndarray] : Eigenvector of root eigenvalue.\
                ``np.ndarray`` part shape is (tau_{p-1}, j_p, tau_p).

    """
    ndim = sum([x.size for x in psi_states])
    n_iter = min(ndim, 3000)

    alpha = np.array([])  # diagonal term
    beta = np.array([0.0])  # semi-diagonal term

    psi = multiplyOp.stack(psi_states)
    cveclist = [psi]
    for i_iter in range(n_iter + 1):
        """Tri-diagonal matrix by Lanczos algorithm"""
        trial_states = multiplyOp.split(cveclist[-1])
        sigvec_states = multiplyOp.dot(trial_states)
        sigvec = multiplyOp.stack(sigvec_states)

        alpha = np.append(alpha, np.inner(np.conj(cveclist[-1]), sigvec).real)
        sigvec -= cveclist[-1] * alpha[-1]
        if len(cveclist) >= 2:
            sigvec -= cveclist[-2] * beta[-1]
        beta = np.append(beta, scipy.linalg.norm(sigvec))
        sigvec /= beta[-1]

        # if alpha is real, eigh_tridiagonal is faster than eigh
        if np.all(np.isreal(alpha)):
            _, eigvecs = scipy.linalg.eigh_tridiagonal(alpha, beta[1:-1])
        else:
            mat = (
                np.diag(alpha, 0)
                + np.diag(beta[1:-1], -1)
                + np.diag(beta[1:-1], 1)
            )
            _, eigvecs = scipy.linalg.eig(mat)
        psi_next = (
            np.array(cveclist).T @ eigvecs[:, root].reshape(i_iter + 1, 1)
        ).reshape(ndim)
        if scipy.linalg.norm(beta[-1]) < EPS:
            next_psi_states = multiplyOp.split(psi_next)
            _Debug.niter_krylov[_Debug.site_now] = i_iter + 1
            return next_psi_states
        elif i_iter == 0:
            psi_next_sv = psi_next
        else:
            err = scipy.linalg.norm(psi_next - psi_next_sv)
            if err < thresh or i_iter == ndim:
                next_psi_states = multiplyOp.split(psi_next)
                _Debug.niter_krylov[_Debug.site_now] = i_iter + 1
                return next_psi_states
            psi_next_sv = psi_next
        cveclist.append(sigvec)
    raise ValueError("Lanczos Diagonalization is not converged in 3000 basis")


def _is_jax(x: np.ndarray | jax.Array) -> bool:
    return isinstance(x, jax.Array)


def _stack(
    states: list[np.ndarray] | list[jax.Array],
    multiplyOp: SplitStack,
    extend: bool = False,
) -> np.ndarray | jax.Array:
    return multiplyOp.stack(states, extend=extend)


def _split(
    vec: np.ndarray | jax.Array, multiplyOp: SplitStack, truncate: bool = False
) -> list[np.ndarray] | list[jax.Array]:
    return multiplyOp.split(vec, truncate=truncate)


def _dot(
    states: list[np.ndarray] | list[jax.Array], multiplyOp: SplitStack
) -> list[np.ndarray] | list[jax.Array]:
    return multiplyOp.dot(states)


def _norm(x: np.ndarray | jax.Array) -> float:
    if _is_jax(x):
        return float(jnp.linalg.norm(x))
    else:
        return float(np.linalg.norm(x))


def _H(
    psi_states: list[np.ndarray] | list[jax.Array], multiplyOp: SplitStack
) -> np.ndarray | jax.Array:
    return multiplyOp.stack(multiplyOp.dot(psi_states))


def _iter_info(
    psi_states: list[np.ndarray] | list[jax.Array],
) -> tuple[int, int, int]:
    maxsize = max([x.size for x in psi_states])
    ndim = min(maxsize, 20)
    n_warmup = min(
        maxsize, min(max(0, _Debug.niter_krylov[_Debug.site_now] - 2), 15)
    )
    return maxsize, ndim, n_warmup


def _normalize(
    v: np.ndarray | jax.Array,
) -> tuple[np.ndarray | jax.Array, float, np.ndarray | jax.Array]:
    if const.conserve_norm:
        return v, 1.0, jnp.ones(1) if _is_jax(v) else np.ones(1)
    else:
        if isinstance(v, jax.Array):
            β0_array = jnp.linalg.norm(v)
        else:
            β0_array = np.linalg.norm(v)
        β0 = float(β0_array)
        if β0 == 0.0:
            raise ValueError("Initial psi has zero norm.")
        v /= β0_array
        return v, β0, β0_array


def _rescale(
    v: np.ndarray | jax.Array, β0_array: np.ndarray | jax.Array | float
) -> np.ndarray | jax.Array:
    if const.conserve_norm:
        v /= _norm(v)
    else:
        v *= β0_array
    return v


@jax.jit
def _orth_step_jax_jittable_part(v: jax.Array, V: jax.Array):
    # V: (k, N), v: (N,)
    # h_i = <V_i, v>
    hcol = jnp.sum(jnp.conj(V) * v[jnp.newaxis, :], axis=1)  # (k,)
    # v' = v - Σ_i h_i V_i
    v -= jnp.sum(hcol[:, jnp.newaxis] * V, axis=0)  # (N,)
    beta = jnp.linalg.norm(v).real
    return hcol, v, V, beta


def _orth_step_jax(
    v: jax.Array, V: jax.Array, hessen: np.ndarray, ldim: int
) -> tuple[float, jax.Array, jax.Array, np.ndarray]:
    hcol, v, V, beta_jax = _orth_step_jax_jittable_part(v, V)
    beta = float(beta_jax)
    hessen[: ldim + 1, ldim] = np.asarray(hcol)
    if beta > EPS:
        v /= beta_jax
        V = stack_to_cvecs(v, V)
        if hessen.shape[0] > ldim + 1:
            hessen[ldim + 1, ldim] = beta
            # otherwise Krylov subspace = full space
    return (
        beta,
        v,
        V,
        hessen,
    )


def _orth_step_np(
    v: np.ndarray, V: np.ndarray, hessen: np.ndarray, ldim: int
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    hcol: np.ndarray = np.sum(np.conj(V) * v[np.newaxis, :], axis=1)  # (k,)
    v -= np.sum(hcol[:, np.newaxis] * V, axis=0)  # (N,)
    beta = float(np.linalg.norm(v))
    hessen[: ldim + 1, ldim] = hcol
    if beta > EPS:
        # if beta is sufficiently small, stack is not needed.
        v /= beta
        V = np.vstack([V, v])
        if hessen.shape[0] > ldim + 1:
            hessen[ldim + 1, ldim] = beta
    return beta, v, V, hessen


@jax.jit
def tensordot_jit(coeff: jax.Array, V: jax.Array):
    size = coeff.shape[0]
    return jnp.tensordot(coeff, V[:size,], axes=(0, 0))


@overload
def short_iterative_arnoldi(
    scale: float | complex,
    multiplyOp,
    psi_states: list[np.ndarray],
    thresh: float,
) -> list[np.ndarray]: ...


@overload
def short_iterative_arnoldi(
    scale: float | complex,
    multiplyOp,
    psi_states: list[jax.Array],
    thresh: float,
) -> list[jax.Array]: ...


def short_iterative_arnoldi(
    scale: float | complex,
    multiplyOp: SplitStack,
    psi_states: list[np.ndarray] | list[jax.Array],
    thresh: float,
) -> list[np.ndarray] | list[jax.Array]:
    """Approximate matrix exponential by \
        Short-time iterative Arnoldi algorithm

    If Hamiltonian is not Hermitian, such as complex potential, \
        this method should be used instead of \
        `short_iterative_lanczos`.

    compute exp[scale x mat] psi by Short-time Iterative Arnoldi

    Args:
        scale (complex) : The time step width. \
            (In relaxation this part is real float.)
        multiplyOp (mps_cls.multiplyH_MPS_direct) : operator for propagation. \
                In back propagation calculation, \
                this object is `mps_cls.multiplyK_MPS_direct`.
        psi_states (List[np.ndarray]) : MPS site function before propagation.\
                ``np.ndarray`` part shape is \
                (tau_{p-1}, j_p, tau_p) or (tau_{p-1}, tau_p).
        thresh (float) : The threshold of convergence tolerance.

    Returns:
        List[np.ndarray] : MPS site function after propagation.\
                ``np.ndarray`` part shape is \
                (tau_{p-1}, j_p, tau_p) or (tau_{p-1}, tau_p).

    - Vector side (cveclist, v, psi_next) can be GPU-executed if stored in JAX
    - Hessenberg (<=20x20) and its eigendecomposition use CPU (NumPy/SciPy)
    - First few iterations expand Krylov basis without eigendecomposition (same spirit as SIL)
    - Warning if Hessenberg is essentially tridiagonal
    """

    # --- JIT-able orthogonalization (SIL-style batch orthogonalization) ---

    def _maybe_warn_tridiag(Hsub: np.ndarray) -> None:
        if not const.pytest_enabled:
            return
        k = Hsub.shape[0]
        if k <= 2:
            return
        # Check above the first super-diagonal (third diagonal and beyond)
        mask = np.triu(np.ones_like(Hsub, dtype=bool), 2)
        off = np.linalg.norm(Hsub[mask])
        if off <= 1e-12:
            logger.warning(
                f"Arnoldi Hessenberg became tridiagonal {Hsub=}; consider using Lanczos for efficiency."
            )

    stack = partial(_stack, multiplyOp=multiplyOp)
    split = partial(_split, multiplyOp=multiplyOp)
    H = partial(_H, multiplyOp=multiplyOp)

    if const.use_jax:
        _orth_step = _orth_step_jax

        def tensordot(coeff: np.ndarray, V: jax.Array) -> jax.Array:
            _coeff = jnp.asarray(coeff, dtype=jnp.complex128)
            psi_next = tensordot_jit(_coeff, V)
            return psi_next
    else:
        _orth_step = _orth_step_np  # type:ignore

        def tensordot(coeff: np.ndarray, V: np.ndarray) -> np.ndarray:  # type: ignore
            psi_next = np.tensordot(coeff, V[: coeff.shape[0], :], axes=(0, 0))
            return psi_next

    # --- Parameters / warmup gating ---
    psi_next_sv = None
    maxsize, ndim, n_warmup = _iter_info(psi_states)
    # Hessenberg matrix `hessen` is a A (operator) in Krylov subspace V
    # `hessen` is always stored in CPU because its shape < (21, 20)
    # while `V` is stored in GPU (as needed) because its shape reaches (20, Nm^2)
    # where m is a bond dimension.
    hessen = np.zeros((ndim + 1, ndim), dtype=np.complex128)

    # --- Initial vector ---
    v0 = stack(psi_states, extend=True)
    v0, β0, β0_array = _normalize(v0)
    v: np.ndarray | jax.Array
    if const.use_jax:
        V = stack_to_cvecs(v0)
    else:
        V = np.vstack([v0])

    for ldim in range(ndim):
        # --- Arnoldi step: v = A @ v_k ---
        if ldim == 0:
            trial_states = psi_states
        else:
            # v = V[-1].
            trial_states = split(v, truncate=True)  # noqa F841
        v_l = H(trial_states)
        if not const.conserve_norm and ldim == 0:
            # v_1 = H (v_0 / |v_0|) = (H v_0) / |v_0|
            v_l /= β0_array

        # --- Orthogonalise ---
        beta, v, V, hessen = _orth_step(v_l, V, hessen, ldim)  # type: ignore

        # --- Breakdown: this is the only place that requires eigendecomposition ---
        is_converged = beta < EPS or ldim + 1 == maxsize
        if ldim < n_warmup and not is_converged:
            # --- Warmup: skip eigendecomposition and basis expansion ---
            continue

        if ldim == 0:
            psi_next = v0 * cmath.exp(scale * hessen[0, 0])
        else:
            # --- Ritz update on current subspace ---
            subH = hessen[: ldim + 1, : ldim + 1]
            eigvals, eigvecs = np.linalg.eig(subH)
            # Vinv = np.linalg.inv(eigvecs)
            # coeff = eigvecs @ (np.exp(scale * eigvals) * Vinv[:, 0])
            e0 = np.zeros(ldim + 1, dtype=subH.dtype)
            e0[0] = 1
            y = np.linalg.solve(eigvecs, e0)
            coeff = eigvecs @ (np.exp(scale * eigvals) * y)
            psi_next = tensordot(coeff, V)

        if is_converged:
            # When Krylov subspace is the same as the whole space,
            # calculated psi_next must be the exact solution.
            _Debug.niter_krylov[_Debug.site_now] = ldim + 1
            psi_next = _rescale(psi_next, β0_array)
            return split(psi_next)

        # --- Convergence check ---
        if psi_next_sv is None:
            psi_next_sv = psi_next
        else:
            err = _norm(psi_next - psi_next_sv)
            if err < thresh:
                _Debug.niter_krylov[_Debug.site_now] = ldim + 1
                psi_next = _rescale(psi_next, β0_array)
                _maybe_warn_tridiag(subH)
                return split(psi_next)
            psi_next_sv = psi_next

    raise ValueError(
        f"Short Iterative Arnoldi is not converged in 20 basis: {beta=} {maxsize=} {ndim=} {n_warmup=}. Try shorter time interval or larger Krylov"
    )


@overload
def short_iterative_lanczos(
    scale: float | complex,
    multiplyOp,
    psi_states: list[np.ndarray],
    thresh: float,
) -> list[np.ndarray]: ...


@overload
def short_iterative_lanczos(
    scale: float | complex,
    multiplyOp,
    psi_states: list[jax.Array],
    thresh: float,
) -> list[jax.Array]: ...


def short_iterative_lanczos(
    scale: float | complex,
    multiplyOp: SplitStack,
    psi_states: list[np.ndarray] | list[jax.Array],
    thresh: float,
) -> list[np.ndarray] | list[jax.Array]:
    """Approximate matrix exponential by \
        Short-time iterative Lanczos algorithm

    compute exp[scale x mat] psi by Short-time Iterative Arnoldi

    Args:
        scale (complex) : The time step width. \
            (In relaxation this part is real float.)
        multiplyOp (mps_cls.multiplyH_MPS_direct) : operator for propagation. \
                In back propagation calculation, \
                this object is `mps_cls.multiplyK_MPS_direct`. \
                Note that Hamiltonian must be (anti) Hermitian.
        psi_states (List[np.ndarray | jax.Array]) : MPS site function before propagation.\
                ``np.ndarray`` part shape is \
                (tau_{p-1}, j_p, tau_p) or (tau_{p-1}, tau_p).
        thresh (float) : The threshold of Lanczos convergence tolerance.

    Returns:
        List[np.ndarray | jax.Array] : MPS site function after propagation.\
                ``np.ndarray`` part shape is \
                (tau_{p-1}, j_p, tau_p) or (tau_{p-1}, tau_p).

    References:
        - Tae Jun Park and J. C. Light. Unitary quantum time evolution by iterative Lanczos reduction. \
          The Journal of Chemical Physics, Vol. 85, No. 10, pp. 5870–5876, November 1986.

    To Do:
        jax.lax.while_loop may achive acceleration.

    Psuedo Code:

    Prepare
    - initial state |ψ0> `psi_states`
    - Diagonal element of Hessenberg matrix α `alpha`
    - Semi-diagonal element of Hessenberg matrix β `beta`
    - Maxdimension of Krylov subspace n `ndim`
    - Effective Hamiltonian `H`
    - Step width Δt `scale`

    ```
    for k in 1..n:
        α[k-1] = <ψk|H|ψk>
        if k == 1:
            |ψk+1> = H|ψk> - α[k-1]|ψk>
        else:
            |ψk+1> = H|ψk> - α[k-1]|ψk> - β[k-1]|ψk-1>
        β[k-1] = |ψk+1|
        |ψk+1> = |ψk+1>/β[k-1]

        Λ, |φ> = eigh() of Tridiagonal matrix [α, β[:-1]]
        |ψ(Δt)> = Σ_l,m Σ_i,j |ψm><ψm| |φi><φi| exp(-iHΔt) |φj><φj| |ψl><ψl| |ψ0>
                = Σ_m Σ_i |ψi> <ψm|φi> exp(-iΛiΔt) <φi|ψ0> (∵ <φi|exp(-iHΔt)|φj> = δ_ij exp(-iΛiΔt), <ψl|ψ0> = δ_l0)

        if is_converged:
            return |ψ(Δt)>
    ```
    """
    stack = partial(_stack, multiplyOp=multiplyOp)
    split = partial(_split, multiplyOp=multiplyOp)
    H = partial(_H, multiplyOp=multiplyOp)

    psi_next_sv = None
    maxsize, ndim, n_warmup = _iter_info(psi_states)
    alpha = []  # diagonal term
    beta = []  # semi-diagonal term
    v0 = stack(psi_states, extend=True)
    v0, β0, β0_array = _normalize(v0)
    use_jax = isinstance(v0, jax.Array)
    alpha_is_real = True
    if use_jax:
        v0_conj: np.ndarray | jax.Array = jnp.conj(v0)
        V = stack_to_cvecs(v0)
    else:
        v0_conj = np.conj(v0)
        V = np.vstack([v0])

    for ldim in range(ndim):
        """Hessenberg matrix by Lanczos algorithm"""
        if ldim == 0:
            trial_states = psi_states
        else:
            trial_states = split(V[-1], truncate=True)

        v_l = H(trial_states)
        if not const.conserve_norm and ldim == 0:
            v_l /= β0_array

        if use_jax:
            if ldim == 0:
                β_l = None
            _, V, α_l, β_l = _next_sigvec_cvecs_alpha_beta(v_l, V, v0_conj, β_l)
            alpha.append(complex(α_l))
            beta.append(float(β_l))
        else:
            α_l = np.inner(v0_conj, v_l)
            alpha.append(complex(α_l))
            v_l -= V[-1] * α_l
            if ldim > 0:
                v_l -= V[-2] * β_l  # noqa: F821
            β_l = scipy.linalg.norm(v_l)
            beta.append(float(β_l))
            if beta[-1] >= EPS:
                v_l /= β_l
            else:
                # Krylov space exhausted
                v_l = np.empty_like(v_l)
            V = np.vstack([V, v_l])
        is_converged = beta[-1] < EPS or ldim + 1 == maxsize
        if alpha_is_real and np.abs(alpha[-1].imag) > 1e-12:
            alpha_is_real = False
            if const.conserve_norm:
                logger.warning(
                    f"{ldim=} {ndim=} {maxsize=} {is_converged=}"
                    + f"Diagonal element of Hessenberg matrix is complex, {alpha},"
                    + " it usually means that the Hamiltonian is not Hermitian. but you have set conserve_norm=True."
                )
        if ldim < n_warmup and not is_converged:
            continue

        if ldim == 0:
            psi_next = v0 * cmath.exp(scale * alpha[-1])
        else:
            # If jax.scipy.linalg.eigh_tridiagonal(eigvals_only=True) is available,
            # we will change whole loop implemented in JAX.
            if use_jax:
                # # This method is slow when using GPU
                # psi_next = _get_psi_next_jax(alpha, beta[:-1], cvecs, scale)
                if alpha_is_real:
                    Λ, Φ = scipy.linalg.eigh_tridiagonal(
                        np.real(alpha), beta[:-1]
                    )
                    expAΦᵗ0 = np.exp(scale * Λ) * np.conjugate(Φ).T[:, 0]
                else:
                    mat = (
                        np.diag(alpha, 0)
                        + np.diag(beta[:-1], -1).astype(np.complex128)
                        + np.diag(beta[:-1], 1).astype(np.complex128)
                    )
                    Λ, Φ = scipy.linalg.eig(mat)
                    # expAΦᵗ0 = np.exp(scale * Λ) * np.linalg.inv(Φ)[:, 0]
                    e0 = np.zeros(ldim + 1, dtype=mat.dtype)
                    e0[0] = 1
                    y = np.linalg.solve(Φ, e0)
                    expAΦᵗ0 = np.exp(scale * Λ) * y
                # eigvec_expLU = np.einsum("ij,j->i", eigvecs, expLU)
                ΦexpAΦᵗ0 = Φ @ expAΦᵗ0
                # psi_next = jnp.einsum(
                #    "kj,k->j",
                #    cvecs[:-1, :],
                #    jnp.array(eigvec_expLU, dtype=jnp.complex128),
                # )
                psi_next = jnp.dot(
                    jnp.array(ΦexpAΦᵗ0, dtype=jnp.complex128), V[:-1, :]
                )
            else:
                if alpha_is_real:
                    Λ, Φ = scipy.linalg.eigh_tridiagonal(
                        np.real(alpha), beta[:-1]
                    )
                    expAΦᵗ0 = np.exp(scale * Λ) * np.conjugate(Φ).T[:, 0]
                else:
                    mat = (
                        np.diag(alpha, 0)
                        + np.diag(beta[:-1], -1).astype(np.complex128)
                        + np.diag(beta[:-1], 1).astype(np.complex128)
                    )
                    Λ, Φ = scipy.linalg.eig(mat)
                    # expAΦᵗ0 = np.exp(scale * Λ) * np.linalg.inv(Φ)[:, 0]
                    e0 = np.zeros(ldim + 1, dtype=mat.dtype)
                    e0[0] = 1
                    y = np.linalg.solve(Φ, e0)
                    expAΦᵗ0 = np.exp(scale * Λ) * y
                # eigvec_expLU = np.einsum("ij,j->i", eigvecs, expLU)
                ΦexpAΦᵗ0 = Φ @ expAΦᵗ0
                # psi_next = np.einsum("kj,k->j", cvecs[:-1, :], eigvec_expLU)
                psi_next = np.dot(ΦexpAΦᵗ0, V[:-1, :])
        if is_converged:
            # When Krylov subspace is the same as the whole space,
            # calculated psi_next must be the exact solution.
            _Debug.niter_krylov[_Debug.site_now] = ldim + 1
            psi_next = _rescale(psi_next, β0_array)
            return split(psi_next)
        elif psi_next_sv is None:
            psi_next_sv = psi_next
        else:
            err = _norm(psi_next - psi_next_sv)
            if err < thresh:
                _Debug.niter_krylov[_Debug.site_now] = ldim + 1
                psi_next = _rescale(psi_next, β0_array)
                return split(psi_next)
            psi_next_sv = psi_next
    raise ValueError(
        f"Short Iterative Lanczos is not converged in {ldim} basis when {maxsize=}. Try shorter time interval."
    )


@jax.jit
def stack_to_cvecs(v: jax.Array, V: jax.Array | None = None) -> jax.Array:
    if V is None:
        return jnp.vstack([v])
    else:
        return jnp.vstack([V, v])


@jax.jit
def _next_sigvec_cvecs_alpha_beta(
    v: jax.Array,
    V: jax.Array,
    psi_conj: jax.Array,
    beta: float | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    alpha = jnp.inner(psi_conj, v)
    if beta is None:
        v -= V[-1] * alpha
    else:
        v -= V[-1] * alpha + V[-2] * beta
    beta = jnp.linalg.norm(v).real
    v /= beta
    V = stack_to_cvecs(v, V)
    return v, V, alpha.astype(jnp.complex128), beta.astype(jnp.float64)  # type: ignore


@jax.jit
def _get_psi_next_jax(
    a: list[float], b: list[float], cvecs: jax.Array, scale: float | complex
) -> jax.Array:
    # This method is slow when using GPU
    _a = jnp.array(a, dtype=jnp.float64)
    _b = jnp.array(b, dtype=jnp.float64)
    mat = jnp.diag(_a, 0) + jnp.diag(_b, -1) + jnp.diag(_b, 1)
    eigvals, eigvecs = jax.scipy.linalg.eigh(
        mat,
        # lower=True,
        eigvals_only=False,
    )
    expLU = jnp.exp(scale * eigvals) * jnp.conjugate(eigvecs).T[:, 0]
    eigvec_expLU = jnp.einsum("ij,j->i", eigvecs, expLU)
    psi_next = jnp.einsum("kj,k->j", cvecs[:-1, :], eigvec_expLU)
    return psi_next
