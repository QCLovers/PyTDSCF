"""
Integrator module for MPS time evolution & diagonalization.
"""

from __future__ import annotations

import cmath
from typing import overload

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg

from pytdscf._const_cls import const
from pytdscf._contraction import SplitStack
from pytdscf._helper import _Debug


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

        _, eigvecs = scipy.linalg.eigh_tridiagonal(alpha, beta[1:-1])
        psi_next = (
            np.array(cveclist).T @ eigvecs[:, root].reshape(i_iter + 1, 1)
        ).reshape(ndim)
        if scipy.linalg.norm(beta[-1]) < 1e-15:
            next_psi_states = multiplyOp.split(psi_next)
            _Debug.niter_krylov[_Debug.site_now] = i_iter
            return next_psi_states
        elif i_iter == 0:
            psi_next_sv = psi_next
        else:
            err = scipy.linalg.norm(psi_next - psi_next_sv)
            if err < thresh or i_iter == ndim:
                next_psi_states = multiplyOp.split(psi_next)
                _Debug.niter_krylov[_Debug.site_now] = i_iter
                return next_psi_states
            psi_next_sv = psi_next
        cveclist.append(sigvec)
    raise ValueError("Lanczos Diagonalization is not converged in 3000 basis")


def short_iterative_arnoldi(scale, multiplyOp, psi_states, thresh):
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

    """
    ndim = min(sum([x.size for x in psi_states]), 20)
    # short iterative lanczos should converge in a few steps
    hessen = np.zeros((ndim + 1, ndim + 1), dtype=complex)
    psi = multiplyOp.stack(psi_states)
    if const.space == "liouville":
        β0 = np.linalg.norm(psi).item()
        psi /= β0
    else:
        β0 = 1.0
    cveclist = [psi]
    for ldim in range(ndim + 1):
        """Hessenberg matrix by Arnoldi algorithm"""
        trial_states = multiplyOp.split(cveclist[-1])
        sigvec_states = multiplyOp.dot(trial_states)
        sigvec = multiplyOp.stack(sigvec_states)
        for icol, cvec in enumerate(cveclist):
            hessen[icol, ldim] = np.inner(np.conj(cvec), sigvec)
            sigvec -= cvec * hessen[icol, ldim]

        subH = hessen[: ldim + 1, : ldim + 1]
        eigvals, eigvecs = scipy.linalg.eig(subH)
        # Naive implementation
        psi_next = sum(
            [
                cvec
                * np.sum(
                    eigvecs[k, :]
                    * np.exp(scale * eigvals)
                    * scipy.linalg.inv(eigvecs)[:, 0]
                )
                for k, cvec in enumerate(cveclist)
            ]
        )

        if scipy.linalg.norm(sigvec) < 1e-15:
            _Debug.niter_krylov[_Debug.site_now] = ldim
            if const.space == "liouville":
                psi_next *= β0
            return multiplyOp.split(psi_next)
        elif ldim == 0:
            psi_next_sv = psi_next
        else:
            err = scipy.linalg.norm(psi_next - psi_next_sv)
            if err < thresh:
                _Debug.niter_krylov[_Debug.site_now] = ldim
                if const.space == "liouville":
                    psi_next *= β0
                return multiplyOp.split(psi_next)
            psi_next_sv = psi_next
        hessen[ldim + 1, ldim] = scipy.linalg.norm(sigvec)
        cveclist.append(sigvec / hessen[ldim + 1, ldim])
    raise ValueError("Short Iterative Arnoldi is not converged in 20 basis")


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

    def norm(psi: np.ndarray | jax.Array) -> float:
        if isinstance(psi, jax.Array):
            return jnp.linalg.norm(psi).item()
        else:
            return np.linalg.norm(psi).item()

    def H(
        psi_states: list[np.ndarray] | list[jax.Array],
    ) -> np.ndarray | jax.Array:
        return multiplyOp.stack(multiplyOp.dot(psi_states))

    psi_next_sv = None
    nstep_skip_conv_check = min(
        max(0, _Debug.niter_krylov[_Debug.site_now] - 2),
        15,
    )

    maxsize = sum([x.size for x in psi_states])
    ndim = min(maxsize, 20)
    # short iterative lanczos should converge in a few steps
    alpha = []  # diagonal term
    beta = []  # semi-diagonal term
    v1 = multiplyOp.stack(psi_states, extend=True)
    use_jax = isinstance(v1, jax.Array)
    if const.space == "liouville":
        β0 = norm(v1)
        v1 /= β0
    else:
        β0 = 1.0
    if use_jax:
        v1_conj: np.ndarray | jax.Array = jnp.conj(v1)
        V = stack_to_cvecs(v1)
    else:
        v1_conj = np.conj(v1)
        V = np.vstack([v1])

    for ldim in range(ndim + 1):
        """Hessenberg matrix by Lanczos algorithm"""
        if ldim == 0:
            trial_states = psi_states
        else:
            trial_states = multiplyOp.split(V[-1], truncate=True)

        v_l = H(trial_states)
        if const.space == "liouville" and ldim == 0:
            v_l /= β0

        if use_jax:
            if ldim == 0:
                β_l = None
            v_l, V, α_l, β_l = _next_sigvec_cvecs_alpha_beta(
                v_l, V, v1_conj, β_l
            )
            alpha.append(float(α_l))
            beta.append(float(β_l))
        else:
            α_l = np.inner(v1_conj, v_l).real
            alpha.append(float(α_l))
            v_l -= V[-1] * α_l
            if ldim > 0:
                v_l -= V[-2] * β_l  # noqa: F821
            β_l = scipy.linalg.norm(v_l)
            beta.append(float(β_l))
            if beta[-1] >= 1e-15:
                v_l /= β_l
            else:
                # Krylov space exhausted
                v_l = np.empty_like(v_l)
            V = np.vstack([V, v_l])
        if is_converged := (beta[-1] < 1e-15):
            pass
        else:
            if ldim < min(nstep_skip_conv_check, maxsize):
                continue

        if ldim == 0:
            psi_next = v1 * cmath.exp(scale * alpha[-1])
        else:
            # If jax.scipy.linalg.eigh_tridiagonal(eigvals_only=True) is available,
            # we will change whole loop implemented in JAX.
            if use_jax:
                # # This method is slow when using GPU
                # psi_next = _get_psi_next_jax(alpha, beta[:-1], cvecs, scale)
                Λ, Φ = scipy.linalg.eigh_tridiagonal(alpha, beta[:-1])
                expAΦᵗ0 = np.exp(scale * Λ) * np.conjugate(Φ).T[:, 0]
                # eigvec_expLU = np.einsum("ij,j->i", eigvecs, expLU)
                ΦexpAΦᵗ0 = scipy.linalg.blas.zgemv(
                    alpha=1.0,
                    a=Φ,
                    x=expAΦᵗ0,
                )
                # psi_next = jnp.einsum(
                #    "kj,k->j",
                #    cvecs[:-1, :],
                #    jnp.array(eigvec_expLU, dtype=jnp.complex128),
                # )
                psi_next = jnp.dot(
                    jnp.array(ΦexpAΦᵗ0, dtype=jnp.complex128), V[:-1, :]
                )

            else:
                Λ, Φ = scipy.linalg.eigh_tridiagonal(alpha, beta[:-1])
                expAΦᵗ0 = np.exp(scale * Λ) * np.conjugate(Φ).T[:, 0]
                # eigvec_expLU = np.einsum("ij,j->i", eigvecs, expLU)
                # NOTE: If scipy backend is MKL, numpy and mpi4py align with MKL.
                #       Otherwise, parallelization will inefficient.
                #       We recommend to use OpenBLAS and OpenMPI.
                ΦexpAΦᵗ0 = scipy.linalg.blas.zgemv(
                    alpha=1.0,
                    a=Φ,
                    x=expAΦᵗ0,
                )
                # psi_next = np.einsum("kj,k->j", cvecs[:-1, :], eigvec_expLU)
                psi_next = np.dot(ΦexpAΦᵗ0, V[:-1, :])
        if is_converged:
            _Debug.niter_krylov[_Debug.site_now] = ldim
            if const.space == "liouville":
                psi_next *= β0
            return multiplyOp.split(psi_next)
        elif ldim == maxsize:
            # When Krylov subspace is the same as the whole space,
            # calculated psi_next must be the exact solution.
            _Debug.niter_krylov[_Debug.site_now] = ldim
            if const.space == "liouville":
                psi_next *= β0
            else:
                psi_next /= norm(psi_next)
            return multiplyOp.split(psi_next)

        if psi_next_sv is None:
            psi_next_sv = psi_next
        else:
            err = norm(psi_next - psi_next_sv)
            if err < thresh:
                _Debug.niter_krylov[_Debug.site_now] = ldim
                if const.space == "liouville":
                    psi_next *= β0
                else:
                    # |C| should be 1.0
                    psi_next /= norm(psi_next)
                return multiplyOp.split(psi_next)
            psi_next_sv = psi_next
    raise ValueError(
        f"Short Iterative Lanczos is not converged in {ldim} basis when {maxsize=}. Try shorter time interval."
    )


@jax.jit
def stack_to_cvecs(psi: jax.Array, cvecs: jax.Array | None = None) -> jax.Array:
    if cvecs is None:
        return jnp.vstack([psi])
    else:
        return jnp.vstack([cvecs, psi])


@jax.jit
def _next_sigvec_cvecs_alpha_beta(
    sigvec: jax.Array,
    cvecs: jax.Array,
    psi_conj: jax.Array,
    beta: float | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    alpha = jnp.inner(psi_conj, sigvec).real
    if beta is None:
        sigvec = sigvec - cvecs[-1] * alpha
    else:
        sigvec = sigvec - cvecs[-1] * alpha - cvecs[-2] * beta
    beta = jnp.linalg.norm(sigvec).real
    sigvec /= beta
    cvecs = stack_to_cvecs(sigvec, cvecs)
    return sigvec, cvecs, alpha.astype(jnp.float64), beta.astype(jnp.float64)  # type: ignore


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
