"""
Integrator module for MPS time evolution & diagonalization.
"""

from __future__ import annotations

import cmath

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg

from pytdscf._helper import _Debug


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
    _Debug.ncall_krylov += 1

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
            _Debug.niter_krylov += i_iter
            return next_psi_states
        elif i_iter == 0:
            psi_next_sv = psi_next
        else:
            err = scipy.linalg.norm(psi_next - psi_next_sv)
            if err < thresh or i_iter == ndim:
                next_psi_states = multiplyOp.split(psi_next)
                _Debug.niter_krylov += i_iter
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
    _Debug.ncall_krylov += 1
    ndim = min(sum([x.size for x in psi_states]), 20)
    # short iterative lanczos should converge in a few steps
    hessen = np.zeros((ndim + 1, ndim + 1), dtype=complex)
    psi = multiplyOp.stack(psi_states)
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
            _Debug.niter_krylov += ldim
            return multiplyOp.split(psi_next)
        elif ldim == 0:
            psi_next_sv = psi_next
        else:
            err = scipy.linalg.norm(psi_next - psi_next_sv)
            if err < thresh:
                _Debug.niter_krylov += ldim
                return multiplyOp.split(psi_next)
            psi_next_sv = psi_next
        hessen[ldim + 1, ldim] = scipy.linalg.norm(sigvec)
        cveclist.append(sigvec / hessen[ldim + 1, ldim])
    raise ValueError("Short Iterative Arnoldi is not converged in 20 basis")


def short_iterative_lanczos(
    scale: float | complex,
    multiplyOp,
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

    To Do:
        jax.lax.while_loop may achive acceleration.

    """
    _Debug.ncall_krylov += 1
    psi_next_sv = None
    if _Debug.ncall_krylov == 0:
        nstep_skip_conv_check = 0
    else:
        nstep_skip_conv_check = min(
            max(0, int(_Debug.niter_krylov / _Debug.ncall_krylov) - 2), 15
        )

    ndim = min(sum([x.size for x in psi_states]), 20)
    # short iterative lanczos should converge in a few steps
    alpha = []  # diagonal term
    beta = []  # semi-diagonal term
    psi = multiplyOp.stack(psi_states)
    if use_jax := isinstance(psi, jax.Array):
        psi_conj = jnp.conj(psi)
        cvecs = stack_to_cvecs(psi)
    else:
        psi_conj = np.conj(psi)
        cvecs = np.vstack([psi])

    for ldim in range(ndim + 1):
        """Hessenberg matrix by Lanczos algorithm"""
        if ldim == 0:
            trial_states = psi_states
        else:
            trial_states = multiplyOp.split(cvecs[-1])

        sigvec_states = multiplyOp.dot(trial_states)
        sigvec = multiplyOp.stack(sigvec_states)

        if use_jax:
            if ldim == 0:
                beta_l = None
            sigvec, cvecs, alpha_l, beta_l = _next_sigvec_cvecs_alpha_beta(
                sigvec, cvecs, psi_conj, beta_l
            )
            alpha.append(float(alpha_l))
            beta.append(float(beta_l))
        else:
            alpha_l = np.inner(psi_conj, sigvec).real
            alpha.append(float(alpha_l))
            sigvec -= cvecs[-1] * alpha_l
            if ldim > 0:
                sigvec -= cvecs[-2] * beta_l  # noqa: F821
            beta_l = scipy.linalg.norm(sigvec)
            beta.append(float(beta_l))
            sigvec /= beta_l
            cvecs = np.vstack([cvecs, sigvec])
        if is_converged := (beta[-1] < 1e-15):
            pass
        else:
            if ldim < nstep_skip_conv_check:
                continue

        if ldim == 0:
            psi_next = psi * cmath.exp(scale * alpha[-1])
        else:
            # Calculation of eigenvectors is not implemented in JAX
            eigvals, eigvecs = scipy.linalg.eigh_tridiagonal(alpha, beta[:-1])
            expLU = np.exp(scale * eigvals) * np.conjugate(eigvecs).T[:, 0]
            eigvec_expLU = np.einsum("ij,j->i", eigvecs, expLU)
            if use_jax:
                eigvec_expLU = jnp.array(eigvec_expLU, dtype=jnp.complex128)
                psi_next = jnp.einsum("kj,k->j", cvecs[:-1, :], eigvec_expLU)
            else:
                psi_next = np.einsum("kj,k->j", cvecs[:-1, :], eigvec_expLU)
        if is_converged:
            _Debug.niter_krylov += ldim
            return multiplyOp.split(psi_next)

        if psi_next_sv is None:
            psi_next_sv = psi_next
        else:
            if use_jax:
                err = jnp.linalg.norm(psi_next - psi_next_sv)
            else:
                err = scipy.linalg.norm(psi_next - psi_next_sv)
            if err < thresh:
                _Debug.niter_krylov += ldim
                # |C| should be 1.0
                if use_jax:
                    psi_next /= jnp.linalg.norm(psi_next)
                else:
                    psi_next /= np.linalg.norm(psi_next)
                return multiplyOp.split(psi_next)
            psi_next_sv = psi_next
    raise ValueError(
        f"Short Iterative Lanczos is not converged in {ldim} basis."
        + "Try shoter time interval."
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
