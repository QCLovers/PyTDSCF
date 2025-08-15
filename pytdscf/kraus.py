"""
Kraus operator functions for Lindblad master equation
"""

from math import isqrt
from typing import Literal, overload

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm


def lindblad_to_kraus(
    Lops: list[np.ndarray],
    dt: float,
    backend: Literal["numpy", "jax"] = "numpy",
) -> np.ndarray | jnp.ndarray:
    """
    Convert set of Lindblad operators {L_j} to a Kraus operator set {B_q}

    exp(D dt) = ∑_q=1^k B_q ⊗ B_q*

    where

    D = ∑_j [L_j⊗L_j* - 1/2 (L_j†L_j ⊗ I + I⊗L_j⊤L_j*)]

    See DOI: https://doi.org/10.1103/PhysRevLett.116.237201

    Args:
        Lops (list[np.ndarray]): Lindblad operators
        dt (float): time step

    Returns:
        np.ndarray | jnp.ndarray: Kraus operator tensor
    """
    assert all(L.ndim == 2 for L in Lops)
    assert all(L.shape[0] == L.shape[1] for L in Lops)
    assert dt > 0

    L = Lops.pop()
    Ldag = L.conj().T
    I = np.eye(L.shape[0])
    D = np.kron(L, L.conj()) - 1 / 2 * (
        np.kron(Ldag @ L, I) + np.kron(I, L.T @ L.conj())
    )
    if np.allclose(D.imag, 0):
        D = D.real
    while Lops:
        L = Lops.pop()
        Ldag = L.conj().T
        _D = np.kron(L, L.conj()) - 1 / 2 * (
            np.kron(Ldag @ L, I) + np.kron(I, L.T @ L.conj())
        )
        if np.allclose(_D.imag, 0):
            _D = _D.real
        else:
            D = D.astype(complex)
        D += _D

    dissipator = expm(D * dt)
    # eigenvalues of dissipator should be positive
    assert np.all(np.linalg.eigvalsh(dissipator) > -1e-14)

    # Kraus operators
    def supergate_to_kraus(G, d, tol=1e-14):
        S4 = G.reshape(d, d, d, d, order="F")  # S[α,β,μ,ν]
        J = np.transpose(S4, (0, 2, 1, 3)).reshape(
            d * d, d * d, order="F"
        )  # J[(αμ),(βν)] = S[α,β,μ,ν]
        J = (J + J.conj().T) / 2  # hermitize
        if np.allclose(J, J.conj().T):
            w, V = np.linalg.eigh(J)
        else:
            w, V = np.linalg.eig(J)
        kraus = []
        for lam, v in zip(w, V.T, strict=True):
            lam = lam.real
            if lam > tol:
                kraus.append(np.sqrt(lam) * v.reshape(d, d, order="F"))
        return kraus  # satisfies sum_q B_q† B_q ≈ I

    Bs = supergate_to_kraus(dissipator, isqrt(dissipator.shape[0]))

    # Confirm exp(D dt) = ∑_q=1^k B_q ⊗ B_q*
    np.testing.assert_allclose(
        dissipator,
        np.sum([np.kron(B, B.conj()) for B in Bs], axis=0),
        atol=1e-14,
    )
    """
    TN diagram of Kraus operators

      d
      |
    --B
    | |
    k d

    tensor shape (k, d, d)
    """
    k = len(Bs)
    d = Bs[0].shape[0]
    B: np.ndarray | jax.Array
    match backend:
        case "numpy":
            Bs = [np.array(B, dtype=np.complex128) for B in Bs]
            B = np.stack(Bs, axis=0)
        case "jax":
            Bs = [jnp.array(B, dtype=jnp.complex128) for B in Bs]
            B = jnp.stack(Bs, axis=0)
        case _:
            raise ValueError(f"Invalid backend: {backend}")
    assert B.shape == (k, d, d), (
        f"Kraus operator shape mismatch: {B.shape} != ({k}, {d}, {d})"
    )
    return B


@overload
def kraus_contract(B: np.ndarray, core: np.ndarray) -> np.ndarray: ...


@overload
def kraus_contract(B: jax.Array, core: np.ndarray) -> jax.Array: ...


def kraus_contract(
    B: np.ndarray | jax.Array, core: np.ndarray | jax.Array
) -> np.ndarray | jax.Array:
    if isinstance(B, np.ndarray) and isinstance(core, np.ndarray):
        return _kraus_contract_np(B, core)
    elif isinstance(B, jax.Array) and isinstance(core, jax.Array):
        return _kraus_contract_jax(B, core)
    else:
        raise ValueError(f"Invalid backend: {type(B)=} while {type(core)=}")


def _kraus_contract_np(B: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
      x
      |
    k-B
      |
      d

      dK
      |
    m-A-n
    """
    k, d, x = B.shape
    assert d == x
    m, dK, n = A.shape
    assert dK % d == 0, f"Kraus contract: dK={dK} must be divisible by d={d}"
    K = dK // d
    # 1. reshape A to (m, d, K, m)
    A = A.reshape(m, d, K, n)
    """
      x
      |
    k-B
      |
      d

      d
      |
    m-A-n
      |
      K
    """
    # 2. contract "d" legs of B and A
    C = np.einsum("kxd,mdKn->mnxkK", B, A)
    r"""
      x
      |
    m-C-n
      |\
      k K
    """

    # 3. reshape C to (m, n, x, kK)
    C = C.reshape(m * n * x, k * K)
    """
    mnx-C-kK
    """

    # 4. SVD of C
    U, S, _ = np.linalg.svd(C, full_matrices=False)
    """
    mnx-U-kK kK-S-kK kK-Vh-kK
    """
    # 5. truncate singular values
    S = S[:K]
    U = U[:, :K]
    """
    mnx-U-K K-S-K
    """
    # 6. concatenate U and S as new A
    A = U * S[np.newaxis, :]
    np.testing.assert_allclose(A, U @ np.diag(S))
    """
    mnx-A-K
    """
    # 7. reshape A to (m, n, xK) and swap indices n and xK
    A = A.reshape(m, n, x * K).swapaxes(1, 2)
    """
      xK
      |
    m-A-n
    """
    return A


@jax.jit
def _kraus_contract_jax(B: jax.Array, A: jax.Array) -> jax.Array:
    """
      x
      |
    k-B
      |
      d

      dK
      |
    m-A-n
    """
    k, d, x = B.shape
    m, dK, n = A.shape
    K = dK // d

    # 1. reshape A to (m, d, K, n)
    A = A.reshape(m, d, K, n)
    """
      x
      |
    k-B
      |
      d

      d
      |
    m-A-n
      |
      K
    """
    # 2. contract "d" legs of B and A
    C = jnp.einsum("kxd,mdKn->mnxkK", B, A)
    r"""
      x
      |
    m-C-n
      |\
      k K
    """

    # 3. reshape C to (m, n, x, kK)
    C = C.reshape(m * n * x, k * K)
    """
    mnx-C-kK
    """

    # 4. SVD of C
    U, S, _ = jnp.linalg.svd(C, full_matrices=False)
    """
    mnx-U-kK kK-S-kK kK-Vh-kK
    """
    # 5. truncate singular values
    S = S[:K]
    U = U[:, :K]
    """
    mnx-U-K K-S-K
    """
    # 6. concatenate U and S as new A
    A = U * S[jnp.newaxis, :]
    """
    mnx-A-K
    """
    # 7. reshape A to (m, n, xK) and swap indices n and xK
    A = A.reshape(m, n, x * K).swapaxes(1, 2)
    """
      xK
      |
    m-A-n
    """
    return A


def trace_kraus_dim(rdm: np.ndarray, d: int):
    r"""
    dK    d   K      d
    |      \ /|      |
    C   =>  C |  =>  C
    |      / \|      |
    dK    d   K      d
    """
    dK = rdm.shape[-1]
    assert dK % d == 0, (
        f"Kraus dimension reduction: dK={dK} must be divisible by d={d}"
    )
    K = dK // d
    if rdm.ndim == 2:
        rdm = rdm.reshape(d, K, d, K)
        rdm = np.einsum("dKxK->dx", rdm)
    else:
        assert rdm.ndim == 3, f"rdm.ndim={rdm.ndim} must be 2 or 3"
        rdm = rdm.reshape(-1, d, K, d, K)
        rdm = np.einsum("tdKxK->tdx", rdm)

    return rdm
