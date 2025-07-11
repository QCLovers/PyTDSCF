"""
Boson basis module
"""

import numpy as np
from numpy.typing import NDArray


class Boson:
    """
    Boson basis class.

    Args:
        nstate (int): Number of basis states :math:`|0\rangle, \\; |1\rangle, \\dots, |n_{\text{state}}-1\rangle`.
    """

    nstate: int
    names: list[str]

    def __init__(self, nstate: int) -> None:
        self.nstate = nstate

    def get_creation_matrix(self, margin: int = 0) -> NDArray[np.float64]:
        r"""
        Returns the creation matrix for the boson basis

        .. math::
            b^\dagger |n> = \sqrt{n+1} |n+1>

        Examples:
           >>> boson = Boson(3)
           >>> boson.get_creation_matrix()
           array([[0., 0., 0.],
                  [1., 0., 0.],
                  [1.41421356, 0., 0.]])
           >>> state = np.array([[1.0],
                                 [1.0],
                                 [0.0]])
           >>> np.allclose(boson.get_creation_matrix() @ state, np.array([[0.0], [1.0], [math.sqrt(2)]]))
           True
        """
        return self.get_annihilation_matrix(margin=margin).T

    def get_annihilation_matrix(self, margin: int = 0) -> NDArray[np.float64]:
        r"""
        Returns the annihilation matrix for the boson basis

        .. math::
            b |n> = \sqrt{n} |n-1>

        Examples:
           >>> exciton = Boson(4)
           >>> exciton.get_annihilation_matrix()
           array([[0., 1., 0., 0.],
                  [0., 0., 1.41421356, 0.],
                  [0., 0., 0., 1.73205081],
                  [0., 0., 0., 0.]])
           >>> state = np.array([[1.0],
                                 [1.0],
                                 [1.0],
                                 [0.0]])
           >>> np.allclose(exciton.get_annihilation_matrix() @ state, np.array([[0.0], [1.0], [math.sqrt(2)], [math.sqrt(3)]]))
           True
        """
        # mat = np.zeros((self.nstate, self.nstate), dtype=np.float64)
        # for i in range(self.nstate - 1):
        #    mat[i, i+1] = np.sqrt(i + 1)
        mat = np.diag(np.sqrt(np.arange(1, self.nstate + margin)), 1)
        return mat

    def get_number_matrix(self) -> NDArray[np.float64]:
        r"""
        Returns the number matrix for the boson basis

        .. math::
            b^\dagger b |n> = n |n>

        Examples:
           >>> boson = Boson(3)
           >>> boson.get_number_matrix()
           array([[0., 0., 0.],
                  [0., 1., 0.],
                  [0., 0., 2.]])
           >>> state = np.array([[1.0],
                                 [1.0],
                                 [0.0]])
           >>> np.allclose(boson.get_number_matrix() @ state, np.array([[0.0], [1.0], [0.0]]))
           True
        """
        mat = np.diag(np.arange(self.nstate))
        return mat

    def get_q_matrix(self, margin: int = 0) -> NDArray[np.float64]:
        r"""
        Returns the position operator (``q``) matrix for the boson basis.

        .. math::

            q = \frac{1}{\sqrt{2}}\,(b + b^\dagger)

        """
        return (
            1
            / np.sqrt(2)
            * (
                self.get_creation_matrix(margin=margin)
                + self.get_annihilation_matrix(margin=margin)
            )
        )

    def get_p_matrix(self, margin: int = 0) -> NDArray[np.complex128]:
        r"""
        Returns the p matrix for the boson basis

        .. math::
            p = 1/\sqrt{2} i (b - b^\dagger) = i/\sqrt{2} (b^\dagger - b)

        """
        return (
            1
            / np.sqrt(2)
            * 1j
            * (
                self.get_creation_matrix(margin=margin)
                - self.get_annihilation_matrix(margin=margin)
            )
        )

    def get_q2_matrix(self) -> NDArray[np.float64]:
        r"""
        Returns the :math:`q^2` matrix for the boson basis.

        .. math::
            q^2 = \frac{1}{2}\left(b^\dagger b + b b^\dagger + b^{\dagger 2} + b^2\right)

        """
        b = self.get_annihilation_matrix(margin=1)
        bd = self.get_creation_matrix(margin=1)
        bd_b = bd + b
        return 1 / 2 * (bd_b @ bd_b)[:-1, :-1]

    def get_p2_matrix(self) -> NDArray[np.float64]:
        r"""
        Returns the p^2 matrix for the boson basis

        .. math::
           p^2 = -1/2 (b^\dagger b + b b^\dagger - b^\dagger ^2 - b^2)

        """
        b = self.get_annihilation_matrix(margin=1)
        bd = self.get_creation_matrix(margin=1)
        bd_b = bd - b
        return -1 / 2 * (bd_b @ bd_b)[:-1, :-1]

    @property
    def nprim(self) -> int:
        return self.nstate

    def __len__(self) -> int:
        return self.nstate


if __name__ == "__main__":
    boson = Boson(3)
    state0 = np.array([[1.0], [1.0], [0.0]])
    state1 = np.array([[0.0], [1.0], [np.sqrt(2)]])
    state2 = np.array([[1.0], [0.0], [0.0]])
    state3 = np.array([[0.0], [0.0], [np.sqrt(2)]])
    state4 = np.array([[1.0], [2.0], [0.0]])
    cmat = boson.get_creation_matrix()
    amat = boson.get_annihilation_matrix()
    nmat = boson.get_number_matrix()
    np.testing.assert_allclose(cmat @ amat, nmat)
    for state, mat, ans in zip(
        [state0, state0, state1, state1],
        [cmat, amat, cmat, amat],
        [state1, state2, state3, state4],
        strict=False,
    ):
        np.testing.assert_allclose(mat @ state, ans)
    boson = Boson(5)
    cmat = boson.get_creation_matrix()
    amat = boson.get_annihilation_matrix()
    nmat = boson.get_number_matrix()
    pmat = boson.get_p_matrix()
    qmat = boson.get_q_matrix()
    np.testing.assert_allclose(1 / np.sqrt(2) * (qmat + 1j * pmat), amat)
    np.testing.assert_allclose(1 / np.sqrt(2) * (qmat - 1j * pmat), cmat)
    pmat = boson.get_p_matrix(margin=1)
    qmat = boson.get_q_matrix(margin=1)
    # [p,q] = -i
    np.testing.assert_allclose(
        (pmat @ qmat)[:-1, :-1] - (qmat @ pmat)[:-1, :-1],
        -1.0j * np.eye(boson.nstate),
    )
    p2 = boson.get_p2_matrix()
    q2 = boson.get_q2_matrix()
    np.testing.assert_allclose(p2, (pmat @ pmat)[:-1, :-1])
    np.testing.assert_allclose(q2, (qmat @ qmat)[:-1, :-1])
    np.testing.assert_allclose(
        1 / 2 * (p2 + q2), nmat + 1 / 2 * np.eye(boson.nstate)
    )
