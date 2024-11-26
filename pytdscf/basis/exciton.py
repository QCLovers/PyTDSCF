"""
Exciton basis module
"""
import numpy as np
from numpy.typing import NDArray


class Exciton:
    """
    Exciton basis class

    Args:
        nstate (int): number of states
        names (list[str], optional): names of the states such as ["down", "up"].
            Defaults to None, in which case the names are set to ["S0", "S1", ...].

    """
    nstate: int
    names: list[str]

    def __init__(self, nstate: int, names: list[str] | None = None) -> None:
        self.nstate = nstate
        if names is None:
            self.names = [f"S{i}" for i in range(nstate)]
        else:
            self.names = names
        assert len(self.names) == nstate, f"len(names)={len(names)} != nstate={nstate}"
        assert all(isinstance(name, str) for name in self.names), "names must be list of str"

    def get_creation_matrix(self) -> NDArray[np.float64]:
        """
        Returns the creation matrix for the exciton basis

        Example:
        >>> exciton = Exciton(2)
        >>> exciton.get_creation_matrix()
        array([[0., 0.],
               [1., 0.]])
        >>> state = np.array([[1.0],
                              [0.0]])
        >>> np.allclose(exciton.get_creation_matrix() @ state, np.array([[0.0], [1.0]]))
        True
        >>> state = np.array([[0.0],
                              [1.0]])
        >>> np.allclose(exciton.get_creation_matrix() @ state, np.array([[0.0], [0.0]]))
        True
        """
        return self.get_annihilation_matrix().T

    def get_annihilation_matrix(self) -> NDArray[np.float64]:
        """
        Returns the annihilation matrix for the exciton basis

        Example:
        >>> exciton = Exciton(2)
        >>> exciton.get_annihilation_matrix()
        array([[0., 1.],
               [0., 0.]])
        >>> state = np.array([[1.0],
                              [0.0]])
        >>> np.allclose(exciton.get_annihilation_matrix() @ state, np.array([[0.0], [0.0]]))
        True
        >>> state = np.array([[0.0],
                              [1.0]])
        >>> np.allclose(exciton.get_annihilation_matrix() @ state, np.array([[1.0], [0.0]]))
        """
        mat = np.zeros((self.nstate, self.nstate), dtype=np.float64)
        for i in range(self.nstate - 1):
            mat[i, i + 1] = 1.0
        return mat

    @property
    def nprim(self) -> int:
        return self.nstate

    def __len__(self) -> int:
        return self.nstate


if __name__ == "__main__":
    exciton = Exciton(2)
    down = np.array([1.0, 0.0])
    up = np.array([0.0, 1.0])
    zero = np.array([0.0, 0.0])
    cmat = exciton.get_creation_matrix()
    amat = exciton.get_annihilation_matrix()
    for state, mat, ans in zip([down, down, up, up], [cmat, amat, cmat, amat], [up, zero, zero, down]):
        np.testing.assert_allclose(state @ mat, ans)
