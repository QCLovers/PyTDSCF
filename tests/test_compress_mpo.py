import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pytest

from pytdscf.dvr_operator_cls import tensor_dict_to_mpo


def _get_val_tensor(
    tensor_dict: Dict[Tuple[int, ...], np.ndarray], J: np.ndarray
) -> float:
    retval = 0.0
    for key, tensor in tensor_dict.items():
        if key == ():
            continue
        retval += tensor[tuple(J[np.ix_(key)])]
    return retval


def _get_val_mpo(mpo: List[np.ndarray], J: np.ndarray) -> float:
    retval = 0.0
    for i, W in enumerate(mpo):
        if i == 0:
            retval = W[0, J[i], :]
        else:
            retval = np.einsum("i,ij->j", retval, W[:, J[i], :])
    return retval[0].real


@pytest.mark.parametrize(
    "rate, J",
    [
        [1.0, np.array([2, 2, 2, 2, 2, 2])],
        [0.999999999999, np.array([1, 2, 3, 1, 2, 3])],
    ],
)
def test_compress_mpo(rate, J):
    tensor_dict = pickle.load(
        open(f"{os.path.dirname(os.path.abspath(__file__))}/h2co.tensor", "rb")
    )
    mpo = tensor_dict_to_mpo(tensor_dict, rate=rate)
    val_tensor = _get_val_tensor(tensor_dict, J)
    val_mpo = _get_val_mpo(mpo, J)
    print("J | rate | abs. error | bond-dimension")
    print(
        f"{J} | {rate} | {abs(val_tensor - val_mpo)} | {[W.shape[-1] for W in mpo[:-1]]}"
    )
    assert abs(val_tensor - val_mpo) < 1.0e-10


if __name__ == "__main__":
    # test_compress_mpo(rate=0.999999999, J=np.array([1, 2, 3, 1, 2, 3]))

    test_compress_mpo(rate=0.999999999, J=np.random.randint(0, 5, 6))
