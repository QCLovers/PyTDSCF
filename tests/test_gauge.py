import pytest
import pytdscf
import pytdscf._const_cls
import pytdscf._mps_cls
from logging import getLogger, DEBUG

import pytdscf._site_cls

logger = getLogger(__name__)
logger.setLevel(DEBUG)


def test_gauge():
    pytdscf._const_cls.const.set_runtype(use_jax=False)

    lattice_info = pytdscf._mps_cls.LatticeInfo([[4], [4], [4], [4]])
    logger.debug(f"lattice_info: {lattice_info}")
    weight_vib = [
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ]
    superblock = lattice_info.alloc_superblock_random(5, 1.0, weight_vib)
    logger.debug(f"superblock: {superblock}")
    mps = [core.data for core in superblock]
    logger.debug(f"mps: {mps}")
    for isite in range(1, 4):
        pytdscf._site_cls.validate_Btensor(mps[isite])


    # raise NotImplementedError



if __name__ == "__main__":
    pytest.main([__file__])
