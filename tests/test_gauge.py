import loguru
import numpy as np
import pytest

import pytdscf
import pytdscf._const_cls
import pytdscf._mps_cls
import pytdscf._site_cls

logger = loguru.logger


def test_gauge():
    pytdscf._const_cls.const.set_runtype(use_jax=False)

    lattice_info = pytdscf._mps_cls.LatticeInfo([[4], [4], [4], [4], [4]])
    logger.debug(f"lattice_info: {lattice_info}")
    weight_vib = [
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ]
    superblock = lattice_info.alloc_superblock_random(5, 1.0, weight_vib)

    logger.debug(f"superblock: {superblock}")
    contracted = pytdscf._mps_cls.contract_all_superblock(superblock)
    for isite in range(1, 5):
        pytdscf._site_cls.validate_Btensor(superblock[isite])

    pytdscf._mps_cls.canonicalizeA(superblock[:3])
    logger.debug(f"superblock: {superblock}")
    for isite in range(2):
        pytdscf._site_cls.validate_Atensor(superblock[isite])
    for isite in range(3, 5):
        pytdscf._site_cls.validate_Btensor(superblock[isite])
    np.testing.assert_allclose(
        pytdscf._mps_cls.contract_all_superblock(superblock), contracted
    )

    pytdscf._mps_cls.canonicalize(superblock, 3)
    logger.debug(f"superblock: {superblock}")
    for isite in range(3):
        pytdscf._site_cls.validate_Atensor(superblock[isite])
    for isite in range(4, 5):
        pytdscf._site_cls.validate_Btensor(superblock[isite])
    np.testing.assert_allclose(
        pytdscf._mps_cls.contract_all_superblock(superblock), contracted
    )

    shapes = [
        (1, 4, 3),
        (3, 4, 5),
        (5, 4, 4),
        (4, 4, 3),
        (3, 4, 1),
    ]
    random_cores = [np.random.rand(*shape) for shape in shapes]
    superblock = [
        pytdscf._site_cls.SiteCoef(core, "C", isite)
        for isite, core in enumerate(random_cores)
    ]
    logger.debug(f"superblock: {superblock}")
    contracted = pytdscf._mps_cls.contract_all_superblock(superblock)
    center = 4
    pytdscf._mps_cls.canonicalize(superblock, center)
    logger.debug(f"superblock: {superblock}")
    for isite in range(center):
        pytdscf._site_cls.validate_Atensor(superblock[isite])
    for isite in range(center + 1, 5):
        pytdscf._site_cls.validate_Btensor(superblock[isite])
    np.testing.assert_allclose(
        pytdscf._mps_cls.contract_all_superblock(superblock), contracted
    )


if __name__ == "__main__":
    pytest.main([__file__])
