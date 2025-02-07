from pytdscf._const_cls import const
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError | ModuleNotFoundError:
    MPI = None
    comm = None
    rank = 0
    size = 1
import sys
import loguru
import pytest
import numpy as np
logger = loguru.logger.bind(name='rank')

@pytest.mark.skipif(MPI is None, reason="MPI is not installed")
@pytest.mark.skipif(MPI.COMM_WORLD.Get_size() == 1, reason="Not running under MPI")
def test_mpi():

    assert const.mpi_rank == MPI.COMM_WORLD.Get_rank()
    assert const.mpi_size == MPI.COMM_WORLD.Get_size()
    logger.info(f"MPI rank: {const.mpi_rank}, MPI size: {const.mpi_size}")
    logger.info(f"MPI command: {' '.join(sys.argv)}")
    logger.info(f"MPI version: {MPI.Get_version()}")
    logger.info(f"MPI implementation: {MPI.Get_library_version()}")
    # Universe size
    logger.info(f"Universe size: {MPI.UNIVERSE_SIZE}")
    logger.info(f"{[arg for arg in sys.argv]}")

@pytest.mark.skipif(MPI is None, reason="MPI is not installed")
@pytest.mark.skipif(MPI.COMM_WORLD.Get_size() not in [2, 3], reason="Run under 2 or 3 MPI ranks")
def test_mpi_autocorr():
    import pytdscf
    import pytdscf._const_cls
    import pytdscf._mps_cls
    import pytdscf._mps_parallel

    if pytdscf._const_cls.const.mpi_size == 2:
        pytdscf._const_cls.const.set_runtype(use_jax=False, parallel_split_indices=[(0, 2), (3, 5)])
    elif pytdscf._const_cls.const.mpi_size == 3:
        pytdscf._const_cls.const.set_runtype(use_jax=False, parallel_split_indices=[(0, 1), (2, 3), (4, 5)])
    if pytdscf._const_cls.const.mpi_rank == 0:
        lattice_info = pytdscf._mps_cls.LatticeInfo([[4], [4], [4], [4], [4], [4]])
        logger.debug(f"lattice_info: {lattice_info}")
        weight_vib = [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
        superblock = lattice_info.alloc_superblock_random(6, 1.0, weight_vib)
        logger.debug(f"superblock: {superblock}")
    else:
        superblock = None
    mps_parallel = pytdscf._mps_parallel.MPSCoefParallel()
    mps_parallel.superblock_states_all_B_world = [superblock]
    mps_parallel = pytdscf._mps_parallel.distribute_superblock_states(mps_parallel)

    autocorr = mps_parallel.autocorr(ints_spf=None)
    if rank == 0:
        assert pytest.approx(1.0) == autocorr, f"{autocorr=}"
    else:
        assert autocorr is None

    mps_parallel.sync_world()
    if rank == 0:
        block = np.ones((1, 1), dtype=complex)
        ci_bra = mps_parallel.superblock_states_all_B_world[0]
        ci_ket = ci_bra
        for site_bra, site_ket in zip(ci_bra, ci_ket, strict=True):
            bra = site_bra.data
            ket = site_ket.data
            block = np.einsum(
                "abc,abk->ck", bra, np.einsum("ibk,ai->abk", ket, block)
            )
        assert pytest.approx(1.0) == block[0, 0], f"{block=}"

def test_mpi_expectation():
    pass

if __name__ == "__main__":
    test_mpi_autocorr()
