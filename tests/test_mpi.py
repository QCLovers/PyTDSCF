from pytdscf._const_cls import const
try:
    from mpi4py import MPI
except ImportError | ModuleNotFoundError:
    MPI = None

import sys
import loguru
import pytest

logger = loguru.logger

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
