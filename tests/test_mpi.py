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
    MPI.COMM_WORLD.Barrier()

@pytest.mark.skipif(MPI is None, reason="MPI is not installed")
@pytest.mark.skipif(MPI.COMM_WORLD.Get_size() not in [2, 3], reason="Run under 2 or 3 MPI ranks")
def test_mpi_autocorr_norm():
    import pytdscf
    import pytdscf._const_cls
    import pytdscf._mps_cls
    import pytdscf._mps_parallel

    if pytdscf._const_cls.const.mpi_size == 2:
        pytdscf._const_cls.const.set_runtype(use_jax=False, parallel_split_indices=[(0, 2), (3, 5)])
    elif pytdscf._const_cls.const.mpi_size == 3:
        pytdscf._const_cls.const.set_runtype(use_jax=False, parallel_split_indices=[(0, 1), (2, 3), (4, 5)])
    else:
        raise ValueError(f"MPI size must be 2 or 3, but {pytdscf._const_cls.const.mpi_size} is given.")
    if pytdscf._const_cls.const.mpi_rank == 0:
        lattice_info = pytdscf._mps_cls.LatticeInfo([[4], [4], [4], [4], [4], [4]])
        weight_vib = [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
        superblock = lattice_info.alloc_superblock_random(6, 1.0, weight_vib)
    else:
        superblock = None
    mps_parallel = pytdscf._mps_parallel.MPSCoefParallel()
    mps_parallel.superblock_states_all_B_world = superblock
    mps_parallel = pytdscf._mps_parallel.distribute_superblock_states(mps_parallel)

    autocorr = mps_parallel.autocorr(ints_spf=None)
    if rank == 0:
        assert pytest.approx(1.0) == autocorr, f"{autocorr=}"
    else:
        assert autocorr is None
    norm = mps_parallel.norm()
    if rank == 0:
        assert pytest.approx(1.0) == norm, f"{norm=}"
    else:
        assert norm is None

    mps_parallel.sync_world()
    if rank == 0:
        block = np.ones((1, 1), dtype=complex)
        ci_bra = mps_parallel.superblock_states_all_B_world
        ci_ket = ci_bra
        for site_bra, site_ket in zip(ci_bra, ci_ket, strict=True):
            bra = site_bra.data
            ket = site_ket.data
            block = np.einsum(
                "abc,abk->ck", bra, np.einsum("ibk,ai->abk", ket, block)
            )
        assert pytest.approx(1.0) == block[0, 0], f"{block=}"
    MPI.COMM_WORLD.Barrier()

@pytest.mark.skipif(MPI is None, reason="MPI is not installed")
@pytest.mark.skipif(MPI.COMM_WORLD.Get_size() not in [2, 3], reason="Run under 2 or 3 MPI ranks")
def test_mpi_expectation():
    import pytdscf
    import pytdscf._const_cls
    import pytdscf._mps_cls
    import pytdscf._mps_parallel
    from pytdscf import TensorHamiltonian, TensorOperator

    if pytdscf._const_cls.const.mpi_size == 2:
        pytdscf._const_cls.const.set_runtype(use_jax=False, parallel_split_indices=[(0, 2), (3, 5)])
    elif pytdscf._const_cls.const.mpi_size == 3:
        pytdscf._const_cls.const.set_runtype(use_jax=False, parallel_split_indices=[(0, 1), (2, 3), (4, 5)])
    else:
        raise ValueError(f"MPI size must be 2 or 3, but {pytdscf._const_cls.const.mpi_size} is given.")
    if pytdscf._const_cls.const.mpi_rank == 0:
        lattice_info = pytdscf._mps_cls.LatticeInfo([[4], [4], [4], [4], [4], [4]])
        weight_vib = [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
        superblock = lattice_info.alloc_superblock_random(6, 1.0, weight_vib)
    else:
        superblock = None
    mps_parallel = pytdscf._mps_parallel.MPSCoefParallel()
    mps_parallel.superblock_states_all_B_world = superblock
    mps_parallel = pytdscf._mps_parallel.distribute_superblock_states(mps_parallel)

    if rank == 0:
        # Define a Hamiltonian
        mpo = []
        for i in range(6):
            core = np.zeros((1, 4, 4, 1), dtype=np.complex128)
            if i == 0:
                core[0, :, :, 0] = np.eye(4) * 2.0
            else:
                core[0, :, :, 0] = np.eye(4)
            mpo.append(core)
        potential=[[{((0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)):
        TensorOperator(mpo=mpo, legs=(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5))}]]
    else:
        potential = None
    hamiltonian = TensorHamiltonian(
        ndof=6,
        potential=potential,
        backend="numpy",
    )
    hamiltonian.distribute_mpo_cores()
    expectation = mps_parallel.expectation(ints_spf=None, matOp=hamiltonian)
    if rank == 0:
        assert pytest.approx(2.0) == expectation, f"{expectation=}"
    else:
        assert expectation is None
    MPI.COMM_WORLD.Barrier()

@pytest.mark.skipif(MPI is None, reason="MPI is not installed")
@pytest.mark.skipif(MPI.COMM_WORLD.Get_size() not in [2, 3], reason="Run under 2 or 3 MPI ranks")
def test_mpi_reduced_density():
    import pytdscf
    import pytdscf._const_cls
    import pytdscf._mps_cls
    import pytdscf._mps_parallel

    if pytdscf._const_cls.const.mpi_size == 2:
        pytdscf._const_cls.const.set_runtype(use_jax=False, parallel_split_indices=[(0, 2), (3, 5)])
    elif pytdscf._const_cls.const.mpi_size == 3:
        pytdscf._const_cls.const.set_runtype(use_jax=False, parallel_split_indices=[(0, 1), (2, 3), (4, 5)])
    else:
        raise ValueError(f"MPI size must be 2 or 3, but {pytdscf._const_cls.const.mpi_size} is given.")
    if pytdscf._const_cls.const.mpi_rank == 0:
        lattice_info = pytdscf._mps_cls.LatticeInfo([[4], [4], [4], [4], [4], [4]])
        weight_vib = [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
        superblock = lattice_info.alloc_superblock_random(6, 1.0, weight_vib)
    else:
        superblock = None
    mps_parallel = pytdscf._mps_parallel.MPSCoefParallel()
    mps_parallel.superblock_states_all_B_world = superblock
    mps_parallel = pytdscf._mps_parallel.distribute_superblock_states(mps_parallel)
    reduced_density = mps_parallel.get_reduced_densities(base_tag=0, rd_key=(5, 5))
    if rank == 0:
        np.testing.assert_allclose(reduced_density[0], np.ones((4, 4))*0.25)
    reduced_density = mps_parallel.get_reduced_densities(base_tag=1, rd_key=(0,))
    if rank == 0:
        np.testing.assert_allclose(reduced_density[0], np.array([1.0, 0.0, 0.0, 0.0]))
    reduced_density = mps_parallel.get_reduced_densities(base_tag=2, rd_key=(0, 1, 4))
    if rank == 0:
        assert reduced_density[0].shape == (4, 4, 4), f"{reduced_density[0].shape=}"
        np.testing.assert_allclose(reduced_density[0][:1, :2, :4], np.ones((1, 2, 4))*1/8)
if __name__ == "__main__":
    #test_mpi()
    #test_mpi_autocorr_norm()
    #test_mpi_expectation()
    test_mpi_reduced_density()
    MPI.COMM_WORLD.Barrier()
