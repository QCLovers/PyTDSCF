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
import numpy as np
import pytest

logger = loguru.logger.bind(name="rank")


def get_mps_parallel(adaptive: bool):
    import pytdscf
    import pytdscf._const_cls
    import pytdscf._mps_cls
    import pytdscf._mps_parallel

    match pytdscf._const_cls.const.mpi_size:
        case 2:
            pytdscf._const_cls.const.set_runtype(
                use_jax=False,
                parallel_split_indices=[(0, 5), (6, 11)],
                adaptive=adaptive,
                adaptive_Dmax=30,
                adaptive_dD=30,
                adaptive_p_proj=1e-04,
                adaptive_p_svd=1e-7,
            )
        case 3:
            pytdscf._const_cls.const.set_runtype(
                use_jax=False,
                parallel_split_indices=[(0, 3), (4, 7), (8, 11)],
                adaptive=adaptive,
                adaptive_Dmax=30,
                adaptive_dD=30,
                adaptive_p_proj=1e-04,
                adaptive_p_svd=1e-7,
            )
        case 4:
            pytdscf._const_cls.const.set_runtype(
                use_jax=False,
                parallel_split_indices=[(0, 2), (3, 5), (6, 8), (9, 11)],
                adaptive=adaptive,
                adaptive_Dmax=30,
                adaptive_dD=30,
                adaptive_p_proj=1e-04,
                adaptive_p_svd=1e-7,
            )
        case _:
            raise ValueError(
                f"MPI size must be 2 or 3 or 4, but {pytdscf._const_cls.const.mpi_size} is given."
            )
    if pytdscf._const_cls.const.mpi_rank == 0:
        lattice_info = pytdscf._mps_cls.LatticeInfo(
            [[4], [4], [4], [4], [4], [4], [4], [4], [4], [4], [4], [4]]
        )
        weight_vib = [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
        superblock = lattice_info.alloc_superblock_random(
            m_aux_max=1 if not adaptive else 10,
            scale=1.0,
            core_weight=weight_vib,
        )
    else:
        superblock = None
    mps_parallel = pytdscf._mps_parallel.MPSCoefParallel()
    mps_parallel.superblock_all_B_world = superblock
    mps_parallel = pytdscf._mps_parallel.distribute_superblock_states(
        mps_parallel
    )
    return mps_parallel


def get_hamiltonian():
    from pytdscf import TensorHamiltonian, TensorOperator

    if rank == 0:
        # Define a Hamiltonian
        mpo = []
        for i in range(12):
            core = np.zeros((1, 4, 4, 1), dtype=np.complex128)
            if i == 0:
                core[0, :, :, 0] = np.eye(4) * 2.0
            else:
                core[0, :, :, 0] = np.eye(4)
            mpo.append(core)
        potential = [
            [
                {
                    (
                        (0, 0),
                        (1, 1),
                        (2, 2),
                        (3, 3),
                        (4, 4),
                        (5, 5),
                        (6, 6),
                        (7, 7),
                        (8, 8),
                        (9, 9),
                        (10, 10),
                        (11, 11),
                    ): TensorOperator(
                        mpo=mpo,
                        legs=(
                            0,
                            0,
                            1,
                            1,
                            2,
                            2,
                            3,
                            3,
                            4,
                            4,
                            5,
                            5,
                            6,
                            6,
                            7,
                            7,
                            8,
                            8,
                            9,
                            9,
                            10,
                            10,
                            11,
                            11,
                        ),
                    )
                }
            ]
        ]
    else:
        potential = None
    hamiltonian = TensorHamiltonian(
        ndof=12,
        potential=potential,
        backend="numpy",
    )
    return hamiltonian


@pytest.mark.skipif(MPI is None, reason="MPI is not installed")
@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() == 1, reason="Not running under MPI"
)
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
@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() not in [2, 3, 4],
    reason="Run under 2 or 3 or 4 MPI ranks",
)
@pytest.mark.parametrize("adaptive", [True, False])
def test_mpi_autocorr_norm(adaptive: bool):
    mps_parallel = get_mps_parallel(adaptive)
    autocorr = mps_parallel.autocorr(ints_spf=None)
    if rank == 0:
        assert pytest.approx(1.0, abs=1.0e-05) == autocorr, f"{autocorr=}"
    else:
        assert autocorr is None
    comm.Barrier()
    norm = mps_parallel.norm()
    if rank == 0:
        assert pytest.approx(1.0, abs=1.0e-05) == norm, f"{norm=}"
    else:
        assert norm is None

    mps_parallel.sync_world_AB()
    if rank == 0:
        block = np.ones((1, 1), dtype=complex)
        ci_bra = mps_parallel.superblock_all_B_world
        ci_ket = ci_bra
        for site_bra, site_ket in zip(ci_bra, ci_ket, strict=True):
            bra = site_bra.data
            ket = site_ket.data
            block = np.einsum(
                "abc,abk->ck", bra, np.einsum("ibk,ai->abk", ket, block)
            )
        assert pytest.approx(1.0, abs=1.0e-05) == block[0, 0], f"{block=}"
    MPI.COMM_WORLD.Barrier()


@pytest.mark.skipif(MPI is None, reason="MPI is not installed")
@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() not in [2, 3, 4],
    reason="Run under 2 or 3 or 4 MPI ranks",
)
@pytest.mark.parametrize("adaptive", [True, False])
def test_mpi_expectation(adaptive: bool):
    mps_parallel = get_mps_parallel(adaptive)
    hamiltonian = get_hamiltonian()
    hamiltonian.distribute_mpo_cores()
    expectation = mps_parallel.expectation(ints_spf=None, matOp=hamiltonian)
    if rank == 0:
        assert pytest.approx(2.0) == expectation, f"{expectation=}"
    else:
        assert expectation is None
    MPI.COMM_WORLD.Barrier()


@pytest.mark.skipif(MPI is None, reason="MPI is not installed")
@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() not in [2, 3, 4],
    reason="Run under 2  or 3 or 4 MPI ranks",
)
@pytest.mark.parametrize("adaptive", [True, False])
def test_mpi_reduced_density(adaptive: bool):
    mps_parallel = get_mps_parallel(adaptive)
    reduced_density = mps_parallel.get_reduced_densities(
        base_tag=0, rd_key=(5, 5)
    )
    if rank == 0:
        np.testing.assert_allclose(reduced_density, np.ones((4, 4)) * 0.25)
    reduced_density = mps_parallel.get_reduced_densities(
        base_tag=1, rd_key=(0,)
    )
    if rank == 0:
        np.testing.assert_allclose(
            reduced_density, np.array([1.0, 0.0, 0.0, 0.0])
        )
    reduced_density = mps_parallel.get_reduced_densities(
        base_tag=2, rd_key=(0, 1, 4)
    )
    if rank == 0:
        assert reduced_density.shape == (
            4,
            4,
            4,
        ), f"{reduced_density.shape=}"
        np.testing.assert_allclose(
            reduced_density[:1, :2, :4], np.ones((1, 2, 4)) * 1 / 8
        )


@pytest.mark.skipif(MPI is None, reason="MPI is not installed")
@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() not in [2, 3, 4],
    reason="Run under 2 or 3 or 4 MPI ranks",
)
@pytest.mark.parametrize("adaptive", [True, False])
def test_mpi_propagate(adaptive: bool):
    mps_parallel = get_mps_parallel(adaptive)
    hamiltonian = get_hamiltonian()
    hamiltonian.distribute_mpo_cores()
    # with pytest.raises(NotImplementedError):
    mps_parallel.propagate(stepsize=0.1, ints_spf=None, matH=hamiltonian)
    mps_parallel.propagate(stepsize=0.1, ints_spf=None, matH=hamiltonian)


if __name__ == "__main__":
    # test_mpi()
    test_mpi_autocorr_norm(True)
    test_mpi_expectation(True)
    test_mpi_reduced_density(True)
    test_mpi_propagate(True)
    MPI.COMM_WORLD.Barrier()
