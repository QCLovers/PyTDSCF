"""MPO-based real-space parallelMPS class"""

from __future__ import annotations

import numpy as np
from loguru import logger

from pytdscf._const_cls import const
from pytdscf._mps_cls import SiteCoef, canonicalize
from pytdscf._mps_mpo import MPSCoefMPO
from pytdscf._spf_cls import SPFInts
from pytdscf.hamiltonian_cls import TensorHamiltonian
from pytdscf.model_cls import Model

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
except Exception:
    MPI = None  # type: ignore
    comm = None  # type: ignore

logger = logger.bind(name="rank")


def _mpi_finalize_on_error(func):
    """MPIプロセスを確実に終了させるデコレータ"""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if MPI is not None:
                comm.Abort(1)  # 全プロセスを強制終了
            raise e

    return wrapper


class MPSCoefParallel(MPSCoefMPO):
    """Parallel MPS Coefficient Class

    !!! THIS CLASS DOES NOT SUPPOORT DIRECT PRODUCT MPS |mps0> otimes |mps1> otimes ... otimes |mpsN>
    !!! IF YOU WANT TO USE DIRECT PRODUCT MPS, USE SECOND QUANTIZATION OPERATORS INSTEAD
    """

    superblock_states: list[list[SiteCoef]]  # each latest rank data
    superblock_states_all_A: list[
        SiteCoef
    ]  # each rank data with gauge AAA except for the right terminal which has gauge AAPsi
    superblock_states_all_B: list[
        SiteCoef
    ]  # each rank data with gauge BBB except for the left terminal which has gauge PsiBB
    superblock_states_all_B_world: list[SiteCoef]  # whole data stored in rank0
    superblock_states_all_A_world: list[SiteCoef]  # whole data stored in rank0

    @classmethod
    def alloc_random(cls, model: Model) -> MPSCoefParallel:
        if MPI is None:
            raise ImportError("mpi4py is not installed")
        mps_coef = cls()
        supercls = super().alloc_random(model)
        for attr in [
            "dofs_cas",
            "lattice_info_states",
            "superblock_states",
            "nstate",
            "nsite",
            "ndof_per_sites",
        ]:
            setattr(mps_coef, attr, getattr(supercls, attr))
        if len(mps_coef.superblock_states) != 1:
            raise NotImplementedError
        if const.mpi_rank == 0:
            mps_coef.superblock_states_all_B_world = mps_coef.superblock_states[
                0
            ]

        mps_coef = distribute_superblock_states(mps_coef)
        return mps_coef

    def propagate(
        self, stepsize: float, ints_spf: SPFInts | None, matH: TensorHamiltonian
    ):
        super().propagate(stepsize, ints_spf, matH)

    def autocorr(self, ints_spf: SPFInts, psite: int = 0) -> complex | None:
        if const.use_jax:
            raise NotImplementedError
        else:
            rank = const.mpi_rank
            size = comm.size
            mid_rank = size // 2

            is_forward_group = rank < mid_rank
            ket_cores: list[np.ndarray] = [
                ket.data  # type: ignore
                for ket in self.superblock_states_all_B
            ]
            bra_cores = ket_cores
            block = None

            if is_forward_group:
                if rank == 0:
                    block = _ovlp_single_state_np_from_left(
                        bra_cores, ket_cores, block
                    )
                else:
                    block = comm.recv(source=rank - 1, tag=0)
                    block = _ovlp_single_state_np_from_left(
                        bra_cores, ket_cores, block
                    )
                if rank != mid_rank - 1:
                    comm.send(block, dest=rank + 1, tag=0)
                elif rank == mid_rank - 1 and rank != 0:
                    comm.send(block, dest=0, tag=0)
                elif rank == mid_rank - 1 and rank == 0:
                    pass
                else:
                    raise ValueError(f"{rank=}")
            else:
                if rank == size - 1:
                    block = _ovlp_single_state_np_from_right(
                        bra_cores, ket_cores, block
                    )
                else:
                    block = comm.recv(source=rank + 1, tag=1)
                    block = _ovlp_single_state_np_from_right(
                        bra_cores, ket_cores, block
                    )
                if rank != mid_rank:
                    comm.send(block, dest=rank - 1, tag=1)
                elif rank == mid_rank:
                    comm.send(block, dest=0, tag=1)
                else:
                    raise ValueError(f"{rank=}")

            if rank == 0:
                if mid_rank - 1 == 0:
                    block_left = block
                else:
                    block_left = comm.recv(source=mid_rank - 1, tag=0)
                block_right = comm.recv(source=mid_rank, tag=1)
                final_result = np.einsum("ab,ab->", block_left, block_right)
                # logger.debug(f"{final_result=}")
                return complex(final_result)
            else:
                return None

    def to_MPSCoefMPO(self) -> MPSCoefMPO | None:
        self.sync_world()
        if const.mpi_rank == 0:
            mps = MPSCoefMPO()
            for attr in [
                "dofs_cas",
                "lattice_info_states",
                "nstate",
                "nsite",
                "ndof_per_sites",
            ]:
                setattr(mps, attr, getattr(self, attr))
            mps_superblock_states = [self.superblock_states_all_B_world]
            mps.superblock_states = mps_superblock_states
            return mps
        else:
            return None

    def expectation(
        self, ints_spf: SPFInts, matOp: TensorHamiltonian, psite: int = 0
    ) -> complex | None:
        raise NotImplementedError

    def sync_world(self):
        # collect all_A and all_B from all ranks and store in rank0
        recv_all_A: list[list[SiteCoef]] = comm.gather(
            self.superblock_states_all_A, root=0
        )  # type: ignore
        recv_all_B: list[list[SiteCoef]] = comm.gather(
            self.superblock_states_all_B, root=0
        )  # type: ignore
        if const.mpi_rank == 0:
            self.superblock_states_all_B_world: list[SiteCoef] = []
            self.superblock_states_all_A_world: list[SiteCoef] = []
            for all_A, all_B in zip(recv_all_A, recv_all_B, strict=True):
                self.superblock_states_all_A_world.extend(all_A)
                self.superblock_states_all_B_world.extend(all_B)


def _ovlp_single_state_np_from_left(
    bra_cores: list[np.ndarray],
    ket_cores: list[np.ndarray],
    _block: np.ndarray | None = None,
) -> np.ndarray:
    a = bra_cores[0]
    b = ket_cores[0]
    if isinstance(_block, np.ndarray):
        block = _block
    else:
        block = np.einsum("ab,ac->bc", a[0, :, :], b[0, :, :])
    for i in range(1, len(bra_cores)):
        a = bra_cores[i]
        b = ket_cores[i]
        block = np.einsum("bed,cef,bc->df", a, b, block)
    return block


def _ovlp_single_state_np_from_right(
    bra_cores: list[np.ndarray],
    ket_cores: list[np.ndarray],
    _block: np.ndarray | None = None,
) -> np.ndarray:
    a = bra_cores[-1]
    b = ket_cores[-1]
    if isinstance(_block, np.ndarray):
        block = _block
    else:
        block = np.einsum("ab,cb->ac", a[:, :, 0], b[:, :, 0])
    assert isinstance(block, np.ndarray)
    for i in range(len(bra_cores) - 2, -1, -1):
        a = bra_cores[i]
        b = ket_cores[i]
        block = np.einsum("dea,fec,ac->df", a, b, block)
    return block


def distribute_superblock_states(mps_coef: MPSCoefParallel) -> MPSCoefParallel:
    # rank0でデータを分割
    if const.mpi_rank == 0:
        # split_indicesに基づいてsuperblock_statesを分割
        states = mps_coef.superblock_states_all_B_world
        canonicalize(states, orthogonal_center=0)
        mps_coef.superblock_states_all_B_world = states
        states_copy = [core.copy() for core in states]
        canonicalize(states_copy, orthogonal_center=len(states) - 1)
        mps_coef.superblock_states_all_A_world = states_copy
        logger.debug(f"{mps_coef.superblock_states_all_B_world=}")
        logger.debug(f"{mps_coef.superblock_states_all_A_world=}")
        split_indices = const.split_indices
        send_data_all_B = []
        send_data_all_A = []
        # 各ランク用のデータを準備
        for i in range(comm.size):
            if i < len(split_indices) - 1:
                rank_states_all_B = mps_coef.superblock_states_all_B_world[
                    split_indices[i] : split_indices[i + 1]
                ]
                rank_states_all_A = mps_coef.superblock_states_all_A_world[
                    split_indices[i] : split_indices[i + 1]
                ]
            else:
                rank_states_all_B = mps_coef.superblock_states_all_B_world[
                    split_indices[i] :
                ]
                rank_states_all_A = mps_coef.superblock_states_all_A_world[
                    split_indices[i] :
                ]
            send_data_all_B.append(rank_states_all_B)
            send_data_all_A.append(rank_states_all_A)
    else:
        send_data_all_B = None
        send_data_all_A = None

    # 分割したデータを各ランクに配布
    recv_data_all_B = comm.scatter(send_data_all_B, root=0)
    recv_data_all_A = comm.scatter(send_data_all_A, root=0)
    mps_coef.superblock_states_all_B = recv_data_all_B
    mps_coef.superblock_states_all_A = recv_data_all_A
    if const.mpi_rank % 2 == 0:
        mps_coef.superblock_states = recv_data_all_B
    else:
        mps_coef.superblock_states = recv_data_all_A
    logger.debug(f"{mps_coef.superblock_states[0]=}")
    logger.debug(f"{mps_coef.superblock_states_all_B=}")
    logger.debug(f"{mps_coef.superblock_states_all_A=}")
    return mps_coef
