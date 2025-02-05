"""MPO-based real-space parallelMPS class"""

from __future__ import annotations

import numpy as np
from loguru import logger

from pytdscf._const_cls import const
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
    """Parallel MPS Coefficient Class"""

    @classmethod
    @_mpi_finalize_on_error
    def alloc_random(cls, model: Model) -> MPSCoefParallel:
        if MPI is None:
            raise ImportError("mpi4py is not installed")
        comm = MPI.COMM_WORLD
        mps_coef = cls()
        # 親クラスのインスタンス変数を一括コピー
        for attr in [
            "dofs_cas",
            "lattice_info_states",
            "superblock_states",
            "nstate",
            "nsite",
            "ndof_per_sites",
        ]:
            setattr(mps_coef, attr, getattr(super().alloc_random(model), attr))
        if len(mps_coef.superblock_states) != 1:
            raise NotImplementedError

        # rank0でデータを分割
        if const.mpi_rank == 0:
            # split_indicesに基づいてsuperblock_statesを分割
            mps_coef.superblock_states_world_left_terminal = (
                mps_coef.superblock_states
            )
            states = mps_coef.superblock_states[0]
            logger.debug(f"{states=}")
            split_indices = const.split_indices
            send_data = []
            # 各ランク用のデータを準備
            for i in range(comm.size):
                if i < len(split_indices) - 1:
                    rank_states = states[
                        split_indices[i] : split_indices[i + 1]
                    ]
                else:
                    rank_states = states[split_indices[i] :]
                send_data.append(rank_states)
        else:
            send_data = None

        # 分割したデータを各ランクに配布
        recv_data = comm.scatter(send_data, root=0)
        mps_coef.superblock_states = [recv_data]
        logger.debug(f"{mps_coef.superblock_states[0]=}")
        if const.mpi_rank == 0:
            logger.debug(f"{mps_coef.superblock_states_world_left_terminal=}")

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
                for ket in self.superblock_states[0]
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

    def expectation(
        self, ints_spf: SPFInts, matOp: TensorHamiltonian, psite: int = 0
    ) -> complex | None:
        raise NotImplementedError

    def sync_to_world_left(self):
        # 転送してから、canonicalizeする。
        raise NotImplementedError


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
