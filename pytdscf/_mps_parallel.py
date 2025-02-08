"""MPO-based real-space parallelMPS class"""

from __future__ import annotations

from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger

from pytdscf import _integrator
from pytdscf._const_cls import const
from pytdscf._contraction import multiplyK_MPS_direct_MPO
from pytdscf._mps_cls import CC2ALambdaB, SiteCoef, canonicalize
from pytdscf._mps_mpo import MPSCoefMPO, myndarray
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
    joint_sigvec: np.ndarray | jax.Array  # Site with sigma gauge

    @classmethod
    def alloc_random(cls, model: Model) -> MPSCoefParallel:
        if MPI is None:
            raise ImportError("mpi4py is not installed")
        mps_coef = cls()
        supercls = super().alloc_random(model)
        for attr in [
            "dofs_cas",
            "lattice_info_states",
            "nstate",
            "ndof_per_sites",
        ]:
            setattr(mps_coef, attr, getattr(supercls, attr))
        superblock_states = supercls.superblock_states
        if len(superblock_states) != 1:
            raise NotImplementedError
        if const.mpi_rank == 0:
            mps_coef.superblock_states_all_B_world = superblock_states[0]

        mps_coef = distribute_superblock_states(mps_coef)
        return mps_coef

    def propagate(
        self, stepsize: float, ints_spf: SPFInts | None, matH: TensorHamiltonian
    ):
        raise NotImplementedError
        super().propagate(stepsize, ints_spf, matH)

    def ovlp(self, conj=True) -> complex | float | None:
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
            if conj:
                bra_cores: list[np.ndarray] = [
                    bra.data.conj()  # type: ignore
                    for bra in self.superblock_states_all_B
                ]
            else:
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
                if conj:
                    return final_result.real
                else:
                    return complex(final_result)
            else:
                return None

    def autocorr(self, ints_spf: SPFInts, psite: int = 0) -> complex | None:
        return self.ovlp(conj=False)

    def norm(self) -> float | None:
        return self.ovlp(conj=True)  # type: ignore

    def pop_states(self) -> list[float | None]:
        # Use reduced density and second quantization operator instead.
        # Only support single MPS for now.
        return [self.norm()]

    def get_reduced_densities(
        self, base_tag: int, rd_key: tuple[int, ...]
    ) -> list[np.ndarray] | None:
        """
        When rd_key is  (3, 3, 4)
        and MPS is A1A2A3A4...A6
        Contruct (A1A1†)(A2A2†)=1 and (B5B5†)(B6B6†)=1 in advance (no calculation needed)
        then, calculate (A3A3†)_ij(A4A4†)_kk
        """
        if const.use_jax:
            raise NotImplementedError
        base_tag = base_tag * 5
        left_site = min(rd_key)
        right_site = max(rd_key)
        counter = Counter(rd_key)
        needed_rank = []
        split_indices = const.split_indices
        opend_legs = []
        for i_rank in range(comm.size):
            if split_indices[i_rank] > right_site:
                break
            elif (
                i_rank + 1 < comm.size
                and split_indices[i_rank + 1] - 1 < left_site
            ):
                continue
            else:
                needed_rank.append(i_rank)
                if i_rank == const.mpi_rank:
                    for isite in range(
                        const.bgn_site_rank, const.end_site_rank + 1
                    ):
                        opend_legs.append(counter[isite])
        if const.mpi_rank != 0 and const.mpi_rank not in needed_rank:
            return None
        mid_rank = (len(needed_rank) - 1) // 2 + needed_rank[0]
        is_forward_group = (
            const.mpi_rank <= mid_rank and const.mpi_rank in needed_rank
        )
        is_backward_group = (
            const.mpi_rank > mid_rank and const.mpi_rank in needed_rank
        )
        if is_forward_group:
            superblock = self.superblock_states_all_A
        elif is_backward_group:
            superblock = self.superblock_states_all_B
        elif const.mpi_rank == 0:
            pass
        else:
            raise ValueError(f"{const.mpi_rank=} {needed_rank=}")

        block = None
        if is_forward_group:
            if const.mpi_rank == needed_rank[0]:
                block = None
            else:
                block = comm.recv(source=const.mpi_rank - 1, tag=base_tag + 0)

            for isite in range(self.nsite):
                if block is None:
                    if opend_legs[isite] == 0:
                        continue
                    else:
                        block = np.eye(
                            superblock[isite].data.shape[0], dtype=complex
                        )
                if opend_legs[isite] == 0:
                    subscript = "...ad,abc,dbf->...cf"
                elif opend_legs[isite] == 1:
                    subscript = "...ad,abc,dbf->...bcf"
                    """
                    |‾˙˙˙‾|‾a a‾|‾c
                    |     |     b
                    |     |     b
                    |_..._|_d d_|_f
                    """
                elif opend_legs[isite] == 2:
                    subscript = "...ad,abc,def->...becf"
                    """
                    |‾˙˙˙‾|‾a a‾|‾c
                    |     |     b
                    |     |     e
                    |_..._|_d d_|_f
                    """
                else:
                    raise ValueError(f"{isite=} {opend_legs[isite]=}")
                core = superblock[isite].data
                block = np.einsum(subscript, block, core.conj(), core)
            if const.mpi_rank != mid_rank:
                comm.send(block, dest=const.mpi_rank + 1, tag=base_tag + 0)
        elif is_backward_group:
            if const.mpi_rank == needed_rank[-1]:
                block = None
            else:
                block = comm.recv(source=const.mpi_rank + 1, tag=base_tag + 1)
            for isite in range(self.nsite - 1, -1, -1):
                if block is None:
                    if opend_legs[isite] == 0:
                        continue
                    else:
                        block = np.eye(
                            superblock[isite].data.shape[2], dtype=complex
                        )
                if opend_legs[isite] == 0:
                    subscript = "lmi,bma,ia...->lb..."
                elif opend_legs[isite] == 1:
                    """
                    l‾|‾i i‾|‾˙˙˙‾|
                      m     ...   |
                    b_|_a a_|_..._|
                    """
                    subscript = "lmi,bma,ia...->lbm..."
                elif opend_legs[isite] == 2:
                    """
                    l‾|‾i i‾|‾˙˙˙‾|
                      m     ...   |
                      n     ...   |
                    b_|_a a_|_..._|
                    """
                    subscript = "lmi,bma,ia...->lbmn..."
                else:
                    raise ValueError(f"{isite=} {opend_legs[isite]=}")
                core = superblock[isite].data
                block = np.einsum(subscript, core.conj(), core, block)
            comm.send(block, dest=const.mpi_rank - 1, tag=base_tag + 1)
        if const.mpi_rank == mid_rank:
            left_block = block
            assert isinstance(left_block, np.ndarray)
            if const.mpi_rank == const.mpi_size - 1:
                # No sigvecs
                sigvec = np.eye(superblock[-1].data.shape[2], dtype=complex)
            else:
                sigvec = self.joint_sigvec  # type: ignore
            assert isinstance(sigvec, np.ndarray)
            if len(needed_rank) == 1:
                right_block = np.eye(sigvec.shape[0], dtype=complex)
            else:
                right_block = comm.recv(
                    source=const.mpi_rank + 1, tag=base_tag + 1
                )
                assert isinstance(right_block, np.ndarray)
            """
            |‾˙˙˙‾|‾a a‾l l‾|˙˙˙‾|
            |     |         |    |
            |     |         |    |
            |_..._|_d d_b b_|..._|
            """
            left_block = np.einsum(
                "...ad,al,db->...lb", left_block, sigvec.conj(), sigvec
            )
            rd = np.tensordot(
                left_block, right_block, axes=([-2, -1], [0, 1])
            )  # ...lb,lb...->...
            assert (
                rd.shape == left_block.shape[:-2] + right_block.shape[2:]
            ), f"{rd.shape=} {left_block.shape=} {right_block.shape=}"
            if mid_rank == 0:
                return [rd]
            else:
                comm.send(rd, dest=0, tag=base_tag + 2)
                return None
        if const.mpi_rank == 0:
            rd = comm.recv(source=mid_rank, tag=base_tag + 2)
            assert isinstance(rd, np.ndarray)
            return [rd]
        else:
            return None

    def expectation(
        self, ints_spf: SPFInts, matOp: TensorHamiltonian, psite: int = 0
    ) -> complex | None:
        rank = const.mpi_rank
        size = comm.size
        mid_rank = size // 2
        is_forward_group = rank < mid_rank
        if is_forward_group:
            superblock_states = [self.superblock_states_all_A]
        else:
            superblock_states = [self.superblock_states_all_B]

        if is_forward_group:
            if rank == 0:
                op_A_block = None
            else:
                # 配列データと属性を別々に受信
                op_A_block = comm.recv(source=rank - 1, tag=0)
                if (
                    op_A_block is not None
                    and (0, 0) in op_A_block
                    and "ovlp" in op_A_block[(0, 0)]
                ):
                    is_identity = comm.recv(source=rank - 1, tag=10)
                    op_A_block[(0, 0)]["ovlp"] = myndarray(
                        op_A_block[(0, 0)]["ovlp"], is_identity=is_identity
                    )

            op_A_block = self.construct_op_sites(
                superblock_states=superblock_states,
                ints_site=None,
                begin_site=0,
                end_site=self.nsite,
                matH_cas=matOp,
                op_initial_block=op_A_block,
            ).pop()

            if rank != mid_rank - 1:
                # 配列データと属性を別々に送信
                comm.send(op_A_block, dest=rank + 1, tag=0)
                if (0, 0) in op_A_block and "ovlp" in op_A_block[(0, 0)]:
                    comm.send(
                        op_A_block[(0, 0)]["ovlp"].is_identity,
                        dest=rank + 1,
                        tag=10,
                    )
            elif rank == mid_rank - 1:
                pass
            else:
                raise ValueError(f"{rank=}")
        else:
            if rank == size - 1:
                op_B_block = None
            else:
                # 配列データと属性を別々に受信
                op_B_block = comm.recv(source=rank + 1, tag=1)
                if (
                    op_B_block is not None
                    and (0, 0) in op_B_block
                    and "ovlp" in op_B_block[(0, 0)]
                ):
                    is_identity = comm.recv(source=rank + 1, tag=11)
                    op_B_block[(0, 0)]["ovlp"] = myndarray(
                        op_B_block[(0, 0)]["ovlp"], is_identity=is_identity
                    )

            op_B_block = self.construct_op_sites(
                superblock_states=superblock_states,
                ints_site=None,
                begin_site=self.nsite - 1,
                end_site=-1,
                matH_cas=matOp,
                op_initial_block=op_B_block,
            ).pop()

            if rank >= mid_rank:
                # 配列データと属性を別々に送信
                comm.send(op_B_block, dest=rank - 1, tag=1)
                if (0, 0) in op_B_block and "ovlp" in op_B_block[(0, 0)]:
                    comm.send(
                        op_B_block[(0, 0)]["ovlp"].is_identity,
                        dest=rank - 1,
                        tag=11,
                    )
            else:
                raise ValueError(f"{rank=}")

        if rank == mid_rank - 1:
            # calculate expectation value
            left_block = op_A_block
            right_block = comm.recv(source=rank + 1, tag=1)
            if (
                right_block is not None
                and (0, 0) in right_block
                and "ovlp" in right_block[(0, 0)]
            ):
                is_identity = comm.recv(source=rank + 1, tag=11)
                right_block[(0, 0)]["ovlp"] = myndarray(
                    right_block[(0, 0)]["ovlp"], is_identity=is_identity
                )
            sigvec: list[np.ndarray] | list[jax.Array] = [self.joint_sigvec]  # type: ignore
            op_lr = self.operators_for_superK(
                psite=self.nsite - 1,
                op_sys=left_block,  # type: ignore
                op_env=right_block,
                hamiltonian=matOp,
                A_is_sys=True,
            )
            multiplyK = multiplyK_MPS_direct_MPO(
                op_lr,
                matOp,
                sigvec,
            )
            expectation_value = _integrator.expectation_Op(
                sigvec,  # type: ignore
                multiplyK,
                sigvec,  # type: ignore
            )
            if rank == 0:
                return complex(expectation_value)
            else:
                comm.send(complex(expectation_value), dest=0, tag=2)
        if rank == 0:
            return comm.recv(source=mid_rank, tag=2)
        return None

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

    def to_MPSCoefMPO(self) -> MPSCoefMPO | None:
        self.sync_world()
        if const.mpi_rank == 0:
            mps = MPSCoefMPO()
            for attr in [
                "dofs_cas",
                "lattice_info_states",
                "nstate",
                "ndof_per_sites",
            ]:
                setattr(mps, attr, getattr(self, attr))
            mps_superblock_states = [self.superblock_states_all_B_world]
            mps.superblock_states = mps_superblock_states
            mps.nsite = len(mps_superblock_states[0])
            return mps
        else:
            return None


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
        states = mps_coef.superblock_states_all_B_world
        canonicalize(states, orthogonal_center=0)
        mps_coef.superblock_states_all_B_world = states

        # joint_sigvecsの準備
        joint_sigvecs: list[np.ndarray] | list[jax.Array] = []
        split_indices = const.split_indices
        states_copy = [core.copy() for core in states]
        for i in range(comm.size - 1):
            canonicalize(
                states_copy,
                orthogonal_center=split_indices[i + 1] - 1,
                incremental=True,
            )
            Psi = states_copy[split_indices[i + 1] - 1]
            B = states_copy[split_indices[i + 1]]
            Lambda = CC2ALambdaB(Psi, B)
            B.gauge = "Psi"
            if isinstance(B.data, np.ndarray):
                B.data = np.einsum("a,abc->abc", Lambda, B.data)
            else:
                B.data = jnp.einsum("a,abc->abc", Lambda, B.data)
            if isinstance(Lambda, np.ndarray):
                joint_sigvecs.append(np.diag(Lambda))  # type: ignore
            else:
                joint_sigvecs.append(jnp.diag(Lambda))  # type: ignore
        canonicalize(
            states_copy,
            orthogonal_center=len(states_copy) - 1,
            incremental=True,
        )
        mps_coef.superblock_states_all_A_world = states_copy

        # scatterのためにjoint_sigvecsを準備 (size個の要素が必要)
        send_joint_sigvecs = [None] * comm.size
        for i in range(comm.size - 1):  # 最後のrank以外にLambdaを送信
            send_joint_sigvecs[i] = joint_sigvecs[i]

        # all_A, all_Bの分配データを準備
        send_data_all_B = []
        send_data_all_A = []
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
        send_joint_sigvecs = None
        send_data_all_B = None
        send_data_all_A = None

    # joint_sigvecsを送信（最後のrankにはNoneが送られる）
    recv_joint_sigvec = comm.scatter(send_joint_sigvecs, root=0)
    if const.mpi_rank != comm.size - 1:
        mps_coef.joint_sigvec = recv_joint_sigvec

    # すべてのrankにall_A, all_Bを送信
    recv_data_all_B = comm.scatter(send_data_all_B, root=0)
    recv_data_all_A = comm.scatter(send_data_all_A, root=0)
    mps_coef.superblock_states_all_B = recv_data_all_B
    mps_coef.superblock_states_all_A = recv_data_all_A
    if const.mpi_rank % 2 == 0:
        mps_coef.superblock_states = recv_data_all_B
    else:
        mps_coef.superblock_states = recv_data_all_A
    mps_coef.nsite = const.end_site_rank - const.bgn_site_rank + 1
    return mps_coef
