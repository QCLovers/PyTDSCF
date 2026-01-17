"""MPO-based real-space parallelMPS class"""

from __future__ import annotations

import functools
import time
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger

from pytdscf import _integrator
from pytdscf._const_cls import const
from pytdscf._contraction import _block_type, _op_keys, multiplyK_MPS_direct_MPO
from pytdscf._mps_cls import (
    CC2ALambdaB,
    SiteCoef,
    canonicalize,
    get_superblock_full,
)
from pytdscf._mps_mpo import MPSCoefMPO, myndarray
from pytdscf._site_cls import (
    multiply_sigvec_pinv,
    truncate_sigvec,
)
from pytdscf._spf_cls import SPFInts
from pytdscf.hamiltonian_cls import TensorHamiltonian
from pytdscf.model_cls import Model

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
except Exception:
    MPI = None
    comm = None

logger = logger.bind(name="rank")


def mpi_abort_on_exception(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            import traceback

            rank = MPI.COMM_WORLD.Get_rank()
            logger.error(f"[Rank {rank}] Exception in '{func.__name__}': {e}")
            traceback.print_exc()
            MPI.COMM_WORLD.Abort(1)

    return wrapper


class MPSCoefParallel(MPSCoefMPO):
    """Parallel MPS Coefficient Class

    !!! THIS CLASS DOES NOT SUPPOORT DIRECT PRODUCT MPS |mps0> otimes |mps1> otimes ... otimes |mpsN>
    !!! IF YOU WANT TO USE DIRECT PRODUCT MPS, USE SECOND QUANTIZATION OPERATORS INSTEAD
    """

    superblock_states: list[list[SiteCoef]]  # each latest rank data
    superblock_all_A: list[SiteCoef]
    # each rank data with gauge AAA except for the right terminal which has gauge AAPsi
    superblock_all_B: list[SiteCoef]
    # each rank data with gauge BBB except for the left terminal which has gauge PsiBB
    superblock_all_B_world: list[SiteCoef]  # whole data stored in rank0
    superblock_all_A_world: list[SiteCoef]  # whole data stored in rank0
    joint_sigvec: np.ndarray | jax.Array  # Site with sigma gauge
    joint_sigvec_not_pinv: np.ndarray | jax.Array  # Site with sigma gauge
    psi_L: SiteCoef | None  # orthogonal center recieved from left rank
    psi_R: SiteCoef | None  # orthogonal center recieved from right rank

    @classmethod
    def alloc_random(cls, model: Model) -> MPSCoefParallel:
        if MPI is None:
            raise ImportError("mpi4py is not installed")
        if const.use_jax:
            raise NotImplementedError("jax is not supported for parallel MPS")
        if const.space == "liouville":
            raise NotImplementedError(
                "liouville space is not supported for parallel MPS"
            )
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
            mps_coef.superblock_all_B_world = superblock_states[0]

        mps_coef = distribute_superblock_states(mps_coef)
        return mps_coef

    @mpi_abort_on_exception
    def propagate(
        self,
        stepsize: float,
        ints_spf: None,
        matH: TensorHamiltonian,
        load_balance: bool = False,
    ):
        # Number of quotes ' indicates propated steps
        #    | rank0     | rank1    | rank2     | rank3   | diff
        # (1)| ψ B B x+  | A A A x  | B B B x+  | A A ψ   | x -> x' (even_rank=True & rank!=size-1)
        # (2)| ψ B B x+  | A A ψ x+ | ψ B B x+  | A A ψ   | Ax'B -> Aψx+ψB
        # ...
        # (3)| A'A'ψ x+  | ψ B'B'x  | A'A'ψ x+  | ψ B'B'  | sweep
        # (4)| A'A'A'x'' | B'B'B'x  | A'A'A'x'' | B'B'B'  | ψ x+ψ -> A'x''B'
        # (5)| A'A'ψ'x+''| ψ'B'B'x  | A'A'ψ'x+''| ψ'B'B'  | A'x''B' -> ψ'x+''ψ'

        ints_site = None
        op_sys_sites_is_none: bool = self.op_sys_sites is None
        reset_op_block: bool = comm.allreduce(op_sys_sites_is_none, op=MPI.LOR)

        if reset_op_block:
            right_op_blocks = self.reset_right_op_blocks(matH)
            left_op_blocks = self.reset_left_op_blocks(matH)
            if const.mpi_rank % 2 == 0:
                self.op_sys_sites = right_op_blocks
            else:
                self.op_sys_sites = left_op_blocks
        self._validate_op_sys_sites()
        self._validate_initial_superblock_states()
        # (1) -> (2) communicate joint sigvec and L block
        self.send_op_sys_to_left(even_rank=True, matH=matH, pop_op_sys=True)
        op_sys_from_right = self.recv_op_sys_from_right(even_rank=False)
        comm.barrier()
        self.send_joint_sigvec_to_right(even_rank=False, truncate=False)
        self.recv_joint_sigvec_from_left(even_rank=True)
        comm.barrier()
        self.send_op_sys_to_right(even_rank=False)
        op_sys_from_left = self.recv_op_sys_from_left(even_rank=True)
        comm.barrier()
        # (2) -> (3)
        if const.mpi_rank % 2 == 0:
            begin_site = 0
            end_site = self.nsite - 1
            op_sys = op_sys_from_left
            if const.mpi_rank == const.mpi_size - 1:
                skip_end_site = False
            else:
                skip_end_site = True
                # skip_end_site = const.adaptive
        else:
            begin_site = self.nsite - 1
            end_site = 0
            op_sys = op_sys_from_right
            skip_end_site = True
            # skip_end_site = const.adaptive
            # if const.mpi_rank == 0:
            #     skip_end_site = False
            # else:
            #     skip_end_site = True
        time_sweep_start = time.time()
        self.propagate_along_sweep(
            ints_site,
            matH,
            stepsize,
            begin_site=begin_site,
            end_site=end_site,
            op_sys_initial=op_sys,
            skip_end_site=skip_end_site,
        )
        time_sweep_end = time.time()
        ms = (time_sweep_end - time_sweep_start) * 1000
        # share time to all ranks
        ms_ranks = comm.allgather(ms)
        if const.mpi_rank == 0 and load_balance:
            formatted_ms_ranks = [f"{ms:.2e}" for ms in ms_ranks]
            logger.info(f"Sweep time [ms]: {formatted_ms_ranks}")
        # (3) -> (4)
        self.send_Psi_to_left(even_rank=False)
        psi_L, psi_R = self.recv_Psi_from_right(even_rank=True)
        comm.barrier()
        self.send_op_sys_to_left(even_rank=False, matH=matH, pop_op_sys=False)
        op_env_previous = self.recv_op_sys_from_right(even_rank=True)
        comm.barrier()
        op_sys_from_right, Bsite = self.propagate_joint_two_sites(
            even_rank=True,
            matH=matH,
            stepsize=stepsize,
            op_env_previous=op_env_previous,
            psi_L=psi_L,
            psi_R=psi_R,
            truncate=True,
            # truncate=False,
        )
        self.send_B_to_right(even_rank=True, Bsite=Bsite)
        self.recv_B_from_left(even_rank=False)
        comm.barrier()
        self.save_all_A(even_rank=True)
        self.save_all_B(even_rank=False)
        # (4) -> (5)
        self.send_joint_sigvec_to_right(even_rank=True, truncate=False)
        self.recv_joint_sigvec_from_left(even_rank=False)
        comm.barrier()
        self.send_op_sys_to_right(even_rank=True)
        op_sys_from_left = self.recv_op_sys_from_left(even_rank=False)
        comm.barrier()
        # (5)=(3) -> (2)
        if const.mpi_rank % 2 == 0:
            begin_site = self.nsite - 1
            end_site = 0
            op_sys = op_sys_from_right
            if const.mpi_rank == 0:
                skip_end_site = False
            else:
                skip_end_site = True
                # skip_end_site = const.adaptive
        else:
            begin_site = 0
            end_site = self.nsite - 1
            op_sys = op_sys_from_left
            if const.mpi_rank == const.mpi_size - 1:
                skip_end_site = False
            else:
                skip_end_site = True
                # skip_end_site = const.adaptive
        self.propagate_along_sweep(
            ints_site,
            matH,
            stepsize,
            begin_site=begin_site,
            end_site=end_site,
            op_sys_initial=op_sys,
            skip_end_site=skip_end_site,
        )
        # (2) -> (1)
        self.send_Psi_to_left(even_rank=True)
        psi_L, psi_R = self.recv_Psi_from_right(even_rank=False)
        comm.barrier()
        self.send_op_sys_to_left(even_rank=True, matH=matH, pop_op_sys=False)
        op_env_previous = self.recv_op_sys_from_right(even_rank=False)
        comm.barrier()
        op_env_from_left, Bsite = self.propagate_joint_two_sites(
            even_rank=False,
            matH=matH,
            stepsize=stepsize,
            op_env_previous=op_env_previous,
            psi_L=psi_L,
            psi_R=psi_R,
            truncate=True,
            # truncate=False,
        )
        self.send_B_to_right(even_rank=False, Bsite=Bsite)
        self.recv_B_from_left(even_rank=True)
        comm.barrier()
        self.send_op_env_to_right(
            even_rank=False,
            op_env_from_left=op_env_from_left,
        )
        self.recv_op_env_from_left(even_rank=True)
        comm.barrier()
        self.save_all_A(even_rank=False)
        self.save_all_B(even_rank=True)
        # self.sync_world_canonicalize()

    def propagate_joint_two_sites(
        self,
        even_rank: bool,
        matH: TensorHamiltonian,
        stepsize: float,
        op_env_previous: dict[tuple[int, int], dict[_op_keys, _block_type]],
        psi_L: SiteCoef | None,
        psi_R: SiteCoef | None,
        truncate: bool,
    ) -> (
        tuple[dict[tuple[int, int], dict[_op_keys, _block_type]], SiteCoef]
        | tuple[None, None]
    ):
        """
        ΨxΨ -> ΨΨ -> ΨB -> AxB -> AΨ -> AxB
        """
        if (
            not is_update_rank(even_rank)
        ) or const.mpi_rank == const.mpi_size - 1:
            return None, None

        assert isinstance(self.op_sys_sites, list)
        assert isinstance(psi_L, SiteCoef)
        assert isinstance(psi_R, SiteCoef)
        assert psi_L.gauge == "Psi"
        assert psi_R.gauge == "Psi"
        l, c, r = psi_R.data.shape  # noqa: E741
        L, C, R = psi_L.data.shape  # noqa: E741
        assert l == R
        assert (
            (l, R)
            == self.joint_sigvec.shape
            == self.joint_sigvec_not_pinv.shape
        )
        op_sys_previous = self.op_sys_sites[-1]
        psi_L.data = multiply_sigvec_pinv(
            X=self.joint_sigvec_not_pinv,
            left_tensor=psi_L.data,
            right_tensor=None,
        )
        superblock = [psi_L, psi_R]
        # Psi_L, A_R = eval_PsiXpinvPsi(
        #     Psi_L=psi_L,
        #     X=self.joint_sigvec_not_pinv,
        #     Psi_R=psi_R,
        # )
        # superblock = [Psi_L, A_R]
        canonicalize(superblock, orthogonal_center=0)
        superblock_full = get_superblock_full(superblock, delta_rank=const.dD)
        superblock = [None] * (self.nsite - 1) + superblock
        superblock_full = [None] * (self.nsite - 1) + superblock_full
        if const.adaptive:
            newD, error, op_env_D_bra, op_env_D_braket = (
                self.get_adaptive_rank_and_block(
                    psite=self.nsite - 1,
                    superblock_states=[superblock],
                    superblock_states_full=[superblock_full],
                    op_env_previous=op_env_previous,
                    hamiltonian=matH,
                    to="->",
                )
            )
            op_env = op_env_D_bra
            tensor_shapes_out = (L, C, newD)
        else:
            op_env = self.renormalize_op_psite(
                psite=self.nsite,
                superblock_states=[superblock],
                op_block_states=op_env_previous,
                ints_site=None,
                hamiltonian=matH,
                A_is_sys=False,
            )
            tensor_shapes_out = None
        # propagate psi_L
        op_lcr = self.operators_for_superH(
            psite=self.nsite - 1,
            op_sys=op_sys_previous,
            op_env=op_env,
            ints_site=None,
            hamiltonian=matH,
            A_is_sys=True,
        )
        self.exp_superH_propagation_direct(
            psite=self.nsite - 1,
            superblock_states=[superblock],
            op_lcr=op_lcr,
            matH_cas=matH,
            stepsize=stepsize,
            tensor_shapes_out=tensor_shapes_out,
        )
        # propagate x
        svalues, op_sys = self.trans_next_psite_AsigmaB(
            psite=self.nsite - 1,
            superblock_states=[superblock],
            op_sys=op_sys_previous,
            ints_site=None,
            matH_cas=matH,
            PsiB2AB=True,
            regularize=True,  # False
        )
        if const.adaptive:
            op_env = op_env_D_braket
        else:
            pass
        op_lr = self.operators_for_superK(
            psite=self.nsite - 1,
            op_sys=op_sys,
            op_env=op_env,
            hamiltonian=matH,
            A_is_sys=True,
        )
        svalues = self.exp_superK_propagation_direct(
            op_lr=op_lr,
            hamiltonian=matH,
            svalues=svalues,
            stepsize=stepsize,
        )
        # propagate psi_R
        self.trans_next_psite_APsiB(
            psite=self.nsite - 1,
            superblock_states=[superblock],
            svalues=svalues,
            A_is_sys=True,
        )
        op_lcr = self.operators_for_superH(
            psite=self.nsite,
            op_sys=op_sys,
            op_env=op_env_previous,
            ints_site=None,
            hamiltonian=matH,
            A_is_sys=True,
        )
        self.exp_superH_propagation_direct(
            psite=self.nsite,
            superblock_states=[superblock],
            op_lcr=op_lcr,
            matH_cas=matH,
            stepsize=stepsize,
        )
        # propagate x
        svalues, op_env = self.trans_next_psite_AsigmaB(
            psite=self.nsite,
            superblock_states=[superblock],
            op_sys=op_env_previous,
            ints_site=None,
            matH_cas=matH,
            PsiB2AB=False,
        )
        op_lr = self.operators_for_superK(
            psite=self.nsite - 1,
            op_sys=op_sys,
            op_env=op_env,
            hamiltonian=matH,
            A_is_sys=True,
        )
        svalues = self.exp_superK_propagation_direct(
            op_lr=op_lr,
            hamiltonian=matH,
            svalues=svalues,
            stepsize=stepsize,
        )
        assert len(svalues) == 1
        Asite, Bsite = superblock[-2:]
        assert Asite.gauge == "A"
        assert Bsite.gauge == "B"
        if truncate:
            Asite, self.joint_sigvec, Bsite = truncate_sigvec(
                Asite=Asite,
                sigvec=svalues[0],
                Bsite=Bsite,
                p=const.p_svd,
                regularize=True,  # False
                keepdim=True,
            )
            superblock[-2] = Asite
            superblock[-1] = Bsite
            # construct R block AAAA
            op_sys = self.renormalize_op_psite(
                psite=self.nsite - 1,
                superblock_states=[superblock],
                op_block_states=op_sys_previous,
                ints_site=None,
                hamiltonian=matH,
                A_is_sys=True,
            )
            # construct L block BBBB
            op_env = self.renormalize_op_psite(
                psite=self.nsite,
                superblock_states=[superblock],
                op_block_states=op_env_previous,
                ints_site=None,
                hamiltonian=matH,
                A_is_sys=False,
            )
        else:
            self.joint_sigvec = svalues[0]
        self.joint_sigvec_not_pinv = self.joint_sigvec
        self.superblock_states[0][-1] = Asite
        self.op_sys_sites.append(op_sys)
        return op_env, Bsite

    def reset_left_op_blocks(self, matH: TensorHamiltonian):
        """
        Contract left block

        A-A-...A-
        | | .  |
        W-W-...W-
        | | .  |
        A-A-...A-
        """
        if const.mpi_rank == 0:
            op_initial_block = self.construct_op_zerosite(
                superblock_states=[self.superblock_all_A],
                operator=matH,
            )
        else:
            op_initial_block = recv_op_block(
                source=const.mpi_rank - 1, tag1=10, tag2=20
            )
        op_block_isites = self.construct_op_sites(
            superblock_states=[self.superblock_all_A],
            ints_site=None,
            begin_site=0,
            end_site=self.nsite
            if const.mpi_rank < const.mpi_size - 1
            else self.nsite - 1,
            matH_cas=matH,
            op_initial_block=op_initial_block,
        )
        if const.mpi_rank != const.mpi_size - 1:
            send_op_block(
                op_block_isites[-1], dest=const.mpi_rank + 1, tag1=10, tag2=20
            )
        return op_block_isites

    def reset_right_op_blocks(
        self, matH: TensorHamiltonian
    ) -> list[dict[tuple[int, int], dict[_op_keys, _block_type]]]:
        """
        Contract right block
        -B-...-B-B
         | .   | |
         W-...-W-W
         | .   | |
        -B-...-B-B
        """
        if const.mpi_rank == const.mpi_size - 1:
            op_initial_block = self.construct_op_zerosite(
                superblock_states=[self.superblock_all_B],
                operator=matH,
            )
        else:
            op_initial_block = recv_op_block(
                source=const.mpi_rank + 1, tag1=11, tag2=21
            )
        op_block_isites = self.construct_op_sites(
            superblock_states=[self.superblock_all_B],
            ints_site=None,
            begin_site=self.nsite - 1,
            end_site=-1 if const.mpi_rank > 0 else 0,
            matH_cas=matH,
            op_initial_block=op_initial_block,
        )
        if const.mpi_rank != 0:
            send_op_block(
                op_block_isites[-1], dest=const.mpi_rank - 1, tag1=11, tag2=21
            )
        return op_block_isites

    def send_joint_sigvec_to_right(self, even_rank: bool, truncate: bool):
        """
        Left rank     | Right rank
        A-A-...-A-x   |   B-...-B-B
        A-A-...-Ψ-x^+ | x-B-...-B-B
        or
        Left rank        | Right rank
        A-A-...-A-x      |     B-...-B-B
        A-A-...-A-U-σ-V  |     B-...-B-B
        A-A-...-A-U-σ-σ+ | σ-V-B-...-B-B
        """
        if (
            not is_update_rank(even_rank)
        ) or const.mpi_rank == const.mpi_size - 1:
            return
        # x -> xx^+x
        joint_sigvec = self.joint_sigvec
        superblock = self.superblock_states[0]
        assert superblock[-1].gauge == "A"
        if not isinstance(joint_sigvec, np.ndarray):
            raise NotImplementedError
        if truncate:
            U, s, Vh = truncate_sigvec(
                Asite=None,
                sigvec=joint_sigvec,
                Bsite=None,
                p=const.p_svd,
                regularize=False,
                keepdim=True,
            )
            joint_sigvec_to_left = U @ s
            joint_sigvec_to_right = s @ Vh
            self.joint_sigvec_not_pinv = s
            self.joint_sigvec = np.linalg.pinv(s)
        else:
            joint_sigvec_to_left = joint_sigvec
            joint_sigvec_to_right = joint_sigvec
            self.joint_sigvec_not_pinv = joint_sigvec
            self.joint_sigvec = np.linalg.pinv(joint_sigvec)
            if const.pytest_enabled:
                np.testing.assert_allclose(
                    joint_sigvec_to_left
                    @ self.joint_sigvec
                    @ joint_sigvec_to_right,
                    joint_sigvec,
                )
        comm.send(
            joint_sigvec_to_right,
            dest=const.mpi_rank + 1,
            tag=2,
        )
        superblock[-1].data = np.tensordot(
            superblock[-1].data,
            joint_sigvec_to_left,
            axes=(2, 0),
        )  # ijk,kl->ijl
        superblock[-1].gauge = "Psi"

    def recv_joint_sigvec_from_left(self, even_rank: bool):
        """
        Left rank       | Right rank
        A-A-...-Ψ-x^+-x |   B-...-B-B
        A-A-...-ψ-x^+   | x-B-...-B-B
        A-A-...-Ψ-x^+   |   Ψ-...-B-B
        """
        if (not is_update_rank(even_rank)) or const.mpi_rank == 0:
            return
        joint_sigvec = comm.recv(
            source=const.mpi_rank - 1,
            tag=2,
        )
        superblock = self.superblock_states[0]
        assert superblock[0].gauge == "B", f"{superblock[0]=}"
        if isinstance(joint_sigvec, np.ndarray):
            superblock[0].data = np.tensordot(
                joint_sigvec,
                superblock[0].data,
                axes=(1, 0),
            )  # ij,jkl->ikl
        else:
            raise NotImplementedError
        superblock[0].gauge = "Psi"

    def send_op_sys_to_right(
        self,
        even_rank: bool,
    ):
        """
        Sent following block to the right rank
        A-A-...A-
        | | .  |
        W-W-...W-
        | | .  |
        A-A-...A-
        """
        if (
            not is_update_rank(even_rank)
            or const.mpi_rank == const.mpi_size - 1
        ):
            return
        assert isinstance(self.op_sys_sites, list)
        op_sys = self.op_sys_sites.pop()
        send_op_block(op_sys, dest=const.mpi_rank + 1, tag1=13, tag2=23)

    def recv_op_sys_from_left(self, even_rank: bool):
        """
        Receive following block from the left rank
        A-A-...A-
        | | .  |
        W-W-...W-
        | | .  |
        A-A-...A-
        """
        if is_update_rank(even_rank) and const.mpi_rank != 0:
            return recv_op_block(source=const.mpi_rank - 1, tag1=13, tag2=23)
        else:
            return None

    def send_op_env_to_right(
        self,
        even_rank: bool,
        op_env_from_left: dict[tuple[int, int], dict[_op_keys, _block_type]],
    ):
        """
        Sent following block to the right rank
        -B-...
         |
        -W-...
         |
        -B-...
        """
        if (
            not is_update_rank(even_rank)
            or const.mpi_rank == const.mpi_size - 1
        ):
            return
        send_op_block(
            op_env_from_left, dest=const.mpi_rank + 1, tag1=14, tag2=24
        )

    def recv_op_env_from_left(self, even_rank: bool):
        """
        Receive following block from the left rank
        -B-...
         |
        -W-...
         |
        -B-...
        """
        if is_update_rank(even_rank) and const.mpi_rank != 0:
            op_sys = recv_op_block(source=const.mpi_rank - 1, tag1=14, tag2=24)
            assert isinstance(self.op_sys_sites, list)
            assert len(self.op_sys_sites) == self.nsite
            self.op_sys_sites.append(op_sys)
        else:
            return None

    def send_Psi_to_left(self, even_rank: bool):
        """
        Left rank     | Right rank
        A-A-...-Ψ-x   | ψ'-B-...-B-B
        A-A-...-Ψ-x-Ψ'|    B-...-B-B
        """
        if is_update_rank(even_rank) and const.mpi_rank != 0:
            superblock = self.superblock_states[0]
            assert superblock[0].gauge == "Psi", f"{superblock[0].gauge=}"
            comm.send(superblock[0], dest=const.mpi_rank - 1, tag=4)

    def recv_Psi_from_right(
        self, even_rank: bool
    ) -> tuple[SiteCoef, SiteCoef] | tuple[None, None]:
        """
        Left rank     | Right rank
        A-A-...-Ψ-x   | ψ'-B-...-B-B
        A-A-...-Ψ-x-Ψ'|    B-...-B-B
        """
        if (
            not is_update_rank(even_rank)
        ) or const.mpi_rank == const.mpi_size - 1:
            return None, None
        superblock = self.superblock_states[0]
        psi_R = comm.recv(source=const.mpi_rank + 1, tag=4)
        psi_L = superblock[-1]
        assert psi_L.gauge == "Psi"
        assert psi_R.gauge == "Psi"
        return psi_L, psi_R

    def send_B_to_right(self, even_rank: bool, Bsite: SiteCoef | None):
        """
        Left rank     | Right rank
        A-A-...-A-x-B |   B-...-B-B
        A-A-...-A-x   | B-B-...-B-B
        """
        if (
            not is_update_rank(even_rank)
        ) or const.mpi_rank == const.mpi_size - 1:
            return
        assert isinstance(Bsite, SiteCoef)
        assert Bsite.gauge == "B"
        comm.send(Bsite, dest=const.mpi_rank + 1, tag=3)

    def recv_B_from_left(self, even_rank: bool):
        """
        Left rank     | Right rank
        A-A-...-A-x-B |   B-...-B-B
        A-A-...-A-x   | B-B-...-B-B
        """
        if (not is_update_rank(even_rank)) or const.mpi_rank == 0:
            return
        Bsite = comm.recv(source=const.mpi_rank - 1, tag=3)
        assert isinstance(Bsite, SiteCoef)
        assert Bsite.gauge == "B"
        self.superblock_states[0][0] = Bsite

    def send_op_sys_to_left(
        self, even_rank: bool, matH: TensorHamiltonian, pop_op_sys: bool
    ) -> None:
        """
        Sent following R block to the left rank
            -B-B-B
             | | |
        R=  -W-W-W
             | | |
            -B-B-B

        Args:
            even_rank: whether even rank will send the block
            matH: the Hamiltonian
            pop_op_sys: whether to pop the op_sys_sites
        """
        assert isinstance(self.op_sys_sites, list)
        if (not is_update_rank(even_rank)) or const.mpi_rank == 0:
            return
        if len(self.op_sys_sites) == self.nsite + 1 and pop_op_sys:
            # left superblock = BBBBBB and op_sys has already calculated
            assert self.superblock_states[0][0].gauge == "B"
            op_sys = self.op_sys_sites.pop()
        elif len(self.op_sys_sites) == self.nsite and not pop_op_sys:
            match self.superblock_states[0][0].gauge:
                case "B":
                    # left superblock = BBBBBB
                    op_sys = self.renormalize_op_psite(
                        psite=0,
                        superblock_states=self.superblock_states,
                        op_block_states=self.op_sys_sites[-1],
                        ints_site=None,
                        hamiltonian=matH,
                        A_is_sys=False,
                        superblock_states_ket=None,
                    )
                case "Psi":
                    # left superblock = PsiBBBBB
                    assert self.superblock_states[0][1].gauge == "B"
                    op_sys = self.op_sys_sites[-1]
                case _:
                    raise ValueError(f"{self.superblock_states[0][0].gauge=}")
        else:
            raise ValueError(
                f"{len(self.op_sys_sites)=} {self.nsite=} {self.superblock_states=}"
            )
        send_op_block(op_sys, dest=const.mpi_rank - 1, tag1=15, tag2=25)

    def recv_op_sys_from_right(self, even_rank: bool):
        if is_update_rank(even_rank) and const.mpi_rank != const.mpi_size - 1:
            return recv_op_block(source=const.mpi_rank + 1, tag1=15, tag2=25)
        else:
            return None

    def save_all_A(self, even_rank: bool):
        if not is_update_rank(even_rank):
            return
        self.superblock_all_A = [
            SiteCoef(
                isite=core.isite,
                data=core.data.copy(),
                gauge=core.gauge,
            )
            for core in self.superblock_states[0]
        ]
        if const.mpi_rank != const.mpi_size - 1:
            assert all(core.gauge == "A" for core in self.superblock_all_A), (
                f"Found {[core.gauge for core in self.superblock_all_A]} in superblock_all_A"
            )
        else:
            assert all(
                core.gauge == "A" for core in self.superblock_all_A[:-1]
            ), (
                f"Found {[core.gauge for core in self.superblock_all_A[:-1]]} in superblock_all_A"
            )
            assert self.superblock_all_A[-1].gauge == "Psi"

    def save_all_B(self, even_rank: bool):
        if not is_update_rank(even_rank):
            return
        self.superblock_all_B = [
            SiteCoef(
                isite=core.isite,
                data=core.data.copy(),
                gauge=core.gauge,
            )
            for core in self.superblock_states[0]
        ]
        if const.mpi_rank != 0:
            assert all(core.gauge == "B" for core in self.superblock_all_B), (
                f"Found {[core.gauge for core in self.superblock_all_B]} in superblock_all_B"
            )
        else:
            assert all(
                core.gauge == "B" for core in self.superblock_all_B[1:]
            ), (
                f"Found {[core.gauge for core in self.superblock_all_B[1:]]} in superblock_all_B"
            )
            assert self.superblock_all_B[0].gauge == "Psi"

    def ovlp(self, conj=True, lazy=False) -> complex | float | None:
        rank = const.mpi_rank
        size = const.mpi_size
        mid_rank = size // 2

        is_forward_group = rank < mid_rank
        if const.use_jax:
            raise NotImplementedError
        if lazy:
            if is_forward_group:
                superblock_states = self.superblock_all_A
            else:
                superblock_states = self.superblock_all_B
            superblock_data = [core.data.copy() for core in superblock_states]
        else:
            superblock_data = [
                core.data.copy()
                if i == len(self.superblock_states[0]) - 1
                else core.data
                for i, core in enumerate(self.superblock_states[0])
            ]

        if not lazy and const.mpi_rank != const.mpi_size - 1:
            # einsum ijk,kl->ijl
            if (
                const.mpi_rank % 2 == 0
                and self.superblock_states[0][-1].gauge == "B"
            ):
                # BBBBBBx+
                assert isinstance(self.joint_sigvec_not_pinv, np.ndarray)
                assert isinstance(superblock_data[-1], np.ndarray)
                superblock_data[-1] = multiply_sigvec_pinv(
                    X=self.joint_sigvec_not_pinv,
                    left_tensor=superblock_data[-1],
                    right_tensor=None,
                )
            elif (
                const.mpi_rank % 2 == 1
                and self.superblock_states[0][-1].gauge == "A"
            ):
                # AAAAAx
                superblock_data[-1] = np.tensordot(
                    superblock_data[-1], self.joint_sigvec, axes=(2, 0)
                )
            else:
                raise ValueError(
                    f"{self.superblock_states[0][-1].gauge=} {const.mpi_rank=}"
                )
        ket_cores: list[np.ndarray] = [ket for ket in superblock_data]
        if conj:
            bra_cores: list[np.ndarray] = [
                bra.conj() for bra in superblock_data
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
                comm.send(self.joint_sigvec_not_pinv, dest=0, tag=10)
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
            if lazy:
                if mid_rank - 1 == 0:
                    sigvec = self.joint_sigvec_not_pinv
                else:
                    sigvec = comm.recv(source=mid_rank - 1, tag=10)
                if conj:
                    """
                    |‾˙˙˙‾|‾b b‾d d‾|‾˙˙˙‾|
                    |_..._|_a a_c c_|_..._|
                    """
                    final_result = np.einsum(
                        "ba,ac,bd,dc->",
                        block_left,
                        sigvec,
                        sigvec.conj(),
                        block_right,
                    )
                else:
                    final_result = np.einsum(
                        "ba,ac,bd,dc->", block_left, sigvec, sigvec, block_right
                    )
            else:
                final_result = np.einsum("ab,ab->", block_left, block_right)
            if conj:
                return final_result.real.item()
            else:
                return complex(final_result)
        else:
            return None

    def autocorr(
        self, ints_spf: SPFInts, psite: int = 0, repeat=False
    ) -> complex | None:
        autocorr = self.ovlp(conj=False)
        if const.mpi_rank == 0:
            assert isinstance(autocorr, complex)
            if abs(autocorr) > 1.0 + 1e-04:
                logger.info(f"MPS is reorhogonalized when {autocorr=}")
                reortho_flag = True
            else:
                reortho_flag = False
        else:
            reortho_flag = None
        # share reortho_flag to all ranks
        if repeat and const.mpi_rank == 0:
            logger.info(f"After reorthogonalization, {autocorr=}")
        reortho_flag = comm.bcast(reortho_flag, root=0)
        if reortho_flag and not repeat:
            self.sync_world_canonicalize()
            return self.autocorr(ints_spf, psite, repeat=True)
        else:
            return autocorr

    def norm(self, repeat=False) -> float | None:
        norm = self.ovlp(conj=True)
        if const.mpi_rank == 0:
            assert isinstance(norm, float)
            if abs(norm - 1.0) > 1e-03:
                logger.info(f"MPS is reorhogonalized when {norm=:.12f}")
                reortho_flag = True
            else:
                reortho_flag = False
        else:
            reortho_flag = None
        # share reortho_flag to all ranks
        reortho_flag = comm.bcast(reortho_flag, root=0)
        if repeat and const.mpi_rank == 0:
            logger.info(f"After reorthogonalization, {norm=:.12f}")
        if reortho_flag and not repeat:
            self.sync_world_canonicalize()
            return self.norm(repeat=True)
        else:
            return norm

    def pop_states(self) -> list[float | None]:
        # Use reduced density and second quantization operator instead.
        # Only support single MPS for now.
        return [self.norm()]

    @mpi_abort_on_exception
    def get_reduced_densities(
        self, base_tag: int, rd_key: tuple[int, ...]
    ) -> np.ndarray | None:
        """
        When rd_key is  (3, 3, 4)
        and MPS is A1A2A3A4...A6
        Contract (A1A1†)(A2A2†)=1 and (B5B5†)(B6B6†)=1 in advance (no calculation needed)
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
        # needed_rank = [const.mpi_rank]
        # opend_legs = [counter[isite] for isite in range(const.bgn_site_rank, const.end_site_rank + 1)]
        if const.mpi_rank != 0 and const.mpi_rank not in needed_rank:
            return None
        mid_rank = (len(needed_rank) - 1) // 2 + needed_rank[0]
        # mid_rank = const.mpi_size // 2
        is_forward_group = (
            const.mpi_rank <= mid_rank and const.mpi_rank in needed_rank
        )
        is_backward_group = (
            const.mpi_rank > mid_rank and const.mpi_rank in needed_rank
        )
        if is_forward_group:
            superblock = self.superblock_all_A
        elif is_backward_group:
            superblock = self.superblock_all_B
        elif const.mpi_rank == 0:
            pass
        else:
            raise ValueError(f"{const.mpi_rank=} {needed_rank=}")

        block = None
        if is_forward_group:
            if const.mpi_rank == needed_rank[0]:
                # if const.mpi_rank == 0:
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
                # if const.mpi_rank == const.mpi_size - 1:
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
            assert isinstance(left_block, np.ndarray), (
                f"{left_block=} {const.mpi_rank=}"
            )
            if const.mpi_rank == const.mpi_size - 1:
                # No sigvecs
                sigvec = np.eye(superblock[-1].data.shape[2], dtype=complex)
            else:
                sigvec = self.joint_sigvec_not_pinv
            assert isinstance(sigvec, np.ndarray), f"{sigvec=}"
            if len(needed_rank) == 1:
                right_block = np.eye(sigvec.shape[0], dtype=complex)
            else:
                right_block = comm.recv(
                    source=const.mpi_rank + 1, tag=base_tag + 1
                )
                assert isinstance(right_block, np.ndarray), f"{right_block=}"
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
            assert rd.shape == left_block.shape[:-2] + right_block.shape[2:], (
                f"{rd.shape=} {left_block.shape=} {right_block.shape=}"
            )
            if mid_rank == 0:
                return rd
            else:
                comm.send(rd, dest=0, tag=base_tag + 2)
                return None
        if const.mpi_rank == 0:
            rd = comm.recv(source=mid_rank, tag=base_tag + 2)
            assert isinstance(rd, np.ndarray), f"{rd=}"
            return rd
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
            superblock_states = [self.superblock_all_A]
        else:
            superblock_states = [self.superblock_all_B]

        if is_forward_group:
            if rank == 0:
                op_A_block = None
            else:
                op_A_block = recv_op_block(source=rank - 1, tag1=10, tag2=20)

            op_A_block = self.construct_op_sites(
                superblock_states=superblock_states,
                ints_site=None,
                begin_site=0,
                end_site=self.nsite,
                matH_cas=matOp,
                op_initial_block=op_A_block,
            ).pop()
            assert isinstance(op_A_block, dict)
            if rank != mid_rank - 1:
                send_op_block(op_A_block, dest=rank + 1, tag1=10, tag2=20)
            elif rank == mid_rank - 1:
                pass
            else:
                raise ValueError(f"{rank=}")
        else:
            if rank == size - 1:
                op_B_block = None
            else:
                op_B_block = recv_op_block(source=rank + 1, tag1=11, tag2=21)

            op_B_block = self.construct_op_sites(
                superblock_states=superblock_states,
                ints_site=None,
                begin_site=self.nsite - 1,
                end_site=-1,
                matH_cas=matOp,
                op_initial_block=op_B_block,
            ).pop()
            assert isinstance(op_B_block, dict)
            if rank >= mid_rank:
                send_op_block(op_B_block, dest=rank - 1, tag1=11, tag2=21)
            else:
                raise ValueError(f"{rank=}")

        if rank == mid_rank - 1:
            # calculate expectation value
            left_block = op_A_block
            right_block = recv_op_block(source=rank + 1, tag1=11, tag2=21)
            # if self.superblock_states[0][-1].gauge == "A":
            #     # AAA => sigvec is x
            #     sigvec: list[np.ndarray] | list[jax.Array] = [self.joint_sigvec]
            # elif self.superblock_states[0][-1].gauge == "B":
            #     # PsiBBB or BBB => sigvec is x^+
            #     sigvec = [np.linalg.pinv(self.joint_sigvec)]
            # else:
            #     raise ValueError(f"{self.superblock_states[0][-1].gauge=}")
            sigvec: list[np.ndarray] | list[jax.Array] = [
                self.joint_sigvec_not_pinv
            ]
            op_lr = self.operators_for_superK(
                psite=self.nsite - 1,
                op_sys=left_block,
                op_env=right_block,
                hamiltonian=matOp,
                A_is_sys=True,
            )
            multiplyK = multiplyK_MPS_direct_MPO(
                op_lr_states=op_lr,
                psi_states=sigvec,
                hamiltonian=matOp,
            )
            expectation_value = _integrator.expectation_Op(
                sigvec,
                multiplyK,
                sigvec,
            )
            if rank == 0:
                return complex(expectation_value)
            else:
                comm.send(complex(expectation_value), dest=0, tag=2)
        if rank == 0:
            return comm.recv(source=mid_rank - 1, tag=2)
        return None

    @mpi_abort_on_exception
    def sync_world_canonicalize(self):
        return
        self._sync_world_canonicalizeB()  # [PsiBBx+][AAAx][BBB] -> [PsiBB][BBB][BBB]

        # [AAAx][BBB][BBB] -> [AAA][xBBB][BBB]
        if const.mpi_rank == 0:
            recv_joint_sigvec = np.ones((1, 1), dtype=complex)
        else:
            recv_joint_sigvec = comm.recv(source=const.mpi_rank - 1, tag=0)
        if const.mpi_rank % 2 == 0:
            self.superblock_states = [self.superblock_all_B]
        # [AAA][xBBB][BBB] -> [AAA][PsiBB][BBB]
        superblock = [core.copy() for core in self.superblock_states[0]]
        assert (
            superblock[0].gauge == "B"
            if const.mpi_rank > 0
            else superblock[0].gauge == "Psi"
        )
        l_core = superblock[0]
        l_core.data = np.tensordot(recv_joint_sigvec, l_core.data, axes=(1, 0))
        # einsum ij,jkl->ikl
        l_core.gauge = "Psi"
        # [AAA][PsiBB][BBB] -> [AAA][AAPsi][BBB]
        canonicalize(superblock, orthogonal_center=self.nsite - 1)
        assert superblock[-1].gauge == "Psi"
        if const.mpi_rank != const.mpi_size - 1:
            # [AAA][AAPsi][BBB] -> [AAA][AAAx][BBB]
            superblock[-1], joint_sigvec = superblock[-1].gauge_trf(
                "Psi2Asigma",
            )
            # This part may be numerically unstable when the singular values are close to zero
            if const.mpi_rank % 2 == 1:
                # AAAxBBB
                self.joint_sigvec_not_pinv = joint_sigvec
                self.joint_sigvec = joint_sigvec
            else:
                # BBBx+AAA
                U, S, Vh = np.linalg.svd(joint_sigvec)
                self.joint_sigvec_not_pinv = joint_sigvec
                self.joint_sigvec = np.linalg.pinv(joint_sigvec)
            comm.send(joint_sigvec, dest=const.mpi_rank + 1, tag=0)
        if const.mpi_rank % 2 == 1:
            self.superblock_states = [superblock]
        self.superblock_all_A = [core.copy() for core in superblock]
        self.op_sys_sites = None
        # wait all communication
        comm.barrier()

    @mpi_abort_on_exception
    def _sync_world_canonicalizeB(self):
        # PsiBBBx+AAAAxBBBBx+AAAPsi -> PsiBBB1BBBB1BBBB1BBBB
        self._validate_initial_superblock_states()
        self.superblock_all_A = None
        if const.mpi_rank == const.mpi_size - 1:
            joint_sigvec: np.ndarray = np.ones((1, 1), dtype=complex)
        else:
            joint_sigvec = comm.recv(source=const.mpi_rank + 1, tag=1)
            if const.mpi_rank % 2 == 0:
                # This part is numericall unstable !!!
                assert isinstance(self.joint_sigvec_not_pinv, np.ndarray)
                joint_sigvec = multiply_sigvec_pinv(
                    X=self.joint_sigvec_not_pinv,
                    left_tensor=None,
                    right_tensor=joint_sigvec,
                )
            else:
                joint_sigvec = self.joint_sigvec @ joint_sigvec
            self.joint_sigvec = np.eye(joint_sigvec.shape[0], dtype=complex)
            self.joint_sigvec_not_pinv = self.joint_sigvec
        assert isinstance(joint_sigvec, np.ndarray)
        superblock = [core.copy() for core in self.superblock_states[0]]
        r_core = superblock[-1]
        r_core.data = np.tensordot(r_core.data, joint_sigvec, axes=(2, 0))
        # einsum ijk,kl->ijl
        if const.mpi_rank != const.mpi_size - 1:
            r_core.gauge = "Psi"
        canonicalize(superblock, orthogonal_center=0)
        assert superblock[0].gauge == "Psi"
        if const.mpi_rank == 0:
            l_core = superblock[0]
            norm = np.einsum(
                "jk,jk->",
                l_core.data[0, :, :].conj(),
                l_core.data[0, :, :],
            )
            l_core.data /= np.sqrt(norm)
        else:
            superblock[0], joint_sigvec = superblock[0].gauge_trf("Psi2sigmaB")
            comm.send(joint_sigvec, dest=const.mpi_rank - 1, tag=1)
        self.superblock_all_B = superblock
        self.superblock_states = [[core.copy() for core in superblock]]
        self.op_sys_sites = None
        # wait all communication
        comm.barrier()

    def sync_world_AB(self):
        # collect all_A and all_B from all ranks and store in rank0
        recv_all_A: list[list[SiteCoef]] = comm.gather(
            self.superblock_all_A, root=0
        )
        recv_all_B: list[list[SiteCoef]] = comm.gather(
            self.superblock_all_B, root=0
        )
        if const.mpi_rank == 0:
            self.superblock_all_B_world: list[SiteCoef] = []
            self.superblock_all_A_world: list[SiteCoef] = []
            for all_A, all_B in zip(recv_all_A, recv_all_B, strict=True):
                self.superblock_all_A_world.extend(all_A)
                self.superblock_all_B_world.extend(all_B)
        # wait all communication
        comm.barrier()

    def to_MPSCoefMPO(self) -> MPSCoefMPO | None:
        self.sync_world_canonicalize()
        self.sync_world_AB()
        if const.mpi_rank == 0:
            mps = MPSCoefMPO()
            for attr in [
                "dofs_cas",
                "lattice_info_states",
                "nstate",
                "ndof_per_sites",
            ]:
                setattr(mps, attr, getattr(self, attr))
            mps_superblock_states = [self.superblock_all_B_world]
            mps.superblock_states = mps_superblock_states
            mps.nsite = len(mps_superblock_states[0])
            return mps
        else:
            return None

    def _validate_op_sys_sites(self):
        assert isinstance(self.op_sys_sites, list)
        if const.mpi_rank == 0 or (
            const.mpi_rank == const.mpi_size - 1 and const.mpi_size % 2 == 0
        ):
            assert len(self.op_sys_sites) == self.nsite
        else:
            assert len(self.op_sys_sites) == self.nsite + 1

    def _validate_initial_superblock_states(self):
        if not const.pytest_enabled:
            return
        # for rank0
        # PsiBBBB
        # for even rank except 0
        # BBBBBBB
        # for odd rank except size-1
        # AAAAAAA
        # for size-1
        # AAAAAAPsi or BBBBBBB
        if const.mpi_rank == 0:
            assert self.superblock_states[0][0].gauge == "Psi"
            assert all(
                core.gauge == "B" for core in self.superblock_states[0][1:]
            )
        elif const.mpi_rank % 2 == 0:
            assert all(core.gauge == "B" for core in self.superblock_states[0])
        elif const.mpi_rank == const.mpi_size - 1:
            assert self.superblock_states[0][-1].gauge == "Psi"
            assert all(
                core.gauge == "A" for core in self.superblock_states[0][:-1]
            )
        elif const.mpi_rank % 2 == 1:
            assert all(core.gauge == "A" for core in self.superblock_states[0])
        else:
            raise ValueError(f"{const.mpi_rank=}")


def _ovlp_single_state_np_from_left(
    bra_cores: list[np.ndarray],
    ket_cores: list[np.ndarray],
    _block: np.ndarray | None = None,
) -> np.ndarray:
    """
    -b b-|-d
    |    e
    -c c-|-f
    """
    a = bra_cores[0]
    b = ket_cores[0]
    if isinstance(_block, np.ndarray):
        block = np.einsum("bed,cef,bc->df", a, b, _block)
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
    """
    d-|-a a-
      e    |
    f-|-c c-
    """
    a = bra_cores[-1]
    b = ket_cores[-1]
    if isinstance(_block, np.ndarray):
        block = np.einsum("dea,fec,ac->df", a, b, _block)
    else:
        block = np.einsum("ab,cb->ac", a[:, :, 0], b[:, :, 0])
    assert isinstance(block, np.ndarray)
    for i in range(len(bra_cores) - 2, -1, -1):
        a = bra_cores[i]
        b = ket_cores[i]
        block = np.einsum("dea,fec,ac->df", a, b, block)
    return block


def distribute_superblock_states(mps_coef: MPSCoefParallel) -> MPSCoefParallel:
    if const.mpi_rank == 0:
        superblock = mps_coef.superblock_all_B_world
        canonicalize(superblock, orthogonal_center=0)
        mps_coef.superblock_all_B_world = superblock

        joint_sigvecs: list[np.ndarray] | list[jax.Array] = []
        split_indices = const.split_indices
        superblock_copy = [core.copy() for core in superblock]
        for i in range(comm.size - 1):
            canonicalize(
                superblock_copy,
                orthogonal_center=split_indices[i + 1] - 1,
                incremental=True,
            )
            Psi = superblock_copy[split_indices[i + 1] - 1]
            B = superblock_copy[split_indices[i + 1]]
            Lambda = CC2ALambdaB(Psi, B)
            B.gauge = "Psi"
            if isinstance(B.data, np.ndarray):
                B.data = np.einsum("a,abc->abc", Lambda, B.data)
            else:
                B.data = jnp.einsum("a,abc->abc", Lambda, B.data)
            if isinstance(Lambda, np.ndarray):
                if i % 2 == 0:
                    joint_sigvec = np.linalg.pinv(np.diag(Lambda))
                else:
                    joint_sigvec = np.diag(Lambda)
            else:
                if i % 2 == 0:
                    joint_sigvec = jnp.linalg.pinv(jnp.diag(Lambda))
                else:
                    joint_sigvec = jnp.diag(Lambda)
            joint_sigvecs.append(joint_sigvec)
        canonicalize(
            superblock_copy,
            orthogonal_center=len(superblock_copy) - 1,
            incremental=True,
        )
        mps_coef.superblock_all_A_world = superblock_copy

        send_joint_sigvecs = [None] * comm.size
        for i in range(comm.size - 1):
            send_joint_sigvecs[i] = joint_sigvecs[i]

        send_superblock_all_B = []
        send_superblock_all_A = []
        for i in range(comm.size):
            if i < len(split_indices) - 1:
                rank_superblock_all_B = mps_coef.superblock_all_B_world[
                    split_indices[i] : split_indices[i + 1]
                ]
                rank_superblock_all_A = mps_coef.superblock_all_A_world[
                    split_indices[i] : split_indices[i + 1]
                ]
            else:
                rank_superblock_all_B = mps_coef.superblock_all_B_world[
                    split_indices[i] :
                ]
                rank_superblock_all_A = mps_coef.superblock_all_A_world[
                    split_indices[i] :
                ]
            send_superblock_all_B.append(rank_superblock_all_B)
            send_superblock_all_A.append(rank_superblock_all_A)
    else:
        send_joint_sigvecs = None
        send_superblock_all_B = None
        send_superblock_all_A = None

    recv_joint_sigvec = comm.scatter(send_joint_sigvecs, root=0)
    if const.mpi_rank != comm.size - 1:
        mps_coef.joint_sigvec = recv_joint_sigvec.astype(complex)
        mps_coef.joint_sigvec_not_pinv = mps_coef.joint_sigvec

    recv_superblock_all_B = comm.scatter(send_superblock_all_B, root=0)
    recv_superblock_all_A = comm.scatter(send_superblock_all_A, root=0)
    mps_coef.superblock_all_B = recv_superblock_all_B
    mps_coef.superblock_all_A = recv_superblock_all_A
    if const.mpi_rank % 2 == 0:
        mps_coef.superblock_states = [
            [core.copy() for core in recv_superblock_all_B]
        ]
    else:
        mps_coef.superblock_states = [
            [core.copy() for core in recv_superblock_all_A]
        ]
    mps_coef.nsite = const.end_site_rank - const.bgn_site_rank + 1
    return mps_coef


def send_op_block(
    op_block: dict[tuple[int, int], dict[_op_keys, _block_type]],
    dest: int,
    tag1: int = 10,
    tag2: int = 20,
):
    comm.send(op_block, dest=dest, tag=tag1)
    if (0, 0) in op_block and "ovlp" in op_block[(0, 0)]:
        assert isinstance(op_block[(0, 0)]["ovlp"], myndarray)
        comm.send(
            op_block[(0, 0)]["ovlp"].is_identity,
            dest=dest,
            tag=tag2,
        )


def recv_op_block(
    source: int, tag1: int = 10, tag2: int = 20
) -> dict[tuple[int, int], dict[_op_keys, _block_type]]:
    op_block = comm.recv(source=source, tag=tag1)
    if (0, 0) in op_block and "ovlp" in op_block[(0, 0)]:
        is_identity = comm.recv(source=source, tag=tag2)
        op_block[(0, 0)]["ovlp"] = myndarray(
            op_block[(0, 0)]["ovlp"], is_identity=is_identity
        )
    return op_block


def is_update_rank(even_rank: bool) -> bool:
    if even_rank:
        return const.mpi_rank % 2 == 0
    else:
        return const.mpi_rank % 2 == 1
