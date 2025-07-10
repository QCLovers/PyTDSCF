"""The main simulator module of PyTDSCF

This module consists of Simulator class.

"""

import os
import pickle
from copy import deepcopy
from time import time
from typing import Any, Literal

import dill
from loguru import logger as _logger
from tqdm.auto import tqdm

import pytdscf._helper as helper
from pytdscf import units
from pytdscf._const_cls import const
from pytdscf._mps_cls import MPSCoef
from pytdscf._mps_mpo import MPSCoefMPO
from pytdscf._mps_parallel import MPSCoefParallel
from pytdscf._mps_sop import MPSCoefSoP
from pytdscf._spf_cls import SPFCoef
from pytdscf.basis._primints_cls import PrimInts
from pytdscf.hamiltonian_cls import TensorHamiltonian
from pytdscf.model_cls import Model
from pytdscf.properties import Properties
from pytdscf.wavefunction import WFunc

logger = _logger.bind(name="main")


class Simulator:
    """The simulator of the PyTDSCF

    set parameter of the restart, propagate, operate dipole, save_file etc ...

    Args:
        jobname (str) : the jobname
        model (model_cls.Model) : run parameter (basis, hamiltonian, \
                observable, bond dimension, initial_weight etc.)
        t2_trick (bool, optional) : Use so-called t/2 trick in auto-correlation. \
                Note that it requires initial state to be real. Defaults to ``True``.
        ci_type (str, optional)  ``'mps'`, ``'mcdth'``. Defaults to ``'MPS'``
        backend (str, optional): JAX or Numpy. Defaults to ``'jax'``.
            When polynomial operator, FBR basis and small bond-dimension is used, \
                ``'Numpy'`` is recommended.
        proj_gs (bool, optional) : Initial state is projected from the ground state. Defaults to ``False``. \
            If ``proj_gs=True``, one must be set attribute ``model.primbas_gs: List[Primbas_HO]``.

    """

    backup_interval: int
    stepsize: float
    maxstep: int

    def __init__(
        self,
        jobname: str,
        model: Model,
        ci_type: Literal["mps", "mctdh", "ci"] = "mps",
        backend: Literal["jax", "numpy"] = "jax",
        proj_gs: bool = False,
        t2_trick: bool = True,
        verbose: int = 2,
    ):
        if backend.lower() == "jax":
            self.use_jax = True
        elif backend.lower() == "numpy":
            self.use_jax = False
        else:
            raise ValueError(
                f"backend must be JAX or Numpy, but {backend} is given."
            )
        self.model = model
        self.jobname = jobname
        self.t2_trick = t2_trick
        self.doPrint = False
        self.doSpectra = True
        self.ci_type = ci_type
        self.do_init_proj_gs = proj_gs
        self.verbose = verbose
        if proj_gs and not hasattr(model, "primbas_gs"):
            raise ValueError(
                "If proj_gs is True, one must be set attribute model.primbas_gs: List[PrimBas_HO]"
            )
        if self.verbose > 2:
            logger.debug(
                f"doPrint:{self.doPrint} doSpectra:{self.doSpectra} "
                + f"ci_type:{self.ci_type}"
            )

    def relax(
        self,
        stepsize: float = 0.1,
        maxstep: int = 20,
        improved: bool = True,
        restart: bool = False,
        savefile_ext: str = "_gs",
        loadfile_ext: str = "",
        backup_interval: int = 10,
        norm: bool = True,
        populations: bool = True,
        observables: bool = False,
        integrator: Literal["lanczos", "arnoldi"] = "lanczos",
    ) -> tuple[float, WFunc]:
        """Relaxation

        Args:
            stepsize (float, optional): Step size in fs. Defaults to ``0.1``.\
                This is used only when imaginary time propagation is used.
            maxstep (int, optional): Maximum number of steps. Defaults to ``20``.
            improved (bool, optional): Use improved relaxation. Defaults to ``True``.
            restart (bool, optional): Restart from the previous wavefunction. Defaults to ``False``.
            savefile_ext (str, optional): Extension of the save file. Defaults to ``'_gs'``.
            loadfile_ext (str, optional): Extension of the load file. Defaults to ``''``. \
                When ``restart=False``, ``loadfile_ext`` is ignored.
            backup_interval (int, optional): Number of steps at which, the wavefunction is saved. \
                Defaults to ``10``.
            norm (bool, optional): Calculate norm. Defaults to ``True``.
            populations (bool, optional): Calculate populations. Defaults to ``True``.
            observables (bool, optional): Calculate observables. Defaults to ``False``.

        Returns:
            Tuple[float, WFunc]: Energy after relaxation in Eh, and Wavefunction after relaxation.

        """
        self.stepsize = stepsize
        self.maxstep = maxstep
        self.backup_interval = backup_interval
        autocorr = False
        energy = True
        if improved:
            relax: bool | str = "improved"
        else:
            relax = True
        const.set_runtype(
            jobname=self.jobname + "_relax",
            restart=restart,
            relax=relax,
            dvr=self.model.basinfo.is_DVR,
            savefile_ext=savefile_ext,
            loadfile_ext=loadfile_ext,
            maxstep=self.maxstep,
            use_jax=self.use_jax,
            standard_method=self.model.basinfo.is_standard_method,
            verbose=self.verbose,
            use_mpo=self.model.use_mpo,
            space=self.model.space,
            integrator=integrator,
        )
        return self._execute(autocorr, energy, norm, populations, observables)

    def propagate(
        self,
        stepsize: float = 0.1,
        maxstep: int = 5000,
        restart: bool = False,
        savefile_ext: str = "",
        loadfile_ext: str = "_operate",
        backup_interval: int = 1000,
        autocorr: bool = True,
        energy: bool = True,
        norm: bool = True,
        populations: bool = True,
        observables: bool = False,
        reduced_density: tuple[list[tuple[int, ...]], int] | None = None,
        Δt: float | None = None,
        thresh_sil: float = 1.0e-09,
        autocorr_per_step: int = 1,
        observables_per_step: int = 1,
        energy_per_step: int = 1,
        norm_per_step: int = 1,
        populations_per_step: int = 1,
        parallel_split_indices: list[tuple[int, int]] | None = None,
        adaptive: bool = False,
        adaptive_Dmax: int = 20,
        adaptive_dD: int = 5,
        adaptive_p_proj: float = 1.0e-04,
        adaptive_p_svd: float = 1.0e-07,
        integrator: Literal["lanczos", "arnoldi"] = "lanczos",
        step_size_is_fs: bool = True,
        conserve_norm: bool = True,
    ) -> tuple[float, WFunc]:
        r"""Propagation

        Args:
            stepsize (float, optional): Step size in fs. Defaults to ``0.1``.
            maxstep (int, optional): Maximum number of steps. Defaults to ``5000``., \
                i.e. 500 fs.
            restart (bool, optional): Restart from the previous wavefunction. \
                Defaults to ``False``.
            savefile_ext (str, optional): Extension of the save file. Defaults to ``''``.
            loadfile_ext (str, optional): Extension of the load file. Defaults to ``'_operate'``. \
                When ``restart=False``, ``loadfile_ext`` is ignored.
            backup_interval (int, optional): Number of steps at which, the wavefunction is saved. \
                Defaults to ``1000``.
            autocorr (bool, optional): Calculate autocorrelation function. Defaults to ``True``.
            energy (bool, optional): Calculate energy. Defaults to ``True``.
            norm (bool, optional): Calculate norm. Defaults to ``True``.
            populations (bool, optional): Calculate populations. Defaults to ``True``.
            observables (bool, optional): Calculate observables. Defaults to ``False``.
            reduced_density (Dict[Tuple[int, ...], int], optional): Calculate reduced density of the \
                given modes.
                For example, ``([(0, 1),], 10)`` means calculate the diagonal elements of reduced density of the \
                :math:`\rho_{01}=\mathrm{Tr}_{p\notin \{0,1\}}\left|\Psi^{(\alpha)}\rangle\langle\Psi^(\alpha)\right|` \
                in per 10 steps.
                Note that it requires enough disk space.
                Defaults to ``None``.
                It is better if the target modes are close to rightmost, i.e., 0. \
                (Because this program calculate property in the most right-canonized form of MPS.)
                If you want coherence, i.e., off-diagonal elements of density matrix, \
                you need to set like ``([(0, 0), ], 10)``.
            Δt (float, optional): Same as ``stepsize``
            thresh_sil (float): Convergence threshold of short iterative Lanczos. Defaults to 1.e-09.
            step_size_is_fs (bool, optional): If ``True``, ``stepsize`` is in fs. Defaults to ``True``.


        Returns:
            Tuple[float, WFunc]: Energy during propagation (it conserves) and Wavefunction after propagation.

        """
        self.maxstep = maxstep
        if Δt is not None:
            self.stepsize = Δt
        else:
            self.stepsize = stepsize
        if not step_size_is_fs:
            self.stepsize *= units.au_in_fs
        self.backup_interval = backup_interval
        const.set_runtype(
            jobname=self.jobname + "_prop",
            restart=restart,
            relax=False,
            dvr=self.model.basinfo.is_DVR,
            savefile_ext=savefile_ext,
            loadfile_ext=loadfile_ext,
            maxstep=self.maxstep,
            use_jax=self.use_jax,
            standard_method=self.model.basinfo.is_standard_method,
            verbose=self.verbose,
            thresh_sil=thresh_sil,
            use_mpo=self.model.use_mpo,
            parallel_split_indices=parallel_split_indices,
            adaptive=adaptive,
            adaptive_Dmax=adaptive_Dmax,
            adaptive_dD=adaptive_dD,
            adaptive_p_proj=adaptive_p_proj,
            adaptive_p_svd=adaptive_p_svd,
            space=self.model.space,
            integrator=integrator,
            conserve_norm=conserve_norm,
        )

        return self._execute(
            autocorr,
            energy,
            norm,
            populations,
            observables,
            reduced_density,
            autocorr_per_step=autocorr_per_step,
            observables_per_step=observables_per_step,
            energy_per_step=energy_per_step,
            norm_per_step=norm_per_step,
            populations_per_step=populations_per_step,
        )

    def operate(
        self,
        maxstep: int = 10,
        restart: bool = False,
        savefile_ext: str = "_operate",
        loadfile_ext: str = "_gs",
        verbose: int = 2,
    ) -> tuple[float, WFunc]:
        """Apply operator such as dipole operator to the wavefunction

        Args:
            maxstep (int, optional): Maximum number of iteration. Defaults to ``10``.
            restart (bool, optional): Restart from the previous wavefunction. Defaults to ``False``.
            backend (str, optional): JAX or Numpy. Defaults to ``'jax'``.
            savefile_ext (str, optional): Extension of the save file. Defaults to ``'_operate'``.
            loadfile_ext (str, optional): Extension of the load file. Defaults to ``'_gs'``. \
                When ``restart=False``, ``loadfile_ext`` is ignored.
            verbose (int, optional): Verbosity level. Defaults to ``2``.

        Returns:
            Tuple[float, WFunc]: norm of O|Ψ> and Wavefunction after operation.

        """

        self.maxstep = maxstep
        const.set_runtype(
            apply_dipo=True,
            jobname=self.jobname + "_operate",
            restart=restart,
            dvr=self.model.basinfo.is_DVR,
            savefile_ext=savefile_ext,
            loadfile_ext=loadfile_ext,
            maxstep=self.maxstep,
            use_jax=self.use_jax,
            standard_method=self.model.basinfo.is_standard_method,
            verbose=verbose,
            use_mpo=self.model.use_mpo,
        )

        return self._execute(
            autocorr=False,
            energy=False,
            norm=True,
            populations=True,
            observables=False,
        )

    def _execute(
        self,
        autocorr=True,
        energy=True,
        norm=True,
        populations=True,
        observables=True,
        reduced_density=None,
        autocorr_per_step=1,
        observables_per_step=1,
        energy_per_step=1,
        norm_per_step=1,
        populations_per_step=1,
    ) -> tuple[Any, WFunc]:
        """Execute simulation

        Setup & run from the python prompt

        """

        time_fs = const.time_fs_init
        ints_prim = self.get_primitive_integrals()
        wf = self.get_initial_wavefunction(ints_prim)

        if const.doAppDipo:
            logger.info("Start: apply operator to wave function")
            norm = wf.apply_dipole(self.model.hamiltonian)
            self.save_wavefunction(wf, log=True)
            logger.info("End  : apply operator to wave function")
            return (norm, wf)

        self.save_wavefunction(wf, log=True)
        if const.mpi_size > 1:
            # Distribute MPO cores to all ranks
            assert isinstance(self.model.hamiltonian, TensorHamiltonian)
            self.model.hamiltonian.distribute_mpo_cores()
            for op in self.model.observables.values():
                assert isinstance(op, TensorHamiltonian)
                op.distribute_mpo_cores()
        if self.t2_trick:
            properties = Properties(
                wf,
                self.model,
                time=time_fs / units.au_in_fs,
                reduced_density=reduced_density,
            )
        else:
            assert time_fs == 0.0
            properties = Properties(
                wf,
                self.model,
                time=time_fs / units.au_in_fs,
                t2_trick=False,
                wf_init=deepcopy(wf),
                reduced_density=reduced_density,
            )

        logger.info(f"Start initial step {time_fs:8.3f} [fs]")
        stepsize_guess = (
            1.0e-3 / units.au_in_fs
        )  # a.u. [typical values in MCTDH]
        if const.mpi_rank == 0:
            iterator = tqdm(range(self.maxstep))
        else:
            iterator = range(self.maxstep)
        for istep in iterator:
            time_fs = properties.time * units.au_in_fs
            if istep % 100 == 1:
                niter_krylov_list = list(helper._Debug.niter_krylov.values())
                niter_krylov_total = sum(niter_krylov_list)
                ncall_krylov_total = len(niter_krylov_list)
                message = (
                    f"End {istep - 1:5d} step; "
                    + f"propagated {time_fs:8.3f} [fs]; "
                    + f"AVG Krylov iteration: {niter_krylov_total / ncall_krylov_total:.2f}"
                )
                logger.info(message)
            if istep % self.backup_interval == self.backup_interval - 1:
                # Save wave function data can be a bottleneck, so we save it every 100 steps.
                logger.info(f"Saved wavefunction {time_fs:8.3f} [fs]")
                self.save_wavefunction(wf)
            properties.get_properties(
                autocorr=autocorr,
                energy=energy,
                norm=norm,
                populations=populations,
                observables=observables,
                autocorr_per_step=autocorr_per_step,
                energy_per_step=energy_per_step,
                norm_per_step=norm_per_step,
                populations_per_step=populations_per_step,
                observables_per_step=observables_per_step,
            )
            properties.export_properties(
                autocorr_per_step=autocorr_per_step,
                populations_per_step=populations_per_step,
                observables_per_step=observables_per_step,
            )

            helper._ElpTime.steps -= time()
            if const.standard_method:
                if self.model.one_gate_to_apply is not None:
                    wf.apply_one_gate(self.model.one_gate_to_apply)
                stepsize_actual = self.stepsize / units.au_in_fs
                _ = wf.propagate_SM(
                    self.model.hamiltonian, stepsize_actual, istep
                )
            else:
                if const.doDVR:
                    raise NotImplementedError
                g, spf_occ, stepsize_actual, stepsize_guess = wf.propagate_CMF(
                    self.model.hamiltonian, stepsize_guess
                )
            helper._ElpTime.steps += time()
            properties.update(stepsize_actual)
        if self.maxstep > 0:
            niter_krylov_list = list(helper._Debug.niter_krylov.values())
            niter_krylov_total = sum(niter_krylov_list)
            ncall_krylov_total = len(niter_krylov_list)
            message = (
                f"End {self.maxstep - 1:5d} step; "
                + f"propagated {time_fs:8.3f} [fs]; "
                + f"AVG Krylov iteration: {niter_krylov_total / ncall_krylov_total:.2f}"
            )
            logger.info(message)
        logger.info("End simulation and save wavefunction")
        self.save_wavefunction(wf, log=True)
        return (properties.energy, wf)

    def get_primitive_integrals(self) -> PrimInts:
        if const.doDVR:
            logger.debug("Set integral of DVR basis")
        else:
            logger.debug("Set integral of FBR basis")
        _debug = -time()
        if self.model.ints_prim_file is None:
            ints_prim = PrimInts(self.model)
        else:
            filename = self.model.ints_prim_file
            if os.path.exists(filename):
                with open(filename, "rb") as load_f:
                    ints_prim = pickle.load(load_f)
                    if const.verbose > 1:
                        logger.info("file loaded: ints_prim")
            else:
                ints_prim = PrimInts(self.model)
                with open(filename, "wb") as save_f:
                    pickle.dump(ints_prim, save_f)
                    if const.verbose > 1:
                        logger.info("file saved: ints_prim")
        _debug += time()
        if const.verbose > 1:
            logger.debug(f"Time for PrimInts initialization: (sec.) {_debug}")
        return ints_prim

    def get_initial_wavefunction(self, ints_prim: PrimInts) -> WFunc:
        if const.doDVR:
            logger.debug("Set initial wave function (DVR basis)")
        else:
            logger.debug("Set initial wave function (FBR basis)")
        """setup initial w.f."""
        if const.doRestart:
            path = f"wf_{self.jobname}{const.loadfile_ext}.pkl"
            with open(path, "rb") as load_f:
                wf = dill.load(load_f)
                wf = WFunc(wf.ci_coef, wf.spf_coef, ints_prim)
                # Restart from wf.ints_prim has some problem because of the difference of the 'onesite' keys
            logger.info(f"Wave function is loaded from {path}")
        else:
            if self.ci_type.lower() == "mps":
                if const.verbose > 1:
                    logger.debug("Prepare MPS w.f.")
                if self.do_init_proj_gs:
                    logger.debug("Initial SPF: projected from GS")
                    if const.use_mpo:
                        raise NotImplementedError
                    else:
                        wf = WFunc(
                            MPSCoefSoP.alloc_random(self.model),
                            SPFCoef.alloc_proj_gs(self.model),
                            ints_prim,
                        )
                else:
                    logger.debug("Initial SPF: uniform (all 1.0)")
                    spf_coef = SPFCoef.alloc_eye(self.model)
                    if const.use_mpo:
                        if const.mpi_size > 1:
                            _mps_coef_cls: type[MPSCoef] = MPSCoefParallel
                        else:
                            _mps_coef_cls = MPSCoefMPO
                    else:
                        if const.mpi_size > 1:
                            raise NotImplementedError
                        else:
                            _mps_coef_cls = MPSCoefSoP
                    wf = WFunc(
                        _mps_coef_cls.alloc_random(self.model),
                        spf_coef,
                        ints_prim,
                    )
            elif self.ci_type.lower() in [
                "mctdh",
                "ci",
                "standard-method",
                "sm",
            ]:
                if const.doDVR:
                    raise NotImplementedError

                if const.verbose > 1:
                    logger.debug("Prepare MCTDH w.f.")
                if self.do_init_proj_gs:
                    logger.debug("Initial SPF: projected from GS")
                    wf = WFunc(
                        helper.trans_mps2fci(
                            MPSCoefSoP.alloc_random(self.model),
                            self.model.basinfo,
                        ),
                        SPFCoef.alloc_proj_gs(self.model),
                        ints_prim,
                    )
                else:
                    logger.debug("Initial SPF: uniform (all 1.0)")
                    wf = WFunc(
                        helper.trans_mps2fci(
                            MPSCoefSoP.alloc_random(self.model),
                            self.model.basinfo,
                        ),
                        SPFCoef.alloc_eye(self.model),
                        ints_prim,
                    )
            else:
                raise ValueError(
                    f"ci_type must be 'mps' or 'mctdh', but {self.ci_type} is given."
                )
        return wf

    def save_wavefunction(self, wf: WFunc, log: bool = False):
        if const.mpi_size > 1:
            assert isinstance(wf.ci_coef, MPSCoefParallel)
            ci_coef = wf.ci_coef.to_MPSCoefMPO()
            if const.mpi_rank == 0:
                assert isinstance(ci_coef, MPSCoefMPO)
                wf = WFunc(ci_coef, wf.spf_coef, wf.ints_prim)
        if const.mpi_rank == 0:
            path = f"wf_{self.jobname}{const.savefile_ext}.pkl"
            with open(path, "wb") as save_f:
                dill.dump(wf, save_f)
            if log:
                logger.info(f"Wave function is saved in {path}")
