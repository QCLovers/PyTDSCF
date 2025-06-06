"""
Shared constant parameters are defined here.
"""

import datetime
import os
from logging import (
    DEBUG,
    INFO,
    FileHandler,
    Formatter,
    StreamHandler,
    getLogger,
)

from pytdscf import units

CBOLD = "\33[1m"
CVIOLET = "\33[35m"

pytdscf = (
    "\n"
    r"     ____     __________   .____ ____   _____" + "\n"
    r"    / _  |   /__  __/ _ \ / ___ / _  \ / ___/" + "\n"
    r"   / /_) /_  __/ / / / ||/ /__ / / )_// /__" + "\n"
    r"  /  ___/ / / / / / / / |.__  / |  __/ ___/" + "\n"
    r" /  /  / /_/ / / / /_/ /___/ /| \_/ / /" + "\n"
    r"/__/   \__, /_/ /_____/_____/ \____/_/" + "\n"
    r"      /____/" + "\n"
)
CEND = "\033[0m"


class Const:
    """

    This class holds the common calculation parameters \
        required for other modules.

    Attributes:
        doRestart (bool) : Whether restart propagation or not. \
            Default is ``False``.
        doRelax (bool) : Whether imaginary propagation or not. \
            Default is ``False``. \
            Especially if you set this attribute as \
            ``'improved'`` in MPS-MCTDH(SM) level, \
            improved relaxation (by sweep algorithm) will be executed.
        doAppDipo (bool) : Whether operate some operator \
            (e.g. dipole moment operator) to a certain WF \
                (e.g. ground state) or not. If ``True``, propagation \
                and relaxation is not executed.
        doDVR (bool) : If ``True``, \
            Discrete Variable Representation (DVR) integral will be executed.
        savefile_ext (str) : The suffix of file name to save the \
            WF, autocorr, expectation, etc. \
            Default is ``''``.\
            If you set this attribute as ``'foo'``, \
            the calculation results are saved as \
            ``kind_of_results  + jobname + time + 'foo' + '.pickle'``.
        loadfile_ext (str) : The suffix of file name of restart saved file. \
            Defaults to ``''``.
        time_fs_init (float) : The time of restart. Default is ``0.0``.
        maxstep (int) : \
            The number of propagation or relaxation or operation step.
        doOrtho (bool) : Default is ``False``. N.Y.I
        doTDHamil (bool) : \
            Default is ``False``. If you set time-dependent hamiltonian H(t), \
            this bool must be ``True``.
        keys (set) : Defaults to ``'{enable_summed_op}'``. \
            This means that the number of \
            nMR operator is reduced by complementary operator for MPS.
            If you `add` ``'{enable_tdh_dofs}'``, one-basis-site calculation
            was skipped in MPS-sweep.(This option may have bug.)
        mass (float) : \
            The mass of electron. In default atomic unit, this is ``1.0``.\
            Note that some other programs adopt AMU.
        epsrho (float) : \
            The threshold of shift small eigenvalue for regularization. \
            See also `helper.matrix_regularized_inverse` and \
            `mps_cls.SiteCoef.gauge_trf` source code. Default is ``1.0e-08``.
            See also reference :
                - https://doi.org/10.1063/1.5024859
                - https://doi.org/10.1063/1.5042776
        thresh_exp (float) : \
            The threshold of Short Iterative Lanczos (SIL) convergence.\
            Defaults to ``1.0e-08``.
        tol_CMF (float) : \
            The threshold of Constant Mean Field (CMF) propagation method \
            tolerance. Defaults to ``1.0e-14``. (Maybe too small.)
        max_stepsize (float) : \
            The maximum time step width [a.u.] without error. \
            Defaults to 0.010 [fs] (Maybe too small.)
        tol_RK45 (float) : \
            The error tolerance of SPF RungeKutta. Defaults to ``1.0e-08``.

    """

    def __setattr__(self, name, value):
        if name not in ["verbose", "jobname"] and name in self.__dict__:
            if self.__dict__[name] != value:
                self.logger.warning(f"rebind const {name}")
        self.__dict__[name] = value

    def set_runtype(
        self,
        *,
        jobname: str | None = None,
        restart: bool = False,
        relax: bool | str = False,
        apply_dipo: bool = False,
        dvr: bool = False,
        savefile_ext: str = "",
        loadfile_ext: str = "",
        time_fs_init: float = 0.0,
        maxstep: int = 9999999,
        use_jax: bool = False,
        standard_method: bool = True,
        thresh_sil: float = 1.0e-09,
        verbose: int = 2,
        use_mpo: bool = True,
    ):
        """

        Set the `Const` attributes of your calculation. \
            See also `Const` attributes.

        Args:
            jobname (optional, str) : jobname. \
                Defaults to `relax.log`, `propagate.log` or `operate.log`.
            restart (bool) : Defaults to ``False``.
            relax (bool) : Defaults to ``False``.
            apply_dipo (bool) : Defaults to ``False``. Same as const.doAppDipo.
            dvr (bool) : Defaults to ``False``. Same as const.doDVR.
            savefile_ext (str) : Defaults to ``''``.
            loadfile_ext (str) : Defaults to ``''``.
            time_fs_init (float) : Defaults to ``0.0``.
            maxstep (int) : Defaults to ``9999999``.
            use_jax (bool) : Defaults to ``True``.
                In large MPS size (m~30, nspf~5), GPU is much faster, \
                but in small size (m~10, nspf~3), numpy is faster. \
                If True, mpirun will be ignored.
            standard_method (bool) : nspf==nprim
            thresh_sil (float) : convergence threshold of short iterative Lanczos.
                Defaults to 1.e-09.
            verbose (int) : Defaults to 4. 4=noisy for debug and development, \
                3=normal, 2=least calculation output, 1=only logging and warnig
            use_mpo (bool) : Defaults to ``True``.

        """
        if jobname is None:
            if apply_dipo:
                self.jobname = "operate"
            elif relax:
                self.jobname = "relax"
            else:
                self.jobname = "propagate"
        else:
            self.jobname = jobname
        set_main_logger(overwrite=True)
        set_logger("autocorr")
        set_logger("populations")
        set_logger("expectations")
        self.logger = getLogger("main").getChild(__name__)
        if verbose >= 3:
            self.logger.info(CBOLD + CVIOLET + pytdscf + CEND)
        if verbose >= 1:
            self.logger.info(f"START TIME: {datetime.datetime.now()}")
            self.logger.info(f"Log file is ./{self.jobname}/main.log")

        self.verbose = verbose
        self.doRestart = restart
        self.doRelax = relax
        self.doAppDipo = apply_dipo
        self.doDVR = dvr
        self.savefile_ext = savefile_ext
        self.loadfile_ext = loadfile_ext
        self.time_fs_init = time_fs_init
        self.maxstep = maxstep

        self.doOrtho = False
        self.doTDHamil = False
        self.oldcode = False

        self.use_jax = use_jax
        const.thresh_exp = thresh_sil
        self.standard_method = standard_method
        self.use_mpo = use_mpo
        if self.use_mpo:
            assert self.standard_method, (
                "MPO is only available for standard method."
            )


const = Const()
const.keys = {"enable_summed_op"}
const.verbose = 4
const.mass = 1.0  # [m_e]

const.epsrho = 1.0e-8  # default
const.tol_CMF = 1.0e-14
const.max_stepsize = 0.010 / units.au_in_fs  # [au]
const.tol_RK45 = 1.0e-8  # default
const.mpi_rank = 0
const.mpi_size = 1


def set_main_logger(overwrite: bool = True):
    """Set logger"""
    logger = getLogger("main")
    logger.setLevel(DEBUG)
    if hasattr(const, "jobname"):
        if not os.path.exists(f"./{const.jobname}"):
            os.makedirs(f"./{const.jobname}")
        filename = f"{const.jobname}/main.log"
    else:
        filename = "pytdscf.log"
    if overwrite:
        # First definition allows overwrite logfile
        file_handler = FileHandler(filename, "w")
    else:
        file_handler = FileHandler(filename)
    file_handler.setLevel(DEBUG)
    formatter = Formatter("%(asctime)s - %(levelname)s:%(name)s - %(message)s")
    stream_handler = StreamHandler()
    stream_handler.setLevel(INFO)
    # change INFO to DEBUG if you want all messages to console.
    stream_handler.setFormatter(formatter)

    while logger.hasHandlers():
        if len(logger.handlers) > 0:
            logger.removeHandler(logger.handlers[0])
        else:
            break

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def set_logger(name: str):
    """Set logging file"""
    logger = getLogger(name)
    logger.setLevel(DEBUG)
    if not os.path.exists(f"./{const.jobname}"):
        os.makedirs(f"./{const.jobname}")
    filename = f"{const.jobname}/{name}.dat"
    file_handler = FileHandler(filename, "w")
    file_handler.setLevel(DEBUG)

    while logger.hasHandlers():
        if len(logger.handlers) > 0:
            logger.removeHandler(logger.handlers[0])
        else:
            break

    logger.addHandler(file_handler)
