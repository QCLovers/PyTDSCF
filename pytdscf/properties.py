"""Property handling module"""

import os
from logging import getLogger

import netCDF4 as nc
import numpy as np

import pytdscf._helper as helper
from pytdscf import units
from pytdscf._const_cls import const
from pytdscf._mps_cls import MPSCoef

logger = getLogger("main").getChild(__name__)
auto_logger = getLogger("autocorr")
pop_logger = getLogger("populations")
exp_logger = getLogger("expectations")


class Properties:
    """Structure class of calculated property

    Attributes:
        wf (WFunc): wave function
        time (float): time in atomic units
        t2_trick (bool): whether to use so-called T/2 trick
        autocorr (complex): auto-correlation function
        energy (float): energy
        pops (List[float]): populations of each state
        expectations (List[float]): observables of given operator
    """

    def __init__(
        self,
        wf,
        model,
        time=0.0,
        t2_trick=True,
        wf_init=None,
        reduced_density=None,
    ):
        self.wf = wf
        self.model = model
        self.time = time
        self.nstep = 0
        self.nc_row = 0
        self.t2_trick = t2_trick

        self.autocorr = None
        self.energy = None
        self.norm = None
        self.pops = None
        self.expectations = {}
        self.wf_zero = wf_init
        self.nc_file: str | None = None
        assert t2_trick or (wf_init is None)

        if reduced_density is not None:
            self.rd_step = reduced_density[1]
            self.remain_legs: list[tuple[int, ...]] | None = []
            self.rd_keys = []
            for key in reduced_density[0]:
                rd_points = sorted(key, reverse=True)
                _remain_legs = [0 for isite in range(rd_points[0] + 1)]
                isite = 0
                while rd_points:
                    if isite == rd_points[-1]:
                        _remain_legs[isite] += 1
                        rd_points.pop()
                    else:
                        isite += 1
                assert all([0 <= leg <= 2 for leg in _remain_legs])
                self.remain_legs.append(tuple(_remain_legs))
                self.rd_keys.append(key)
            if self.nc_file is None:
                self.nc_file = self._create_nc_file(reduced_density)
        else:
            self.rd_step = None
            self.remain_legs = None

    def get_properties(
        self,
        autocorr=True,
        energy=True,
        norm=True,
        populations=True,
        observables=True,
        autocorr_per_step=1,
        energy_per_step=1,
        norm_per_step=1,
        populations_per_step=1,
        observables_per_step=1,
    ):
        if autocorr and self.nstep % autocorr_per_step == 0:
            self._get_autocorr()
        if energy and self.nstep % energy_per_step == 0:
            self._get_energy()
        if norm and self.nstep % norm_per_step == 0:
            self._get_norm()
        if populations and self.nstep % populations_per_step == 0:
            self._get_pops()
        if observables and self.nstep % observables_per_step == 0:
            self._get_observables()
        if self.remain_legs is not None:
            if self.nstep % self.rd_step == 0:
                self._export_reduced_density()

    def _export_reduced_density(self):
        assert isinstance(self.nc_file, str)
        assert isinstance(self.remain_legs, list)
        complex128 = np.dtype([("real", np.float64), ("imag", np.float64)])
        with nc.Dataset(self.nc_file, "a") as f:
            # Maybe we should keep files open while the simulation is running.
            f.variables["time"][self.nc_row] = self.time * units.au_in_fs
            for remain_leg, key in zip(
                self.remain_legs, self.rd_keys, strict=True
            ):
                densities = self.wf.get_reduced_densities(remain_leg)
                for istate in range(self.model.nstate):
                    data = np.empty(densities[istate].shape, complex128)
                    data["real"] = densities[istate].real
                    data["imag"] = densities[istate].imag
                    f.variables[f"rho_{key}_{istate}"][self.nc_row] = data
        self.nc_row += 1

    def _create_nc_file(
        self, reduced_density: tuple[list[tuple[int, ...]], int]
    ) -> str:
        jobname = const.jobname
        path_to_nc = f"{jobname}/reduced_density.nc"
        if os.path.exists(path_to_nc):
            os.remove(path_to_nc)

        with nc.Dataset(
            f"{jobname}/reduced_density.nc", "w", format="NETCDF4"
        ) as f:
            f.createDimension("step", None)
            f.createDimension("state", self.model.nstate)
            complex128 = np.dtype([("real", np.float64), ("imag", np.float64)])
            complex128_t = f.createCompoundType(complex128, "complex128")
            modes = set()
            for key in reduced_density[0]:
                # key must be ascending order.
                assert key == tuple(sorted(key)), (
                    f"Reduced density key {key} must be ascending order"
                )
                modes = modes.union(set(key))
                for idof in key:
                    if f"Q{idof}" not in f.dimensions:
                        f.createDimension(
                            f"Q{idof}", self.model.basinfo.get_ngrid(0, idof)
                        )
            f.createVariable("time", "f8", ("step",))
            for key in reduced_density[0]:
                if len(key) > 3:
                    logger.warning(
                        f"key {key} is too large to be saved in netCDF4"
                    )
                for istate in range(self.model.nstate):
                    dimensions = ("step",) + tuple(f"Q{idof}" for idof in key)
                    f.createVariable(
                        f"rho_{key}_{istate}", complex128_t, dimensions
                    )
        return path_to_nc

    def _get_autocorr(self):
        if self.t2_trick:
            if const.standard_method and isinstance(self.wf.ci_coef, MPSCoef):
                self.autocorr = self.wf._ints_wf_ovlp_mpssm(
                    self.wf.ci_coef, conj=False
                )
            else:
                self.autocorr = self.wf.autocorr()
        else:
            if const.doDVR:
                if const.standard_method:
                    self.autocorr = self.wf_zero._ints_wf_ovlp_mpo(
                        self.wf.ci_coef
                    )
                else:
                    raise NotImplementedError(
                        "DVR-MCTDH with MPO is not implemented"
                    )
            else:
                self.autocorr = self.wf_zero._ints_wf_ovlp_sop(
                    self.wf.ci_coef, self.wf.spf_coef, self.model.hamiltonian
                )

    def _get_energy(self):
        self.energy = self.wf.expectation(self.model.hamiltonian)

    def _get_norm(self):
        self.norm = self.wf.norm()

    def _get_pops(self):
        self.pops = self.wf.pop_states()

    def _get_observables(self):
        for obs_key, matOp in self.model.observables.items():
            # if type(matOp) is not PolynomialHamiltonian:
            #     raise NotImplementedError(
            #         f'matOp {type(matOp)} is not implemented in Properties._get_observables')
            self.expectations[obs_key] = self.wf.expectation(matOp)

    def export_properties(
        self,
        autocorr_per_step=1,
        populations_per_step=1,
        observables_per_step=1,
    ):
        if self.nstep % autocorr_per_step == 0:
            self._export_autocorr()
        if self.nstep % populations_per_step == 0:
            self._export_populations()
        if self.nstep % observables_per_step == 0:
            self._export_expectations()
        self._export_properties()

    def _export_autocorr(self):
        if self.autocorr is None:
            return
        if self.time == 0.0:
            auto_logger.debug("# time [fs]\t auto-correlation")
        if self.t2_trick:
            time_fs = self.time * units.au_in_fs * 2
        else:
            time_fs = self.time * units.au_in_fs
        auto_logger.debug(
            f"{time_fs:6.9f}\t"
            + f"{self.autocorr.real: 6.9f}{self.autocorr.imag:+6.9f}j"
        )

    def _export_populations(self):
        if self.pops is None:
            return
        if self.time == 0.0:
            pop_logger.debug(
                "# time [fs]\t"
                + "\t".join(
                    [
                        f"pop_{i}" + " " * (11 - len(f"pop_{i}"))
                        for i in range(len(self.pops))
                    ]
                )
            )
        pop_msg = f"{self.time * units.au_in_fs:6.9f}\t"
        for pop in self.pops:
            pop_msg += f"{pop:6.9f}\t"
        pop_msg.rstrip("\t")
        pop_logger.debug(pop_msg)

    def _export_expectations(self):
        if self.expectations == {}:
            return
        if self.time == 0.0:
            exp_logger.debug(
                "# time [fs]\t"
                + "\t".join(
                    [
                        f"{obs_key}" + " " * (11 - len(f"{obs_key}"))
                        for obs_key in self.expectations.keys()
                    ]
                )
            )
        exp_msg = f"{self.time * units.au_in_fs:6.9f}\t"
        for exp in self.expectations.values():
            exp_msg += f"{exp:6.9f}\t"
        exp_msg.rstrip("\t")
        exp_logger.debug(exp_msg)

    def _export_properties(self):
        time_fs = self.time * units.au_in_fs
        norm = self.norm
        pop_states = self.pops
        energy = self.energy
        autocorr = self.autocorr

        if norm is not None:
            if abs(norm - 1.0) > 1.0e-06:
                logger.warning(
                    f"Wave Function norm is not 1.0, but {norm} when {time_fs} fs"
                )
        message = ""
        if const.verbose > 1 and autocorr is not None:
            message += (
                f"| autocorr: {autocorr.real: 6.4f}{autocorr.imag:+6.4f}i"
            )

        if const.verbose > 1:
            if pop_states is not None:
                message += "| pop" + (" {:6.4f} " * len(pop_states[:3])).format(
                    *pop_states[:3]
                )
            if energy is not None:
                message += f"| ene[eV]: {energy * units.au_in_eV:10.7f} "
            message += (
                f"| time[fs]: {time_fs:8.3f} "
                + f"| elapsed[sec]:{helper._ElpTime.steps:9.2f} "
            )
        if const.verbose == 4:
            mflops = (
                (
                    helper._NFlops.ci_expo
                    + helper._NFlops.ci_renm
                    + helper._NFlops.ci_mfop
                )
                / pow(10, 6)
                / max(0.01, helper._ElpTime.zgemm)
            )
            if not const.standard_method:
                message += (
                    f"| spf:{helper._ElpTime.spf:5.1f} "
                    + f"| mfop:{helper._ElpTime.mfop:5.1f} "
                )
            message += (
                f"| ci:{helper._ElpTime.ci:5.1f} "
                + f" (ci_exp:{helper._ElpTime.ci_exp:5.1f}"
                + f"|ci_rnm:{helper._ElpTime.ci_rnm:5.1f}"
                + f"|ci_etc:{helper._ElpTime.ci_etc:5.1f} d) "
                + f"|{mflops:5.0f} MFLOPS "
                + f"({helper._ElpTime.zgemm:5.1f} s) "
            )
        logger.debug(message)

    def update(self, delta_t):
        """update time"""
        self.time += delta_t
        self.nstep += 1
        if const.doTDHamil:
            if const.doDVR:
                raise NotImplementedError
            self.model.hamiltonian = self.model.build_td_hamiltonian(
                time_fs=self.time * units.au_in_fs
            )
