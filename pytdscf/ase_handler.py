"""
Multiprocessing mesh electronic structure calculation
caller using ASE library
"""

import itertools
import os
import pickle
import shutil
import time
import types
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Callable

import numpy as np

try:
    from ase.atoms import Atoms
    from ase.calculators.calculator import Calculator
    from ase.calculators.genericfileio import GenericFileIOCalculator
    from ase.db import connect
except Exception as e:
    raise ModuleNotFoundError("Please install ase package.") from e
from loguru import logger

from pytdscf import units
from pytdscf._helper import from_dbkey, progressbar
from pytdscf.basis.abc import DVRPrimitivesMixin

logger = logger.bind(name="main")


def _run(atoms: Atoms) -> tuple[Atoms]:
    """Run for multiprocessing
    (do not include in DVR_Mesh class because it may become slow.)
    """
    try:
        atoms.get_total_energy()  # 1 / Hartree
        atoms.get_forces()  # 1 / Hartree * Bohr
        atoms.get_dipole_moment()  # 1 / Bohr
    except Exception as e:
        logger.warning(f"ERROR: {e}")
    return (atoms,)


def _todict(self):
    """
    If calculator has not attribute 'todict',
    add this method to enable db.update
    """
    original_dict = vars(self)
    return original_dict["parameters"] | original_dict["results"]


class DVR_Mesh:
    """ DVR grid coordinate

    Args:
        dvr_prims (List[DVRPrimitivesMixin]) : DVR primitive list
        atoms (List[List[str, Tuple[float,float,float]]] or ase.atoms.Atoms) : \
                reference coordinate. Format is the same as PySCF ones
        disp_vec (np.ndarray) : displacement vectors (row vectors) in angstrom
        unit (Optional[bool]) : \
            Input reference coordinate unit. Defaults to 'angstrom'

    """

    displace: dict
    geometry: dict
    grid_id: dict
    jobname: str
    remain_jobs: deque
    reset_calc: bool
    done_jobs: deque
    thrown_jobs: deque

    def __init__(
        self,
        dvr_prims: list[DVRPrimitivesMixin],
        atoms: Atoms,
        disp_vec: np.ndarray,
        unit: str = "angstrom",
    ):
        self.grid_list = [g.get_grids() for g in dvr_prims]
        self.dvr_prims = dvr_prims
        self.ndof = len(dvr_prims)
        self.disp_vec = disp_vec
        self.zero_indices = [None for _ in range(self.ndof)]

        if len(dvr_prims) != len(disp_vec):
            raise TypeError
        if disp_vec.shape[-1] != 3:
            raise TypeError

        for i, prim in enumerate(dvr_prims):
            grids = prim.get_grids()
            for j, grid in enumerate(grids):
                if abs(grid) < 1.0e-10:
                    self.zero_indices[i] = j
                    break

        if type(atoms) is Atoms:
            unit = "angstrom"
            self.symbols = atoms.symbols
            self.positions = atoms.positions
            self.masses = atoms.get_masses()
        else:
            if unit.lower() == "angstrom":
                self.positions = np.array([position for _, position in atoms])
            elif unit.lower() in ["bohr", "au", "a.u."]:
                self.positions = (
                    np.array([position for _, position in atoms])
                    * units.au_in_angstrom
                )
            else:
                raise NotImplementedError
            self.symbols = [element for element, _ in atoms]
        if disp_vec.shape[1] != len(self.positions):
            raise TypeError

    def save_geoms(
        self,
        jobname: str,
        nMR: int | None = None,
        overwrite: bool | None = None,
    ) -> dict[str, dict[str, int]]:
        """
        Generate cartesian coordinate geometry for each grid mesh.

        Args:
            nMR (Optional[int]) : Tne number of mode representation. \
                limits n dimensional mesh. \
                Defaults to ``None``, thus, \
                ``ngrid**ndof`` coords will be generated.
            overwrite (Optional[bool]) : overwrite detabase

        Returns:
            Dict[str, Dict[str, int]] : DVR Mesh coordinates. \
                E.g. [(0,1)][(2,3)] gives 2nd, 3rd grids of 0-mode, \
                    1-mode coordinate.

        """
        self.jobname = jobname
        if nMR is None:
            nMR = self.ndof
        logger.warning("START : Displacement Generation")
        for iMR in range(1, nMR + 1):
            if iMR == 1:
                for idof in range(self.ndof):
                    self.displace[(idof,)] = {}
                    for igrid, coef in enumerate(self.grid_list[idof]):
                        self.displace[(idof,)][(igrid,)] = (
                            coef * self.disp_vec[idof]
                        )
            else:
                for dof_key in itertools.combinations(range(self.ndof), r=iMR):
                    self.displace[dof_key] = {}
                    add_dof = (dof_key[-1],)
                    orig_dof = dof_key[:-1]
                    for orig_grid_key, orig_disp_vec in self.displace[
                        orig_dof
                    ].items():
                        for add_grid_key, add_disp_vec in self.displace[
                            add_dof
                        ].items():
                            grid_key = orig_grid_key + add_grid_key
                            disp_vec = orig_disp_vec + add_disp_vec
                            self.displace[dof_key][grid_key] = disp_vec

        for dof_key, grid_dict in self.displace.items():
            if len(dof_key) < self.ndof == nMR:
                continue
            self.geometry[dof_key] = {}
            for grid_key, disp_vec in grid_dict.items():
                self.geometry[dof_key][grid_key] = self.positions + disp_vec
        logger.warning("DONE : Displacement Generation")

        if os.path.exists(f"{self.jobname}.db"):
            if overwrite is None:
                yes_or_else = input(
                    f"{self.jobname}.db is already exists!"
                    + " Dou you remove the database ?[y/n]"
                    + "(Defaults to 'y')"
                )
            else:
                yes_or_else = "y" if overwrite else "n"
            if yes_or_else.lower() in ["y", "yes", ""]:
                os.remove(f"{self.jobname}.db")
                overwrite = True
            else:
                overwrite = False
        self.grid_id = {}

        with connect(f"{self.jobname}.db") as db:
            _iter = 0
            for dof_key, grid_dict in self.geometry.items():
                self.grid_id[dof_key] = {}
                for grid_key, coord in progressbar(
                    grid_dict.items(),
                    prefix=f"Save geometry in DB of DOFs = {dof_key}",
                ):
                    if overwrite:
                        atoms_grid = Atoms(self.symbols, positions=coord)
                        grid = deepcopy(self.zero_indices)
                        for p, g in zip(dof_key, grid_key, strict=True):
                            grid[p] = g
                        _id = db.write(
                            atoms_grid,
                            dofs="|" + " ".join(map(str, dof_key)),
                            grids="|" + " ".join(map(str, grid)),
                        )
                        self.grid_id[dof_key][grid_key] = _id
                    else:
                        _iter += 1
                        self.grid_id[dof_key][grid_key] = _iter
        with open(f"{self.jobname}_grid_id.pkl", "wb") as f:
            pickle.dump(self.grid_id, f)

            """ nMR coord can get by (n-1)MR coord.
            May be too memory consumption"""

        # Free Memory
        del self.geometry
        del self.displace
        return self.grid_id

    def execute_multiproc(
        self,
        calculator: Calculator,
        max_workers: int | None = None,
        timeout: float = 60.0,
        jobname: str | None = None,
        reset_calc: bool = False,
        judge_func: Callable | None = None,
    ):
        """Execute electronic structure calculation by multiprocessing

        Args:
            calculator (Calculator) : calculator for each geomtry
            max_workers (Optional[int]) : maximum workers in multi-processing.
                     Defaults to None. If None, use cpu_count - 1.
            timeout (float) : Timeout calculation in second. Defaults to 60.0
            jobname (Optional[str]) : jobname
            reset_calc (Optional[bool]) : set new calculator in any case.
                    Defaults to False.
            judge_func (Optional[Callable[[Any],bool]]) : judge function whether re-calculation is needed.
                    Defaults to None.

        """  # noqa: E501
        self.calc = calculator
        if isinstance(jobname, str):
            self.jobname = jobname
        if self.jobname is None:
            raise ValueError("required jobname argument.")

        self.remain_jobs = deque()
        self.thrown_jobs = deque()
        self.done_jobs = deque()

        # In case of duplicated jobs, we should remove duplicated jobs.
        if judge_func is None:
            self.judge_func = lambda row: True
        else:
            self.judge_func = judge_func

        with connect(f"{self.jobname}.db") as db:
            unique_jobs = dict()
            count_unique_jobs = 0
            for row in db.select():
                if self.judge_func(row):
                    dof_key = tuple(from_dbkey(row.dofs))
                    grid_key_tmp = from_dbkey(row.grids)
                    grid_key = tuple([grid_key_tmp[p] for p in dof_key])
                    if row.grids not in unique_jobs:
                        unique_jobs[row.grids] = row.id
                        count_unique_jobs += 1
                    self.remain_jobs.append(
                        (
                            dof_key,
                            grid_key,
                            row.id,
                            unique_jobs[row.grids],
                            None,
                        )
                    )
            logger.warning(f"unique jobs : {count_unique_jobs}")
        self.reset_calc = reset_calc

        if max_workers is None:
            ncpu = os.cpu_count()
            assert isinstance(ncpu, int)
            max_workers = ncpu - 1
        wait_process = max_workers

        logger.warning("START : Electronic Structure Calculations")
        n = len(self.remain_jobs)
        if n > 0:
            with ProcessPoolExecutor(max_workers) as exe:
                with connect(f"{self.jobname}.db") as db:
                    logger.warning(f"Connected: {self.jobname}.db")
                    for _ in range(min(wait_process, len(self.remain_jobs))):
                        self._throw_job_to_queue(exe, db)

                    for _iter in progressbar(range(n)):
                        self._pick_up_job_from_queue(db, timeout)
                        if self.remain_jobs:
                            self._throw_job_to_queue(exe, db)
                    logger.warning("WAIT  : Remaining future task")
                    while self.thrown_jobs:
                        self._pick_up_job_from_queue(db, timeout)

                logger.warning("DONE  : Electronic Structure Calculations")
                if len(self.done_jobs) == n:
                    logger.warning("Your calculation completely finished!")
                else:
                    logger.warning(
                        f"Remained {n - len(self.done_jobs)} jobs!"
                        + " you should execute once again "
                        + "with different conditions and judge_func"
                    )
                for process in exe._processes.values():
                    process.kill()
        logger.warning("DONE  : Shutdown process executor")

    def _throw_job_to_queue(
        self,
        exe,
        db,
    ):
        while self.remain_jobs:
            dof_key, grid_key, _id, unique_id, error = (
                self.remain_jobs.popleft()
            )
            if _id == unique_id:
                row_unique = db.get(unique_id)
                if self.judge_func(row_unique) and hasattr(
                    row_unique, "energy"
                ):
                    atoms = row_unique.toatoms()
                    if not isinstance(atoms.calc, Calculator):
                        atoms.calc.todict = types.MethodType(
                            _todict, atoms.calc
                        )
                    db.update(_id, atoms)
                    self.done_jobs.append((dof_key, grid_key, _id))
                    continue

            atoms = db.get_atoms(_id)
            if atoms.calc is None or self.reset_calc:
                atoms.calc = deepcopy(self.calc)
            if isinstance(atoms.calc, Calculator):
                # Gaussian etc
                atoms.calc.set_label(f"{self.jobname}/{_id:07}/calc")
            elif isinstance(atoms.calc, GenericFileIOCalculator):
                # Orca etc
                atoms.calc.directory = Path(f"{self.jobname}/{_id:07}/calc")
            else:
                raise NotImplementedError
            future = exe.submit(_run, atoms)
            self.thrown_jobs.append(
                (future, dof_key, grid_key, _id, time.time())
            )
            break

    def _pick_up_job_from_queue(self, db, timeout: float = 60.0):
        while True:
            if self.thrown_jobs:
                future, dof_key, grid_key, _id, start_time = (
                    self.thrown_jobs.popleft()
                )
            else:
                break
            if future.done():
                if (error := future.exception()) is None:
                    atoms = future.result()[0]
                    if not isinstance(atoms.calc, Calculator):
                        atoms.calc.todict = types.MethodType(
                            _todict, atoms.calc
                        )
                    db.update(_id, atoms=atoms)
                    self.done_jobs.append((dof_key, grid_key, _id))
                else:
                    logger.warn(
                        f"ERROR:\t{error}\tDOFs={dof_key}\tgrids={grid_key}\tID={_id}"
                    )
                if os.path.exists(f"{self.jobname}/{_id:07}"):
                    shutil.rmtree(f"{self.jobname}/{_id:07}")
                break
            elif time.time() - start_time > timeout:
                try:
                    if not future.running():
                        future.cancel()
                    # Wait 1.0 second and discard the job
                    future.result(1.0)
                except Exception:
                    # TimeoutError or CancelledError
                    pass
                logger.warning(
                    f"TIMEOUT:\tDOFs={dof_key}\tgrids={grid_key}\tID={_id} "
                )
                if os.path.exists(f"{self.jobname}/{_id:07}"):
                    shutil.rmtree(f"{self.jobname}/{_id:07}")
                break
            else:
                time.sleep(1)
                self.thrown_jobs.append(
                    (future, dof_key, grid_key, _id, start_time)
                )
