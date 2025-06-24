from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()


import platform
import sys

import pytdscf


print(sys.version)
print(f"pytdscf version = {pytdscf.__version__}")
print(platform.platform())

from itertools import chain

import numpy as np
from pympo.utils import import_npz

from pytdscf import (
    BasInfo,
    Boson,
    Exciton,
    Model,
    Simulator,
    TensorHamiltonian,
    TensorOperator,
)

n_order = 61  # number of sites for each side of the exciton
basis = (
    [Boson(8)] * n_order
    + [Exciton(nstate=3, names=["S1", "TT", "CS"])]
    + [Boson(8)] * (2 * n_order)
)
basinfo = BasInfo([basis])
ndim = len(basis)
print(ndim)
sys_site = n_order
backend = "numpy" # "jax" is not supported for MPI.


if rank == 0:
    # Hamiltonian prepared in example 12.
    pot_mpo = import_npz("singlet_fission_mpo.npz")
    potential = [
        [{tuple((k, k) for k in range(ndim)): TensorOperator(mpo=pot_mpo)}]
    ]  # key is ((0,0), 1, 2, ..., ndim-1)
else:
    potential = None

H = TensorHamiltonian(
    ndof=len(basis), potential=potential, kinetic=None, backend=backend
)

if rank == 0:
    core = np.zeros((1, 3, 1))
    core[0, 0, 0] = 1.0
    potential = [[{(sys_site,): TensorOperator(mpo=[core], legs=(sys_site,))}]]
else:
    potential = None
s1 = TensorHamiltonian(
    ndof=len(basis),
    potential=potential,
    kinetic=None,
    backend=backend,
)
operators = {"hamiltonian": H, "S1": s1}

sites = chain(
    range(sys_site - 1, -1, -4),
    range(sys_site + 1, 3 * n_order + 1, 8),
    range(sys_site + 2, 3 * n_order + 1, 8),
)
for isite in sites:
    if rank == 0:
        core = np.zeros((1, basis[isite].nprim, 1))
        core[0, :, 0] = np.arange(basis[isite].nprim)
        potential = [[{(isite,): TensorOperator(mpo=[core], legs=(isite,))}]]
    else:
        potential = None
    n = TensorHamiltonian(
        ndof=len(basis),
        potential=potential,
        kinetic=None,
        backend=backend,
    )
    operators[f"N{isite}"] = n


model = Model(basinfo=basinfo, operators=operators)
model.m_aux_max = 1  # Initial bond dimension
# Starts from S1 state
init_boson = [[1.0] + [0.0] * (basis[1].nprim - 1)]
model.init_HartreeProduct = [
    init_boson * n_order + [[1.0, 0.0, 0.0]] + init_boson * (2 * n_order)
]


nproc = size
D = 30
proj = 5
svd = 6

jobname = f"singlet_fission_{D}D-0{proj}proj-0{svd}svd-{nproc}cores"
simulator = Simulator(jobname=jobname, model=model, backend=backend, verbose=2)
simulator.propagate(
    maxstep=2000,
    stepsize=0.2,  # This value affects the accuracy of the simulation.
    reduced_density=(
        [(sys_site, sys_site)],
        10,
    ),  # we want to know diagonal_element of (|S1><S1| |CT><CT| |TT><TT| |S1><CT| |S1><TT| |CS><TT|)
    energy=False,
    autocorr=False,
    norm=False,
    populations=False,
    observables=True,
    observables_per_step=10,
    parallel_split_indices=[
        (184 * k // nproc, 184 * (k + 1) // nproc - 1) for k in range(nproc)
    ],
    adaptive=True,
    adaptive_Dmax=D,
    adaptive_dD=D,
    adaptive_p_proj=pow(10, -proj),
    adaptive_p_svd=pow(10, -svd),
)
