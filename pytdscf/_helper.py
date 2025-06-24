"""
Debugging and helper functions
"""

from __future__ import annotations

import itertools
import sys
from collections import defaultdict

import jax
import numpy as np
import scipy.linalg

from pytdscf._site_cls import SiteCoef


class _Debug:
    """

    Debug class

    """

    nstep_print = 1
    count0 = 0
    count1 = 0
    # number of iteration in SIL
    niter_krylov = defaultdict(int)
    site_now = 0  # which site is updating now


class _NFlops:
    """

    Counter of the number of floating point operations. Mainly used for debug.

    """

    spf_deri = 0
    """
    int : The number of floating point operations in SPF derivatives.
    """
    ci_expo = 0
    """
    int : The number of floating point operations in A-vector SIL.
    """
    ci_renm = 0
    """
    int : The number of floating point operations
        in A-vector MPS renormalization.
    """
    ci_mfop = 0
    """
    int : The number of floating point operations
        in mean field operator construction.
    """


class _ElpTime:
    """

    The memo of the elapsed time in each section. Mainly used for debug.

    """

    steps = 0.0
    itrf = 0.0
    mfop = 0.0
    mfop_0 = 0.0
    mfop_1 = 0.0
    mfop_2 = 0.0
    mfop_3 = 0.0
    mfop_4 = 0.0
    mfop_gen_old = 0.0
    mfop_gen_new = 0.0
    ci = 0.0
    ci_rnm = 0.0
    ci_exp = 0.0
    ci_sw0 = 0.0
    ci_sw1 = 0.0
    ci_etc = 0.0
    ci_ham_0 = 0.0
    ci_ham_1 = 0.0
    ci_ham_2 = 0.0
    ci_ham_3 = 0.0
    ci_ham_4 = 0.0
    spf = 0.0
    dot = 0.0
    dot_new = 0.0
    dot_new1 = 0.0
    dot_new2 = 0.0
    dbg_0 = 0.0
    dbg_1 = 0.0
    dbg_2 = 0.0
    dbg_3 = 0.0
    dbg_4 = 0.0
    dbg_5 = 0.0
    dbg_6 = 0.0
    dbg_7 = 0.0
    zgemm = 0.0


def spf_orthogonal_check(ints_spf_ovi, spf_coef):
    """
    Check SPF orthogonality for debug. Not used now...
    """
    norm_max = []
    norm_min = []
    orth_absmax = []
    for istate in range(spf_coef.nstate):
        for idof in range(spf_coef.ndof):
            ovi = ints_spf_ovi[(istate, istate)][idof]
            norm_max = np.max(np.diag(ovi))
            norm_min = np.max(np.diag(ovi))
            orth_absmax = np.max(np.absolute(np.tril(ovi, -1)))

    print(
        f"norm_max:{norm_max:20.15f}"
        + f"norm_min:{norm_min:20.15f} "
        + f"orth_absmax:{orth_absmax:20.15f}"
    )


def matrix_regularized_inverse(mat, epsrho):
    """

    Construct mean field operator (mfop) density (rho) \
    inverse part by regularization.
    See also :

    - https://doi.org/10.1063/1.5024859
    - https://doi.org/10.1063/1.5042776

    Args:
        mat (numpy.ndarray) : Mean field operator overlap. \
            shape is (nspf, nspf), dtype = complex
        epsrho (float) : The eigenvalue threshold to apply regularization.

    Returns:
        numpy.ndarray : Inverse matrix of mfop overlap. \
            Shape and dtype is the same as input arg `mat`.

    """

    """by the "-"(mat), the ordering of eigen vectors are \
        reversed from linalg.eigh convension"""
    minus_to_order_eigvals = -1.0
    eigvals, eigvecs = scipy.linalg.eigh(minus_to_order_eigvals * mat)
    eigvals *= minus_to_order_eigvals

    eigvals_reg = np.where(
        eigvals > 64.0e0 * epsrho,
        eigvals,
        eigvals + epsrho * np.exp(-eigvals / epsrho),
    )
    invmat = eigvecs.dot(np.diag(1.0 / eigvals_reg)).dot(np.conj(eigvecs).T)

    diag_inds = np.diag_indices_from(invmat)
    invmat[diag_inds] = np.real(invmat[diag_inds])

    return invmat


def trans_mps2fci(mps_coef, basinfo):
    """

    Transform MPS-MCTDH to MCTDH

    Args:
        mps_coef (mps_cls.MPSCoef) : MPS site before transformation.
        basinfo (model_cls.BasInfo) : Information of MPS site basis.

    Returns:
        ci_cls.CICoef : A-vector after transformation.

    """
    from pytdscf._ci_cls import CICoef

    fci_coef = CICoef.alloc(basinfo)
    for istate, superblock in enumerate(mps_coef.superblock_states):
        fci_in_site_rep = build_CIcoef(superblock)
        fci_in_spf_rep = fci_in_site_rep.reshape(basinfo.get_nspf_list(istate))
        for J in itertools.product(
            *[range(x) for x in basinfo.get_nspf_list(istate)]
        ):
            fci_coef[istate][J] = fci_in_spf_rep[J]
    return fci_coef


def diagonalize_CI(ci_coef, ints_spf, matH):
    """SILLY implementation of diagonalization of H_ciop

    Diagonalize MCTDH A-vector for improved relaxation.

    Args:
        ci_coef (ci_cls.CICoef) : A-vector
        ints_spf (spf_cls.SPFInts) : integrals of SPF
        matH (hamiltonian_cls.PolynomialHamiltonian) : PolynomialHamiltonian

    Returns:
        None

    To Do:
        This method could be merged to `helper.matrix_diagonalize_lanczos`.

    """
    from _ci_cls import multiplyH_CI  # type: ignore

    unitvec = ci_coef.alloc_zeros_like()
    nstate = ci_coef.nstate

    multiplyH = multiplyH_CI(ints_spf, matH, ci_coef)

    Hmat: list[list[np.ndarray]]
    Hmat = [
        [None for i in range(nstate)]  # type: ignore # noqa
        for j in range(nstate)
    ]
    for istate_ket, istate_bra in itertools.product(range(nstate), repeat=2):
        Hmat[istate_bra][istate_ket] = np.zeros(
            (ci_coef.size(istate_bra), ci_coef.size(istate_ket)), dtype=complex
        )
        nspf_rangelist_ket = [range(x) for x in ci_coef[istate_ket].shape]
        nspf_rangelist_bra = [range(x) for x in ci_coef[istate_bra].shape]
        for iket, ci_idx_ket in enumerate(
            itertools.product(*nspf_rangelist_ket)
        ):
            unitvec.fill(0.0 + 0.0j)
            unitvec[istate_ket][ci_idx_ket] = 1.0 + 0.0j
            sigmavec = multiplyH.dot(unitvec)
            for ibra, ci_idx_bra in enumerate(
                itertools.product(*nspf_rangelist_bra)
            ):
                unitvec.fill(0.0 + 0.0j)
                unitvec[istate_bra][ci_idx_bra] = 1.0 + 0.0j
                Hmat[istate_bra][istate_ket][ibra, iket] = sigmavec.dot_conj(
                    unitvec
                )
    Hmat_row = []
    for istate_bra in range(nstate):
        Hmat_row.append(np.concatenate(Hmat[istate_bra], axis=1))
    Hmat = np.concatenate((Hmat_row), axis=0)
    _, evecs = scipy.linalg.eigh(Hmat)
    ci_gs_1d = evecs[:, 0]
    for istate_ket in range(nstate):
        nspf_rangelist_ket = [range(x) for x in ci_coef[istate_ket].shape]
        for iket, ci_idx_ket in enumerate(
            itertools.product(*nspf_rangelist_ket)
        ):
            ci_coef[istate_ket][ci_idx_ket] = ci_gs_1d[
                istate_ket * ci_coef.size(istate_ket) + iket
            ]


def build_CIcoef(superblock):
    """
    build CIcoef from MPS [ONLY used for debugging]
    """
    ci_coef = np.array(superblock[0].data)
    for isite in range(1, len(superblock)):
        ci_coef = np.tensordot(ci_coef, superblock[isite], axes=1)
    return ci_coef


def progressbar(it, prefix="", size=40, out=sys.stdout):
    """Progress bar in for loop

    Args:
        it : Iterative class
        prefix (str, optional): Defaults to "".
        size (int, optional): Defaults to 40.
        out (optional): Defaults to sys.stdout.

    Yields:
        _type_: _description_
    """
    count = len(it)

    def show(j):
        x = int(size * j / count)

        print(
            f"{prefix}[{'█' * x}{'.' * (size - x)}] {j}/{count}",
            end="\r",
            file=out,
            flush=True,
        )

    show(0)
    for i, item in enumerate(it):
        yield item
        show(i + 1)
    print("\n", flush=True, file=out)


def get_tensornetwork_diagram_MPS(superblock: list[SiteCoef]):
    """Get diagram of tensor network

    Args:
        superblock (List[SiteCoef]): MPS superblock

    Returns:
        str: diagram of tensor network
    """

    """
    join core __repr__ to string
    when concatenate
    superblock[0].__repr__()
       [5]
        |
    [1]-C-[3]
     ,
    superblock[1].__repr__()
       [5]
        |
    [3]-R-[4]
    and superblock[2].__repr__()
       [5]
        |
    [4]-R-[1]
    the result (=figure) is
       [5]   [5]   [5]
        |     |     |
    [1]-C-[3]-R-[4]-R-[1]
    """

    line1 = ""
    line2 = ""
    line3 = ""
    figure = ""
    for i, core in enumerate(superblock):
        splitlines = core.__repr__().splitlines()
        line1 += splitlines[1]
        line2 += splitlines[2]
        if i == len(superblock) - 1:
            line3 += splitlines[3]
        else:
            line3 += splitlines[3][: -len(str(core.data.shape[-1])) - 2]
        if i != len(superblock) - 1:
            length = max(len(line1), len(line2), len(line3))
            line1 += " " * (length - len(line1))
            line2 += " " * (length - len(line2))
            line3 += "-" * (length - len(line3))
            if length > 100:
                figure += "\n" + line1 + "\n" + line2 + "\n" + line3 + "\n"
                line1 = ""
                line2 = ""
                line3 = ""
    figure = "\n" + line1 + "\n" + line2 + "\n" + line3 + "\n"
    return figure


def get_tensornetwork_diagram_MPO(
    mpo: list[np.ndarray] | list[jax.Array],
) -> str:
    """Get diagram of tensor network

    Args:
        mpo (List[np.ndarray | jax.Array]): MPO

    Returns:
        str: diagram of tensor network
    """
    # When mpo[0].shape = (1, 5, 5, 4), mpo[1].shape = (4, 6, 6, 4), mpo[2].shape = (4, 7, 7, 1)
    # the result (=figure) is
    #    [5]   [6]   [7]
    #     |     |     |
    # [1]-W-[4]-W-[4]-W-[1]
    #     |     |     |
    #    [5]   [6]   [7]

    # When mpo[0].shape = (1, 5, 4), mpo[1].shape = (4, 6, 4), mpo[2].shape = (4, 7, 1)
    # the result (=figure) is
    #    [5]   [6]   [7]
    #     |     |     |
    # [1]-W-[4]-W-[4]-W-[1]

    line1 = ""
    line2 = ""
    line3 = ""
    line4 = ""
    line5 = ""
    for i, core in enumerate(mpo):
        line1 += " " * len(f"[{core.shape[0]}]") + f"[{core.shape[1]}]"
        line2 += " " * len(f"[{core.shape[0]}]-") + "|"
        line3 += f"[{core.shape[0]}]-W-"
        if len(core.shape) == 4:
            line4 += " " * len(f"[{core.shape[0]}]-") + "|"
            line5 += " " * len(f"[{core.shape[0]}]") + f"[{core.shape[2]}]"
        if i == len(mpo) - 1:
            line3 += f"[{core.shape[-1]}]"

        length = max(len(line1), len(line2), len(line3))
        if len(core.shape) == 4:
            length = max(length, len(line4), len(line5))
            line4 += " " * (length - len(line4))
            line5 += " " * (length - len(line5))
        line1 += " " * (length - len(line1))
        line2 += " " * (length - len(line2))
        line3 += "-" * (length - len(line3))

        if length > 100:
            figure = "\n" + line1 + "\n" + line2 + "\n" + line3 + "\n"
            if len(core.shape) == 4:
                figure += line4 + "\n" + line5 + "\n"
                line4 = ""
                line5 = ""
            line1 = ""
            line2 = ""
            line3 = ""
    figure = "\n" + line1 + "\n" + line2 + "\n" + line3 + "\n"
    if len(core.shape) == 4:
        figure += line4 + "\n" + line5 + "\n"

    return figure


def to_dbkey(arr) -> str:
    """Change Array into Database key
    This is because Database does not accept array as key.

    Args:
        arr (_ArrayLike) : Array to be changed into key such as [1,2,3,4,5]

    Returns:
        str : Database key such as "|1 2 3 4 5"

    """
    return "|" + " ".join(map(str, arr))


def from_dbkey(key) -> tuple[int, ...]:
    """Change Database key into Array
    This is because Database does not accept array as key.

    Args:
        key (str) : Database key such as "|1 2 3 4 5"

    Returns:
        Tuple[int, ...] : Array such as (1,2,3,4,5)

    """
    return tuple(map(int, key[1:].split()))


def rank0_only(func):
    """Decorator to run a function only on MPI rank 0

    Args:
        func: Function to be decorated

    Returns:
        Wrapped function that only executes on rank 0
    """
    from functools import wraps

    from pytdscf._const_cls import const

    @wraps(func)
    def wrapper(*args, **kwargs):
        if const.mpi_rank == 0:
            return func(*args, **kwargs)
        else:
            return None

    return wrapper
