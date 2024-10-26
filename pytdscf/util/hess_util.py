"""Hessian Utility Scripts"""

from typing import List, Optional, Tuple

import numpy as np
import scipy.linalg

from pytdscf import units


def read_minfo(
    file_path,
    use_trans: Optional[bool] = False,
    use_rot: Optional[bool] = False,
):
    """
    Read `.minfo` files from MakePES in SINDO package. Not supported localized coordinate by JSindo.

    Args:
        file_path (str) : Path to ``.minfo`` file.
        use_trans (bool) : Use translational vector. Defaults to ``False``.
        use_rot (bool) : Use rotatinal vector. Defaults to ``False``.

    Returns:
        tuple : multiple contents listed below.

    Return contents:
        - list(str) : atoms
        - numpy.ndarray : mass in AMU.
        - numpy.ndarray : coordinate in a.u. \
                Shape is ``(natom, 3)`` , where ``natom = len(atoms)``.
        - numpy.ndarray : frequency in cm-1. Shape is ``(ndof, )``, \
                where ``ndof = 3*natom-6``.
        - numpy.ndarray : displacement vectors in a.u. .\
                Shape is ``(ndof, natom, 3)``. Not mass weighted and normalized !!

    """
    atom = []
    geom = []
    freq = []
    disp_mwc = []
    mass = []
    read_geom = False
    read_disp = False
    read_freq = False
    file = open(file_path, "r")
    for line in file:
        if line == "[ Atomic Data ]\n":
            read_geom = True
            continue
        elif line == "Translational Frequency\n" and use_trans:
            read_freq = True
            read_disp = False
            continue
        elif line == "Translational vector\n" and use_trans:
            read_freq = False
            read_disp = True
            continue
        elif line == "Rotational Frequency\n" and use_rot:
            read_freq = True
            read_disp = False
            continue
        elif line == "Rotational vector\n" and use_rot:
            read_freq = False
            read_disp = True
            continue
        elif line == "Vibrational Frequency\n":
            read_freq = True
            read_disp = False
            continue
        elif line == "Vibrational vector\n":
            read_freq = False
            read_disp = True
            continue

        elif read_geom:
            if line == "\n":
                read_geom = False
                continue
            else:
                line = line.replace(",", "")
                words = line.split()
                if len(words) == 6:
                    atom.append(words[0])
                    mass.append(float(words[2]))
                    geom.append([float(w) for w in words[3:6]])
                else:
                    natom = int(words[0])
                continue
        elif read_freq:
            if line[-2] == " ":
                line = line.replace(",", "")
                words = line.split()
                freq.extend([float(w) for w in words])
                continue
        elif read_disp:
            if line[-2] == " ":
                line = line.replace(",", "")
                words = line.split()
                disp_mwc.extend([float(w) for w in words])
                continue

    mass = np.array(mass)
    geom = np.array(geom)
    freq = np.array(freq)
    ndof = len(disp_mwc) // 3 // natom
    # disp = [[d for d in disp_mwc[3*natom*idof : 3*natom*(idof+1)]] for idof in range(ndof)]
    disp_mwc = np.array(disp_mwc).reshape(ndof, natom, 3)
    disp = np.zeros_like(disp_mwc)
    for iatom in range(natom):
        disp[:, iatom, :] = disp_mwc[:, iatom, :] / np.sqrt(mass[iatom])

    return (atom, mass, geom, freq, disp)


def read_fchk_g16(
    file_path: str,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read Hessian in `.fchk` files from Gaussian16.

    Args:
        file_path (str) : Path to ``fchk`` file.
        use_trans (bool) : Use translational vector. Defaults to ``False``.
        use_rot (bool) : Use rotatinal vector. Defaults to ``False``.

    Returns:
        Tuple : multiple contents listed below.

    Return contents:
        - list(str) : atoms
        - numpy.ndarray : mass in AMU.
        - numpy.ndarray : coordinate in a.u. \
                Shape is ``(natom, 3)`` , where ``natom = len(atoms)``.
        - numpy.ndarray : frequency in cm-1. Shape is ``(ndof, )``, \
                where ``ndof`` = ``3*natom-6`` or ``3*natom-5``.\
                ``Nan`` represents imaginary frequency.
        - numpy.ndarray : displacement vectors in a.u. .\
                Shape is ``(ndof, natom, 3)``. Not mass weighted and normalized !!

    """
    from mendeleev import element

    atom = []
    d1_hess_cartesian = []
    mass = []
    geom = []

    atom_flag = False
    geom_flag = False
    hessian_flag = False
    mass_flag = False

    file = open(file_path, "r")
    lines = file.readlines()
    file.close()
    for line in lines:
        if line.startswith("Cartesian Force Constants"):
            hessian_flag = True
            continue
        elif line.startswith("Vib-AtMass"):
            mass_flag = True
            continue
        elif line.startswith("Opt point       1 Geometries"):
            geom_flag = True
            continue
        elif line.startswith("Atomic numbers"):
            atom_flag = True
            continue

        if hessian_flag | mass_flag | geom_flag | atom_flag:
            if line[-5] != "E" and not atom_flag:
                hessian_flag = False
                mass_flag = False
                geom_flag = False
                continue
            if line.startswith("Nuclear charges"):
                atom_flag = False
                continue

            if hessian_flag:
                d1_hess_cartesian.extend(list(map(float, line.split())))
            elif mass_flag:
                mass.extend(list(map(float, line.split())))
            elif geom_flag:
                geom.extend(list(map(float, line.split())))
            elif atom_flag:
                atomic_num = list(map(int, line.split()))
                atom.extend([element(n).symbol for n in atomic_num])

    def to_symmetric(v, size):
        out = np.zeros((size, size), v.dtype)
        out[np.tri(size, dtype=bool)] = v
        return out.T + out - np.diag(np.diag(out))

    natom = len(atom)
    geom = np.array(geom).reshape(natom, 3)
    hess_cartesian = to_symmetric(np.array(d1_hess_cartesian), natom * 3)
    mass = np.array(mass)
    trans_mass_weighted = 1 / np.sqrt(
        np.repeat(mass / units.au_in_dalton, 3)
    ).reshape(-1, 1)
    mass_weighted_hessian = np.multiply(
        hess_cartesian, trans_mass_weighted @ trans_mass_weighted.T
    )

    E, V = scipy.linalg.eigh(mass_weighted_hessian)
    freq = np.sqrt(E) * units.au_in_cm1

    ndof = V.shape[0]
    disp_mwc = np.array(V.T).reshape(ndof, natom, 3)
    disp = np.zeros_like(disp_mwc)
    for iatom in range(natom):
        disp[:, iatom, :] = disp_mwc[:, iatom, :] / np.sqrt(mass[iatom])

    return (atom, mass, geom, freq, disp)


def get_displce_from_hess(
    hess: np.ndarray,
    mass: List[float],
    massweighted: Optional[bool] = True,
    onedim: Optional[bool] = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get displacement vector (normal mode) from mass-weighted hessian

    Args:
        hess (np.ndarray): hessian in a.u.
        mass (List[float]): List of atom weight in AMU.
        massweighted (Optional[bool]) : input hessian is mass-weighted hessian. Defaults to True
        onedim (Optional[bool]) : input hessian is one-dimensional array of upper-triangle part.\
            Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray] : Non-mass weighted displacement vectors in a.u. and frequency in cm-1.

    """

    def to_symmetric(v, size):
        out = np.zeros((size, size), v.dtype)
        out[np.tri(size, dtype=bool)] = v
        return out.T + out - np.diag(np.diag(out))

    natom = len(mass)
    if onedim:
        hess = to_symmetric(np.array(hess), natom * 3)
    mass = np.array(mass)
    if massweighted:
        mass_weighted_hessian = hess
    else:
        trans_mass_weighted = 1 / np.sqrt(
            np.repeat(mass / units.au_in_dalton, 3)
        ).reshape(-1, 1)
        mass_weighted_hessian = np.multiply(
            hess, trans_mass_weighted @ trans_mass_weighted.T
        )

    E, V = scipy.linalg.eigh(mass_weighted_hessian)
    freq = np.sqrt(E) * units.au_in_cm1
    ndof = V.shape[0]
    disp_mwc = np.array(V.T).reshape(ndof, natom, 3)
    disp = np.zeros_like(disp_mwc)
    for iatom in range(natom):
        disp[:, iatom, :] = disp_mwc[:, iatom, :] / np.sqrt(
            mass[iatom] / units.au_in_dalton
        )

    return disp, freq
