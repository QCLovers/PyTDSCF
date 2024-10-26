"""
This module have been deprecated.
But for testing, this module is still used.
"""

import itertools
import math

import numpy as np

from pytdscf import units
from pytdscf.basis import PrimBas_HO

ndof_per_site = 1


def matJ_1D_exciton(
    nmol,
    nspf,
    s0,
    s1,
    coupleJ,
    *,
    deltaE=0.0,
    coupleE=0.0,
    coupleH=0.0,
    ndof_per_site=1,
    with_CT=False,
):
    statelist = []
    """F.Exciton-State"""
    for imol in range(nmol):
        imol_hole = imol_elec = imol
        statelist.append((imol_hole, imol_elec))
    """CT-State"""
    if with_CT:
        for imol_hole, imol_elec in itertools.permutations(range(nmol), 2):
            statelist.append((imol_hole, imol_elec))

    prim_info = []
    for (
        imol_hole,
        imol_elec,
    ) in statelist:  # itertools.product(range(nmol), repeat=2):
        is_FE_imol = imol_hole == imol_elec
        if is_FE_imol:
            # prim_info.append([s1 if x==imol_hole else s0 for x in range(nmol)])
            prim_info.append(
                list(
                    itertools.chain.from_iterable(
                        [s1 if x == imol_hole else s0 for x in range(nmol)]
                    )
                )
            )
        else:
            raise NotImplmentedError(f"{is_FE_imol=}")
            # prim_info.append(
            #     list(
            #         itertools.chain.from_iterable(
            #             [
            #                 hole
            #                 if x == imol_hole
            #                 else elec
            #                 if x == imol_elec
            #                 else s0
            #                 for x in range(nmol)
            #             ]
            #         )
            #     )
            # )

    matJ = []
    nstate = len(prim_info)
    for (
        imol_hole,
        imol_elec,
    ) in statelist:  # itertools.product(range(nmol), repeat=2):
        matJ_istate = []
        for (
            jmol_hole,
            jmol_elec,
        ) in statelist:  # itertools.product(range(nmol), repeat=2):
            is_FE_imol = imol_hole == imol_elec
            is_FE_jmol = jmol_hole == jmol_elec

            # TEMP>            if   imol_hole == None and jmol_hole == None:
            # TEMP>                matJ_istate.append(0.0)
            # TEMP>            elif imol_hole == None or  jmol_hole == None:
            # TEMP>                matJ_istate.append(0.0)
            # TEMP>            elif is_FE_imol and is_FE_jmol:
            if is_FE_imol and is_FE_jmol:
                distance = abs(imol_hole - jmol_hole)
                if distance == 0:
                    """<FE|H|FE>:diagonal"""
                    matJ_istate.append(0.0)
                elif distance == 1:
                    """<FE|H|FE+1>:nearest-neighbor"""
                    matJ_istate.append(coupleJ)
                else:
                    """<FE|H|FE+n>:too far to interact"""
                    matJ_istate.append(0.0)
            else:
                distance_hole = abs(imol_hole - jmol_hole)
                distance_elec = abs(imol_elec - jmol_elec)
                if distance_hole == 0 and distance_elec == 0:
                    """<CT|H|CT>:diagonal"""
                    matJ_istate.append(deltaE)
                elif distance_hole == 1 and distance_elec == 0:
                    matJ_istate.append(coupleH)
                elif distance_hole == 0 and distance_elec == 1:
                    matJ_istate.append(coupleE)
                else:
                    matJ_istate.append(0.0)

        assert len(matJ_istate) == nstate, "matJ_istate:{0}".format(matJ_istate)
        matJ.append(matJ_istate)

    # spf_info = [[nspf for i in range(nmol)] for j in range(nstate)]
    spf_info = [[nspf for i in prim_istate] for prim_istate in prim_info]

    ndof_per_sites = [
        ndof_per_site for j in range(len(prim_info[0]) // ndof_per_site)
    ]
    if nmol % ndof_per_site != 0:
        ndof_per_sites.append(len(prim_info[0]) % ndof_per_site)

    return prim_info, spf_info, ndof_per_sites, matJ


def matJ_2D_exciton(nmol_row, nmol_col, nspf, coupleJ, s0, s1):
    """F.Exciton-State"""
    statelist = [
        (i, j) for i, j in itertools.product(range(nmol_row), range(nmol_col))
    ]

    prim_info = []
    for i, j in statelist:
        # prim_info.append([s1 if i==a and j==b else s0 for a,b in statelist])
        prim_info.append(
            list(
                itertools.chain.from_iterable(
                    [s1 if i == a and j == b else s0 for a, b in statelist]
                )
            )
        )

    nstate = len(prim_info)
    matJ = np.zeros((nstate, nstate))
    for ist, (row_bra, col_bra) in enumerate(statelist):
        for jst, (row_ket, col_ket) in enumerate(statelist):
            dist_row = abs(row_bra - row_ket)
            dist_col = abs(col_bra - col_ket)
            if dist_row + dist_col == 1:
                matJ[ist, jst] = coupleJ

    matJ = matJ.tolist()

    # spf_info = [[nspf for i in range(nmol_row*nmol_col)] for j in range(nstate)]
    spf_info = [[nspf for i in prim_istate] for prim_istate in prim_info]

    ndof_per_sites = [
        ndof_per_site for j in range((nmol_row * nmol_col) // ndof_per_site)
    ]
    if (nmol_row * nmol_col) % ndof_per_site != 0:
        ndof_per_sites.append((nmol_row * nmol_col) % ndof_per_site)

    return matJ, prim_info, spf_info, ndof_per_sites


###################################################
def matJ_LH2_exciton(nspf):
    # omega_cm1, facHS =  23.3, 0.017
    # omega_cm1, facHS =  88.2, 0.020
    omega_cm1, facHS = 203.3, 0.056
    # omega_cm1, facHS = 361.6, 0.044
    # omega_cm1, facHS = 562.6, 0.021
    # omega_cm1, facHS = 748.2, 0.050
    # omega_cm1, facHS = 915.7, 0.051

    nmol = 27
    """LH2 27-pigment model"""
    x = omega_cm1 * facHS
    miniJ = [
        [490 - x, 27, 3, -25],
        [690 - x, 307, -12, -51],
        [70 - x, -3, 237, -35],
    ]
    matJ = np.zeros((nmol, nmol))
    for i in range(0, nmol, 3):
        if i < 24:
            matJ[i, i : i + 4] = miniJ[0][:]
            matJ[i + 1, i + 1 : i + 1 + 4] = miniJ[1][:]
            matJ[i + 2, i + 2 : i + 2 + 4] = miniJ[2][:]
        if i == 24:
            matJ[i, i : i + 3] = miniJ[0][:3]
            matJ[i + 1, i + 1 : i + 1 + 2] = miniJ[1][:2]
            matJ[i + 2, i + 2 : i + 2 + 1] = miniJ[2][:1]
        if i == 0:
            matJ[i, 24 : 24 + 3] = [miniJ[0][3], miniJ[1][2], miniJ[2][1]]
            matJ[i + 1, 25 : 25 + 2] = [miniJ[1][3], miniJ[2][2]]
            matJ[i + 2, 26 : 26 + 1] = [miniJ[2][3]]
    #        if i < 24:
    #            matJ[i  , i  :i   +3] = [690, 307, -12]
    #            matJ[i+1, i+1:i+1 +3] = [ 70,  -3, 237]
    #            matJ[i+2, i+2:i+2 +3] = [490,  27,   3]
    #        if i == 24:
    #            matJ[i  , i  :i   +3] = [690, 307, -12]
    #            matJ[i+1, i+1:i+1 +2] = [ 70,  -3]
    #            matJ[i+2, i+2:i+2 +1] = [490]
    #        if i == 0:
    #            matJ[i  , 25  :25 +2] = [237,  27]
    #            matJ[i+1, 26  :26 +1] = [  3]
    for i in range(0, nmol):
        for j in range(0, i):
            matJ[i, j] = matJ[j, i]
    np.set_printoptions(formatter={"float": "{:5.0f}".format}, linewidth=200)
    print(matJ)

    nstate = nmol
    imol_reorder = [
        0,
    ]
    for i in range(1, (nstate + 1) // 2):
        imol_reorder.append(i)
        imol_reorder.append(nmol - i)
    print(len(imol_reorder), imol_reorder)

    matJ = matJ[:, imol_reorder]
    matJ = matJ[imol_reorder, :]
    print(matJ)
    matJ = (matJ / units.au_in_cm1).tolist()

    gs = PrimBas_HO(0.0, omega_cm1, 8)
    qy = PrimBas_HO(math.sqrt(2 * facHS), omega_cm1, 8)  # =(2S)^0.5 (S=0.051)
    prim_info = []
    for istate in range(nstate):
        # INCORRECT        prim_info.append([qy if imol==imol_reorder[istate] else gs for imol in range(nmol)])
        prim_info.append([qy if imol == istate else gs for imol in range(nmol)])

    print(prim_info)
    spf_info = [[nspf for i in range(nmol)] for j in range(nstate)]

    ndof_per_sites = [ndof_per_site for j in range(nmol // ndof_per_site)]
    if nmol % ndof_per_site != 0:
        ndof_per_sites.append((nmol) % ndof_per_site)

    return matJ, prim_info, spf_info, ndof_per_sites


###################################################
