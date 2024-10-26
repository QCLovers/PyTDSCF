"""MakePES to PyTDSCF polynomial PES converter (least-aquare based)

.. code-block :: bash

   $ python grid2qff.py pes_mrpes

"""

import glob
import sys
from collections import defaultdict
from math import factorial

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize

debye2au = 2.541762289


def basis_func(ijk, q_ijk):
    """if ijk = '121' return q_i^1*q_j^2*q_k^1"""
    dum = np.ones(q_ijk[0].shape[0])

    for index in range(len(ijk)):
        if ijk[index] > 0:
            dum *= pow(q_ijk[index], ijk[index])
    return dum


def fiting_func1MR(param, q_i, data, ijk_pairs):
    residual = np.zeros(data.size)
    for i in range(len(ijk_pairs)):
        residual -= param[i] * basis_func(ijk_pairs[i], [q_i])
    residual += data
    return residual


def fiting_func2MR(param, q_i, q_j, data, ijk_pairs):
    residual = np.zeros(data.size)
    for ij in range(len(ijk_pairs)):
        residual -= param[ij] * basis_func(ijk_pairs[ij], [q_i, q_j])
    residual += data
    return residual


def fiting_func3MR(param, q_i, q_j, q_k, data, ijk_pairs):
    residual = np.zeros(data.size)
    for ijk in range(len(ijk_pairs)):
        residual -= param[ijk] * basis_func(ijk_pairs[ijk], [q_i, q_j, q_k])
    residual += data
    return residual


def fiting_func4MR(param, q_i, q_j, q_k, q_l, data, ijk_pairs):
    residual = np.zeros(data.size)
    for ijkl in range(len(ijk_pairs)):
        residual -= param[ijkl] * basis_func(
            ijk_pairs[ijkl], [q_i, q_j, q_k, q_l]
        )
    residual += data
    return residual


def debug_plot(nMR, q_ijk, optimised_param, data, ijk_pairs, ijk_name):
    if nMR != 2:
        raise NotImplementedError("Cannot plot for nMR != 2")

    N = 100
    x1 = np.linspace(min(q_ijk[0]), max(q_ijk[0]), N)
    x2 = np.linspace(min(q_ijk[1]), max(q_ijk[1]), N)

    X1, X2 = np.meshgrid(x1, x2)

    f = np.zeros(X1.size)
    Q_ijk = [X1.flatten(), X2.flatten()]
    for ijk in range(len(ijk_pairs)):
        f += optimised_param[0][ijk] * basis_func(ijk_pairs[ijk], Q_ijk)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(*q_ijk, data, color="green")
    ax.plot_surface(X1, X2, f.reshape(N, N), cmap="bwr")

    # ax.set_xlabel(ijk_name[0])
    # ax.set_ylabel(ijk_name[1])
    ax.set_xlabel(r"$q_1$")
    ax.set_ylabel(r"$q_2$")
    ax.set_zlabel("energy")

    plt.savefig(ijk_name[0] + ijk_name[1] + ".pdf")

    # plt.show()


def make_ijk(ijk, index, ijk_pairs, nMR, max_order):
    ijk = list(ijk)
    if sum(ijk) < max_order:  # 4th order
        if index < nMR - 1:
            make_ijk(tuple(ijk), index + 1, ijk_pairs, nMR, max_order)
        ijk[index] += 1
        ijk_pairs.append(tuple(ijk))
        ijk = tuple(ijk)
        make_ijk(tuple(ijk), index, ijk_pairs, nMR, max_order)


def file_write(k_orig, mu, name):
    if mu:
        output_file_name = name + "_dipole.py"
        file = open(output_file_name, "w")
        file.write("# grid dipole surface from MakePES" + "\n")
        file.write("from collections import defaultdict" + "\n")
        file.write("mu = defaultdict(float)" + "\n")
        for key, val in mu.items():
            file.write(
                "mu["
                + str(key)
                + "]"
                + " " * (17 - len(str(key)))
                + "= ["
                + "{:>21.12e}".format(val[0])
                + ", "
                + "{:>21.12e}".format(val[1])
                + ", "
                + "{:>21.12e}".format(val[2])
                + " ]\n"
            )
        file.close()
        print(output_file_name + " is created.")
    if k_orig:
        output_file_name = name + "_potential.py"
        file = open(output_file_name, "w")
        file.write("# grid potential energy surface from MakePES" + "\n")
        file.write("from collections import defaultdict" + "\n")
        file.write("k_orig = defaultdict(float)" + "\n")
        for key, val in k_orig.items():
            file.write(
                "k_orig["
                + str(key)
                + "]"
                + " " * (17 - len(str(key)))
                + " = "
                + "{:>21.12e}".format(val)
                + "\n"
            )
        file.close()
        print(output_file_name + " is created.")


def main():
    import mop_hamiltonian

    mu = defaultdict(lambda: [0.0, 0.0, 0.0])
    # k_orig = defaultdict(float)
    k_orig = mop_hamiltonian.k_orig

    directory_name = sys.argv[1]
    files = glob.glob("./{}/*".format(directory_name))
    for file_name in files:
        # print(file_name)

        # file_name = sys.argv[1]
        if file_name[-6:] == "eq.pot":
            continue

        elif file_name[-9:] == "eq.dipole":
            file = open(file_name, "r")
            line = file.readlines()[-1]
            words = line.split()
            mu[()] = [
                float(words[0]) * debye2au,
                float(words[1]) * debye2au,
                float(words[2]) * debye2au,
            ]

        elif file_name[-7:] == ".dipole" or file_name[-4:] == ".pot":
            file = open(file_name, "r")
            for i, line in enumerate(file):
                words = line.split()
                if i == 3:
                    dipole = True if words[-1] == "Z" else False
                    if dipole:
                        nMR = len(words) - 4
                        ijk_name = words[1:-3]
                        dipo = [[], [], []]
                    else:
                        nMR = len(words) - 2
                        ijk_name = words[1:-1]
                        ene = []
                    # print(*ijk_name)
                    # print('Mode Representation =', nMR)
                    q_ijk = [[] for _ in range(nMR)]
                if i >= 4:
                    if (
                        abs(sum([float(words[k]) for k in range(nMR)]))
                        > 1.0e-15
                    ):
                        """ remove minimal point """
                        for index in range(nMR):
                            q_ijk[index].append(float(words[index]))

                        if dipole:
                            for xyz in range(3):
                                dipo[xyz].append(
                                    float(words[-3 + xyz]) * debye2au
                                )
                        else:
                            ene.append(float(words[-1]))
            for p in range(nMR):
                q_ijk[p] = np.array(q_ijk[p])
            if dipole:
                for xyz in range(3):
                    dipo[xyz] = np.array(dipo[xyz])
            else:
                ene = np.array(ene)

            ijk_pairs = [(1,) * nMR]
            if dipole:
                make_ijk((1,) * nMR, 0, ijk_pairs, nMR, 5)
            else:
                make_ijk((1,) * nMR, 0, ijk_pairs, nMR, 4)
            # print(ijk_pairs)

            # ハイパーパラメータを初期化
            param = [0 for _ in range(len(ijk_pairs))]

            if nMR == 1:
                fiting_func = fiting_func1MR
            elif nMR == 2:
                fiting_func = fiting_func2MR
            elif nMR == 3:
                fiting_func = fiting_func3MR
            elif nMR == 4:
                fiting_func = fiting_func4MR
            if dipole:
                optimised_param_x = optimize.leastsq(
                    fiting_func, param, args=(*q_ijk, dipo[0], ijk_pairs)
                )
                optimised_param_y = optimize.leastsq(
                    fiting_func, param, args=(*q_ijk, dipo[1], ijk_pairs)
                )
                optimised_param_z = optimize.leastsq(
                    fiting_func, param, args=(*q_ijk, dipo[2], ijk_pairs)
                )
            else:
                optimised_param = optimize.leastsq(
                    fiting_func, param, args=(*q_ijk, ene, ijk_pairs)
                )

            ijk = [int(ijk_name[j][1:]) for j in range(nMR)]
            if dipole:
                for k in range(len(ijk_pairs)):
                    key = ()
                    val_x = optimised_param_x[0][k]
                    val_y = optimised_param_y[0][k]
                    val_z = optimised_param_z[0][k]
                    for j in range(nMR):
                        key += (ijk[j],) * ijk_pairs[k][j]
                        val_x *= factorial(ijk_pairs[k][j])
                        val_y *= factorial(ijk_pairs[k][j])
                        val_z *= factorial(ijk_pairs[k][j])
                    mu[key] = [val_x, val_y, val_z]
            else:
                for k in range(len(ijk_pairs)):
                    key = ()
                    val = optimised_param[0][k]
                    for j in range(nMR):
                        key += (ijk[j],) * ijk_pairs[k][j]
                        val *= factorial(ijk_pairs[k][j])
                    k_orig[key] = val
            # if file_name[-12:] == '/q5q4.dipole':
            #    debug_plot(2,q_ijk,optimised_param_x,dipo[0],ijk_pairs,ijk_name)
            #    debug_plot(2,q_ijk,optimised_param_y,dipo[1],ijk_pairs,ijk_name)
            #    debug_plot(2,q_ijk,optimised_param_z,dipo[2],ijk_pairs,ijk_name)
            if not dipole and nMR == 2:
                debug_plot(2, q_ijk, optimised_param, ene, ijk_pairs, ijk_name)

    file_write(k_orig, mu, "test")
    # print(mu)
    # print(k_orig)


if __name__ == "__main__":
    main()
