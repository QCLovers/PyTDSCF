"""PyTDSCF polynomial operator(kroig) to QUANTICS operator(.op) converter
* How to use

.. code-block :: bash

   $ python3 PyTDSCH2mop.py N_FRQS

``N_FRQS = (number of atom)*3 -6`` (if molecule is straight, -6 -> -5)
Qi and Qj can be swaped.
"""

import os
import sys
from collections import Counter
from math import factorial, sqrt

from numpy import dot

if __name__ == "__main__":
    N_FRQS = int(sys.argv[1])

    flag = 0
    if os.path.isfile("butadiene_potential.py"):
        import butadiene_potential

        flag += 1
    if os.path.isfile("butadiene_dipole.py"):
        import butadiene_dipole

        flag += 2

    if flag == 0:
        print("No operator file exist!")
        exit()
    if flag & 1:
        print("butadiene_potential was imported")
        k_orig = butadiene_potential.k_orig
    if flag & 2:
        print("butadiene_diople.py(dipole operator) was imported")
        mu = butadiene_dipole.mu
        print("E-Field [x y z](Hartree Debye-1) :example 1.0 1.0 1.0")
        E = list(map(float, input().split()))

    file = open("MCTDH_prim", "w")
    if flag & 1:
        for p in range(1, N_FRQS + 1):
            key = (p, p)
            val = k_orig[key]
            key = "_".join(map(str, key))
            file.write(
                "    "
                + "w"
                + key
                + " = "
                + str(sqrt(val)).replace("e", "d")
                + "\n"
            )
    file.close()

    cut_off = pow(10, -7)

    file = open("MCTDH_operator", "w")

    file.write("PARAMETER-SECTION" + "\n")
    if flag & 1:
        k_key = []
        for key, val in k_orig.items():
            if abs(val) > cut_off:
                k_key.append(key)
                key = "_".join(map(str, key))
                file.write(
                    "    "
                    + "k"
                    + key
                    + " = "
                    + str(val).replace("e", "d")
                    + "\n"
                )
    if flag & 2:
        mu_key = []
        for key, val_xyz in mu.items():
            val = dot(val_xyz, E)
            if abs(val) > cut_off and len(key) < 3:
                mu_key.append(key)
                key = "_".join(map(str, key))
                file.write(
                    "    "
                    + "mu"
                    + key
                    + " = "
                    + str(val).replace("e", "d")
                    + "\n"
                )

    file.write("end-parameter-section" + "\n")
    file.write("\n")

    if flag & 1:
        file.write("HAMILTONIAN-SECTION" + "\n")
        first_line = "  modes  "
        for v in range(1, N_FRQS + 1):
            if v % 6 == 1 and v != 1:
                first_line += "\n" + "  modes  "
            first_line += "| v" + str(v) + " "
        file.write(first_line + "\n")
        # file.write('-'*(100) + '\n')

        for v in range(1, N_FRQS + 1):
            KE_line = "1.0" + " " * (17) + "|{}  KE  ".format(v)
            file.write(KE_line + "\n")

        for key in k_key:
            cnt = Counter(key)
            pot_line = "k" + "_".join(map(str, key))
            ideal = 1
            for N in cnt.values():
                ideal *= factorial(N)
            if ideal > 1:
                pot_line += "/" + str(ideal) + ".0"
            pot_line += " " * (20 - len(pot_line))
            for v in range(1, N_FRQS + 1):
                if cnt[v] == 0:
                    continue
                    pot_line += "|  1   "
                elif cnt[v] == 1:
                    pot_line += "|{}  q   ".format(v)
                else:
                    pot_line += "|{}  q^{} ".format(v, cnt[v])
            file.write(pot_line + "\n")

        # file.write('-'*(100) + '\n')
        file.write("end-hamiltonian-section" + "\n")
        file.write("\n")

    if flag & 2:
        print("operator name:example Dipole")
        S = input()
        file.write("HAMILTONIAN-SECTION_" + S + "\n")
        file.write("nodiag" + "\n")
        first_line = "  modes  "
        for v in range(1, N_FRQS + 1):
            if v % 6 == 1 and v != 1:
                first_line += "\n" + "  modes  "
            first_line += "| v" + str(v) + " "
        file.write(first_line + "\n")
        # file.write('-'*(100) + '\n')

        for key in mu_key:
            cnt = Counter(key)
            pot_line = "mu" + "_".join(map(str, key))
            ideal = 1
            for N in cnt.values():
                ideal *= factorial(N)
            if ideal > 1:
                pot_line += "/" + str(ideal) + ".0"
            pot_line += " " * (20 - len(pot_line))
            for v in range(1, N_FRQS + 1):
                if cnt[v] == 1:
                    pot_line += "|{}  q   ".format(v)
                elif cnt[v] > 1:
                    pot_line += "|{}  q^{}   ".format(v, cnt[v])
            if key == ():
                pot_line += "|1 1 "
            file.write(pot_line + "\n")

        # file.write('-'*(100) + '\n')
        file.write("end-hamiltonian-section" + "\n")
        file.write("\n")
    file.close()
    print("success!")
