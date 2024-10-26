"""SINDO .minfo to Gaussian output converter
* How to use

.. code-block:: bash

   $ python3 minf2gout.py ---.log N_FRQS

``N_FRQS = (number of atom)*3 -6``
(if molecule is straight, -6 -> -5)
Qi and Qj can be swapped.
"""

import sys
from math import ceil

import numpy as np

if __name__ == "__main__":
    file = open(sys.argv[1], "r")

    start_atom = False
    start_vib_freq = False
    start_vib_vec = False
    start_vib_each = False
    atom_list = []
    freq_list = []
    vibvec_list = []
    for i, line in enumerate(file):
        if line == "[ Atomic Data ]\n":
            start_atom = i
            end_atom = False
            continue
        elif start_atom:
            line = line.replace(",", "").replace("E", "e")
            words = line.split()
            if i == start_atom + 1:
                N_atom = int(words[0])
                continue
            elif i <= start_atom + N_atom + 1:
                atom_list.append([words[0], int(words[1])])
                continue
            else:
                start_atom = False
                continue
        elif line == "Vibrational Frequency\n":
            start_vib_freq = i
            continue
        elif start_vib_freq:
            line = line.replace(",", "").replace("E", "e")
            words = line.split()
            if i == start_vib_freq + 1:
                N_FRQS = int(words[0])
                continue
            elif len(freq_list) < N_FRQS:
                for f in words:
                    freq_list.append(float(f))
            if len(freq_list) == N_FRQS:
                start_vib_freq = False
                continue
        elif line == "Vibrational vector\n":
            start_vib_vec = i
            continue
        elif start_vib_vec:
            line = line.replace(",", "").replace("E", "e")
            words = line.split()
            if words[0] == "Mode":
                continue
            elif not start_vib_each:
                start_vib_each = True
                vec_dum = []
                continue
            if start_vib_each:
                if len(vec_dum) < N_atom * 3:
                    for f in words:
                        vec_dum.append(float(f))
                if len(vec_dum) == N_atom * 3:
                    vibvec_list.append(np.array(vec_dum).reshape(N_atom, 3))
                    start_vib_each = False
                if len(vibvec_list) == N_FRQS:
                    start_vib_vec = False
    file.close()

    file = open("displacement_oc.txt", "w")
    N = 3
    for k in range(ceil(N_FRQS / N)):
        col = ""
        for i in range(7):
            for r in range(N):
                if N * k + r == N_FRQS:
                    break
                if i == 0:
                    col += "{:23d}".format(N * k + r + 1)
                elif i == 1:
                    col += "{: >23}".format("A")
                elif i == 2:
                    if r == 0:
                        col += " Frequencies --"
                        col += "{:12.4f}".format(freq_list[N * k + r])
                    else:
                        col += "{:23.4f}".format(freq_list[N * k + r])
                elif i == 3:
                    if r == 0:
                        col += " Red. masses --"
                elif i == 4:
                    if r == 0:
                        col += " Frc consts  --"
                elif i == 5:
                    if r == 0:
                        col += " IR Inten    --"
                elif i == 6:
                    if r == 0:
                        col += "  Atom  AN      X      Y      Z"
                    else:
                        col += "        X      Y      Z"
            col += "\n"
        for i in range(N_atom):
            col += "{:6d}".format(i + 1)
            col += "{:4d}".format(atom_list[i][1])
            for r in range(N):
                col += " " * 2
                if N * k + r == N_FRQS:
                    break
                for xyz in range(3):
                    col += "{:7.2f}".format(vibvec_list[N * k + r][i][xyz])
            col += "\n"
        file.write(col)
    #    file.write('k_orig[('+ item[0] +')] = ' + str(item[1]) + '\n')
    file.close()
    print("success!")
