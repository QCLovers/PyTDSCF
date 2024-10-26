"""Gaussian anharmonic dipole output to PyTDSCF dipole (mu_orig) converter
- How to use

.. code-block:: bash

   $ python3 gout2dipole.py *.log N_FRQS

``N_FRQS = (number of atom)*3 -6`` (if molecule is straight, -6 -> -5)

- Gaussian input file needs "freq=(Anharm, noRaman) Symmetry=None".
"""

import sys
from collections import defaultdict
from math import sqrt

if __name__ == "__main__":
    file = open(sys.argv[1], "r")
    output_file_name = "gaussian_anharm_dipole.py"
    N_FRQS = int(sys.argv[2])

    dipole = defaultdict(float)

    flag1 = True
    flag2 = False

    ideal = sqrt(1 / 1822.8889)  # amu2emu

    Q_i_change = [k for k in range(N_FRQS, 0, -1)]  # 1-index
    ok = input(
        "Do you want to swap vibration modes?(must be consistent to PES)[y/n]\n"
    )
    if ok == "y":
        order = list(
            map(
                int,
                input(
                    'e.g.: if you swap mode 1 with 3 , input "3 2 1 4 5 6" \n'
                ).split(),
            )
        )
        if len(set(order)) != max(order):
            raise ValueError(f"input error: {len(set(order))} != {max(order)}")
        else:
            Q_i_change = order
            print(Q_i_change)

    cutoff = -1.0e-16
    for i, line in enumerate(file):
        if flag1:
            if line == " Unit of the property: Debye" + "\n":
                start_line = i + 4
                flag2 = True
                flag1 = False
                print("Gaussian output file from  line", i)
        if flag2:
            print(line.rstrip())
            if line == "\n":
                flag2 = False
                break
            elif i == start_line:
                words = line.split()
                z = float(words[-1].replace("D", "e"))
                y = float(words[-2].replace("D", "e"))
                x = float(words[-3].replace("D", "e"))
                dipole[tuple()] = [x, y, z]
            elif i > start_line:
                words = line.split()
                ijk = words[2:-4]
                z = float(words[-1].replace("D", "e")) * pow(ideal, len(ijk))
                y = float(words[-2].replace("D", "e")) * pow(ideal, len(ijk))
                x = float(words[-3].replace("D", "e")) * pow(ideal, len(ijk))
                if max(abs(x), abs(y), abs(z)) < cutoff:
                    continue
                for k in range(len(ijk)):
                    ijk[k] = Q_i_change[int(ijk[k]) - 1]
                ijk.sort()
                ijk = tuple(ijk)
                dipole[ijk] = [x, y, z]
    file.close()

    # write
    file = open(output_file_name, "w")
    file.write("from collections import defaultdict" + "\n")
    file.write("mu = defaultdict(float)" + "\n")
    for key, val in dipole.items():
        file.write(
            "mu["
            + str(key)
            + "] = ["
            + "{:.5e}".format(val[0])
            + ", "
            + "{:.5e}".format(val[1])
            + ", "
            + "{:.5e}".format(val[2])
            + "]\n"
        )
    file.close()
    print("success!", output_file_name, "created.")
