"""MIDAS, SINDO PES(.mop) to PyTDSCF polynomial PES(k_orig) converter
* How to use

.. code-block:: bash

   $ python3 mop2PyTDSCH_hamiltonian.py input.mop N_FRQS

``N_FRQS = (number of atom)*3 -6`` (if molecule is straight, -6 -> -5)
Qi and Qj can be swapped.
"""

import sys
from collections import Counter, defaultdict
from math import factorial, sqrt

if __name__ == "__main__":
    file = open(sys.argv[1], "r")
    N_FRQS = int(sys.argv[2])

    SCL = [1 for _ in range(N_FRQS + 1)]  # 1-index
    dic_K = defaultdict(float)
    cut_off = -1.0e-12

    for i, line in enumerate(file):
        words = line.split()
        if i == 0:
            continue
        elif i <= N_FRQS:
            SCL[i] = sqrt(float(words[-1]))
        elif i == N_FRQS + 1:
            continue
        else:
            index = words[1:]
            cnt_index = Counter(index)
            str_index = ", ".join(index)
            if str_index.count(",") == 0:
                str_index += ","
            coeff = float(words[0])
            for val in cnt_index.values():
                coeff *= factorial(val)
            for item in index:
                k = int(item)
                coeff *= SCL[k]
            if cut_off < abs(coeff):
                dic_K[str_index] = coeff

    file.close()

    coeff_lis = []
    for item in dic_K.items():
        coeff_lis.append([item[0], item[1], len(item[0])])

    coeff_lis.sort(key=lambda x: x[2])

    file = open("mop_hamiltonian.py", "w")
    file.write("from collections import defaultdict" + "\n")
    file.write("k_orig = defaultdict(float)" + "\n")
    for item in coeff_lis:
        file.write("k_orig[(" + item[0] + ")] = " + str(item[1]) + "\n")
    file.close()
    print("success!")
