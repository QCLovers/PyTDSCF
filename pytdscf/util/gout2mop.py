"""Gaussian output anharmonic PES to MIDAS, SINDO PES(.mop) converter
- How to use

.. code-block:: bash

   $ python3 gout2mop.py **.log N_FRQS **.mop

``N_FRQS = (number of atom)*3 -6`` (if molecule is straight, -6 -> -5)

Args:
    outputfile : default is ``prop_no_1.mop`` (for MIDAS)

- defaut nMR is 4 in g16.

- When product Qi odd times, sgn can be reversed.

- Qi and Qj can be swaped.

- Gaussian input file needs "freq=(Anharm, HPMode) Symmetry=None iop(4/34=1) iop(7/33=1)".
"""

import linecache
import sys
from collections import defaultdict
from math import sqrt

if __name__ == "__main__":
    file = open(sys.argv[1], "r")
    N_FRQS = int(sys.argv[2])
    nMR = 4
    if len(sys.argv) == 4:
        output_file_neme = sys.argv[3]
    else:
        output_file_neme = "prop_no_1.mop"

    origin = [tuple("0")]
    for _ in range(1, nMR + 1):
        new = []
        for j in range(1, N_FRQS + 1):
            for item in origin:
                newitem = list(item)
                if int(item[-1]) >= j:
                    continue
                else:
                    while len(newitem) < nMR + 1:
                        newitem.append(str(j))
                        new.append(tuple(newitem))
        origin += new
        if origin[0] == tuple("0"):
            origin = origin[1:]

    dic = defaultdict(float)
    for item in origin:
        s = item[1]
        for k in range(2, len(item)):
            s += " " + item[k]
        dic[s] = 0.0

    level = ""
    flag = False
    end = True
    start = False
    first = True
    flag2 = True
    SCL_FRQS = [-1 for _ in range(N_FRQS + 1)]  # 1-index

    for i, line in enumerate(file):
        words = line.split()
        if level == "":
            if len(line) > 1:
                if line[1] == "#":
                    level = words[1]
        if len(words) > 2:
            if words[0] == "I" and words[1] == "J" and words[-1][0] == "K":
                flag = True
                flag2 = False
        if flag:
            if len(line) == 1 and end:
                start = True
                end = False
                continue
            elif len(line) == 1 and start:
                start = False
                end = True
                flag = False
                continue
        if start:
            if first:
                first = False
                modelist = ["Qi     freqency(cm-1)     SCL_FREQ"]
                modechange = [j for j in range(N_FRQS + 1)]
                print("gaussian outputfile in line", i)
                for k in range(i + 1, i + N_FRQS + 1):
                    newitem = (
                        str(k - i)
                        + " " * 4
                        + linecache.getline(sys.argv[1], k).split()[2]
                        + " " * 7
                    )
                    newitem += "{:.10e}".format(
                        float(linecache.getline(sys.argv[1], k).split()[2])
                        * 4.556335253e-06
                    )
                    modelist.append(newitem)
                for i_mode in modelist:
                    print(i_mode)
                print("Do you want to sort?(ascending order)[y/n]")
                ok = input()
                if ok == "y":
                    new_sort = []
                    for k in range(1, N_FRQS + 1):
                        new_sort.append(modelist[k])
                    new_sort.sort(key=lambda x: float(x.split()[1]))
                    modelist = [modelist[0]]
                    for k in range(N_FRQS):
                        item_k = (
                            new_sort[k].split()[1]
                            + " " * 7
                            + new_sort[k].split()[2]
                        )
                        modelist.append(str(k + 1) + " " * 4 + item_k)
                        modechange[k + 1] = int(new_sort[k].split()[0])
                    for i_mode in modelist:
                        print(i_mode)
                    print("change:", modechange[1:])

                print("Do you want to swap vibration modes?[y/n]")
                ok = input()
                while ok == "y":
                    print(
                        'example: if you want to swap modeQ1 and modeQ2, return "1 2".'
                    )
                    n, m = map(int, input().split())
                    if min(n, m) < 1 or max(n, m) > N_FRQS:
                        print("n,m must be in [1, N_FRQS]")
                        continue
                    modechange[n], modechange[m] = modechange[m], modechange[n]
                    item_n = (
                        modelist[n].split()[1]
                        + " " * 7
                        + modelist[n].split()[2]
                    )
                    item_m = (
                        modelist[m].split()[1]
                        + " " * 7
                        + modelist[m].split()[2]
                    )
                    modelist[n] = str(n) + " " * 4 + item_m
                    modelist[m] = str(m) + " " * 4 + item_n
                    for i_mode in modelist:
                        print(i_mode)
                    print("change:", modechange[1:])
                    print("Do you continue to swap vibration modes?[y/n]")
                    ok = input()

            index = words[:-3]
            for k in range(len(index)):
                index[k] = str(modechange[int(index[k])])
            index.sort(key=lambda x: int(x))
            s = index[0]
            for k in range(1, len(index)):
                s += " " + index[k]
            dic[s] = float(words[-1])
            if len(index) == 2 and len(set(index)) == 1:
                SCL_FRQS[int(index[0])] = float(words[-3]) * 4.556335253e-06

    if flag2:
        print("No Cij found in Gaussian output file.")
        exit()
    if -1 in SCL_FRQS[1:]:
        print("Failed to determine scaling frequencies.")
        exit()

    coeff = []
    ideal = sqrt(1 / 1822.8889)
    FACTORIAL = [1, 1, 2, 6, 24, 120, 720]  # FACTRIAL[n] = n!
    for item in dic.items():
        combi = list(item[0].split())
        c = (ideal ** len(combi)) * float(item[1])
        for item in set(combi):
            c /= FACTORIAL[combi.count(item)]
        for k in range(len(combi)):
            c /= sqrt(SCL_FRQS[int(combi[k])])
        new = [c] + [int(combi[k]) for k in range(len(combi))]
        coeff.append(new)

    file.close()

    # write on mop
    file = open(output_file_neme, "w")
    file.write("SCALING FREQUENCIES " + "N_FRQS=" + str(N_FRQS) + "\n")

    for k in range(1, N_FRQS + 1):
        file.write("{:.22e}".format(SCL_FRQS[k]) + "\n")  # 1-index
    file.write("DALTON_FOR_MIDAS" + " " * 2 + level + "\n")

    cutoff = -1.0e-16
    for item in coeff:
        if cutoff > abs(item[0]):
            continue
        file.write("{:>29.22e}".format(item[0]))
        for k in range(1, len(item)):
            file.write("{:>5}".format(item[k]))
        file.write("\n")
