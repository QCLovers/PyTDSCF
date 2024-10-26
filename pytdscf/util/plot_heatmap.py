"""Plot potential Heatmap from PyTDSCF PES (wat3 example)"""

import itertools
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    import pytdscf.potentials.wat3_potential as pot

    active = [13, 14, 15, 16, 17, 18, 19, 20, 21]
    mode = len(active)
    mat = np.zeros((mode, mode))

    k_orig = pot.k_orig

    dic = {}
    for k in range(mode):
        dic[active[k]] = k
    # order = [0,6,12, 1,7,13, 2,8,14, 3,9,15, 4,10,16, 5,11,17]
    order = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    SCALING = [1 for _ in range(mode)]
    for k in range(mode):
        SCALING[k] = sqrt(k_orig[(active[k], active[k])] / 2)

    scale = True

    cut_off = -1.0e-09

    for key, val in k_orig.items():
        if abs(val) < cut_off:
            continue

        if scale:
            for k in key:
                val /= SCALING[dic[k]]
            # C = Counter(key)
            # for v in C.values():
            #    val /= factorial(v)

        if len(set(key)) == 1:
            ind = order[dic[key[0]]]
            mat[(ind, ind)] += abs(val)

        for pair in itertools.permutations(list(set(key)), 2):
            ind1 = order[dic[pair[0]]]
            ind2 = order[dic[pair[1]]]
            mat[(ind1, ind2)] += abs(val)

    plt.figure()
    # plt.title('Hamiltonian Coupling Term Sum (log10 scale)')
    plt.ylabel("site $Q_i$")
    plt.xlabel("site $Q_j$")
    plt.imshow(
        np.log10(mat), interpolation="nearest", cmap="jet", vmax=2.0, vmin=-2.0
    )
    plt.colorbar()
    plt.show()
