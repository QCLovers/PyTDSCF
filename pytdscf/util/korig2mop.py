"""
PyTDSCF polynomial operator(k_orig) to SINDO, MIDAS PES (.mop) converter
"""

from collections import Counter
from math import factorial, sqrt

if __name__ == "__main__":
    import test_potential

    k_orig = test_potential.k_orig
    nmode = 27
    level = "B3LYP/cc-pvdz"

    SCL_FRQS = [sqrt(k_orig[(k + 1, k + 1)]) for k in range(nmode)]
    for k in range(nmode):
        if abs(SCL_FRQS[k]) < 1.0e-20:
            SCL_FRQS[k] = 1.0

    # write on mop
    output_file_neme = "prop_no1.mop_leastsquare"
    file = open(output_file_neme, "w")
    file.write("SCALING FREQUENCIES " + "N_FRQS=" + str(nmode) + "\n")

    for k in range(nmode):
        file.write("{:.22e}".format(SCL_FRQS[k]) + "\n")  # 1-index
    file.write("DALTON_FOR_MIDAS  " + level + "\n")

    cutoff = 1.0e-20
    for key, val in k_orig.items():
        if cutoff > abs(val):
            continue
        C = Counter(key)
        for order in C.values():
            val /= factorial(order)
        for k in key:
            val /= sqrt(SCL_FRQS[k - 1])
        file.write("{:>29.22e}".format(val))
        for k in key:
            file.write("{:>5}".format(k))
        file.write("\n")
