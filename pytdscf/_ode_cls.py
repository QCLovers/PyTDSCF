"""Ordinary Differential Equation (ODE) solver module"""

from pytdscf._const_cls import const

a11 = 1 / 5
a21 = 3 / 40
a22 = 9 / 40
a31 = 44 / 45
a32 = -56 / 15
a33 = 32 / 9
a41 = 19372 / 6561
a42 = -25360 / 2187
a43 = 64448 / 6561
a44 = -212 / 729
a51 = 9017 / 3168
a52 = -355 / 33
a53 = 46732 / 5247
a54 = 49 / 176
a55 = -5103 / 18656
a61 = 35 / 384
a62 = 0
a63 = 500 / 1113
a64 = 125 / 192
a65 = -2187 / 6784
a66 = 11 / 84
d1 = 71 / 57600
d3 = -71 / 16695
d4 = 71 / 1920
d5 = -17253 / 339200
d6 = 22 / 525
d7 = -1 / 40


# NotUse->b1 = 5179/57600; b3 = 7571/16695; b4 = 393/640; b5 =-92097/339200; b6 = 187/2100; b7 = 1/40


def algorithm_RK4(h, coef, args):
    # print("old coef[0]\n", id(coef), coef[0])
    k0 = coef
    k1 = h * (k0).derivatives(*args)
    k2 = h * (k0 + 0.5 * k1).derivatives(*args)
    k3 = h * (k0 + 0.5 * k2).derivatives(*args)
    k4 = h * (k0 + k3).derivatives(*args)
    dotPsi = 1 / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return dotPsi


def algorithm_DPRK45(h, coef, args):
    """"""
    k0 = coef
    k1 = h * (k0).derivatives(*args)

    k1 = h * (k0).derivatives(*args)
    k2 = h * (k0 + a11 * k1).derivatives(*args)
    k3 = h * (k0 + a21 * k1 + a22 * k2).derivatives(*args)
    k4 = h * (k0 + a31 * k1 + a32 * k2 + a33 * k3).derivatives(*args)
    k5 = h * (k0 + a41 * k1 + a42 * k2 + a43 * k3 + a44 * k4).derivatives(*args)
    k6 = h * (
        k0 + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4 + a55 * k5
    ).derivatives(*args)

    dotPsi_4th = (
        a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5 + a66 * k6
    )  # y_4th
    y_4th = k0 + dotPsi_4th

    k7 = h * (y_4th).derivatives(*args)
    # ->    dotPsi_5th = b1*k1 +b3*k3 +b4*k4 +b5*k5 +b6*k6 +b7*k7
    # ->    y_5th = k0 +dotPsi_5th

    err_45 = d1 * k1 + d3 * k3 + d4 * k4 + d5 * k5 + d6 * k6 + d7 * k7
    err_45 = max(err_45.norm_absmax_rk5(), 1e-10)

    stepsize_allowed = pow(const.tol_RK45 * h / err_45, 1 / 4) * h * 0.8
    #    stepsize_allowed  = pow(const.tol_RK45/err_45, 1/4) * h * 0.8

    return dotPsi_4th, stepsize_allowed
