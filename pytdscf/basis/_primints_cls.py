"""
The integral between Q^n or P^n and HO primitive basis χ.
The evaluation of integrals takes a while,
so you may probably use the pybind option with `_primints.cpp`.
"""

import copy
import itertools
import math

import numpy as np
import scipy.special
from discvar import HarmonicOscillator as HO

import pytdscf
from pytdscf import units
from pytdscf.basis import HarmonicOscillator as _HO
from pytdscf.basis import PrimBas_HO


def ovi_HO_FBR_matrix(pbas_bra, pbas_ket) -> np.ndarray:
    """ Get HO overlap integral matrix H_ij = <HO_i(ω)|HO_j(ω')>

    See also J.-L. Chang (2005) J.Mol.Spec. 232, 102-104 \
            https://doi.org/10.1016/j.jms.2005.03.004

    Args:
        pbas_bra (PrimBas_HO or HarmonicOscillator) : bra of HO primitive basis.
        pbas_ket (PrimBas_HO or HarmonicOscillator) : ket of HO primitive basis.

    Returns:
        numpy.ndarray : The overlap integral (density) matrix of HO primitive basis.\
                The shape is (``pbas_bra.nprim``, ``pbas_ket.nprim``).

    """
    assert isinstance(pbas_bra, (HO, _HO, PrimBas_HO))
    assert isinstance(pbas_ket, (HO, _HO, PrimBas_HO))
    nprim_bra = pbas_bra.nprim
    nprim_ket = pbas_ket.nprim
    a0 = (pbas_bra.freq_cm1 / units.au_in_cm1) / 1.0  # ω / hbar
    a1 = (pbas_ket.freq_cm1 / units.au_in_cm1) / 1.0  # ω / hbar
    x0 = pbas_bra.origin / math.sqrt(a0)  # ζ
    x1 = pbas_ket.origin / math.sqrt(a1)  # ζ

    if a0 == a1 and x0 == x1:
        return np.eye(nprim_bra, nprim_ket)

    d = x1 - x0
    b0 = -a1 * math.sqrt(a0) * d / (a0 + a1)
    b1 = +a0 * math.sqrt(a1) * d / (a0 + a1)

    """ pre-calculations """
    _comb_herm0 = []
    for v0 in range(nprim_bra):
        _comb_herm0.append(
            [
                scipy.special.comb(v0, k0)
                * scipy.special.eval_hermite(v0 - k0, b0)
                for k0 in range(v0 + 1)
            ]
        )

    _comb_herm1 = []
    for v1 in range(nprim_ket):
        _comb_herm1.append(
            [
                scipy.special.comb(v1, k1)
                * scipy.special.eval_hermite(v1 - k1, b1)
                for k1 in range(v1 + 1)
            ]
        )

    _fac_pair = np.zeros((nprim_bra, nprim_ket))
    for k0 in range(nprim_bra):
        for k1 in range(nprim_ket):
            if (k0 + k1) % 2 == 0:
                K = (k0 + k1) // 2
                _fac_pair[k0, k1] = (
                    pow(2 * math.sqrt(a0), k0)
                    * pow(2 * math.sqrt(a1), k1)
                    * float(scipy.special.factorial2(2 * K - 1))
                    / pow(a0 + a1, K)
                )

    """ multiplication """
    S = (a0 * a1 * d * d) / (a0 + a1)
    A = 2.0 * math.sqrt(a0 * a1) / (a0 + a1)
    Aexp_S = A * math.exp(-S)

    ints = np.zeros((nprim_bra, nprim_ket))
    for v0, v1 in itertools.product(range(nprim_bra), range(nprim_ket)):
        C = math.sqrt(
            Aexp_S / (pow(2, v0 + v1) * math.factorial(v0) * math.factorial(v1))
        )
        val = 0.0
        for k0 in range(v0 + 1):
            for k1 in range(v1 + 1):
                if (k0 + k1) % 2 == 0:
                    val += (
                        _comb_herm0[v0][k0]
                        * _comb_herm1[v1][k1]
                        * _fac_pair[k0, k1]
                    )

        ints[v0, v1] = C * val

    return ints


def ovi_HO_DVR_matrix(pbas_bra: HO, pbas_ket: HO) -> np.ndarray:
    """Get HO overlap integral matrix H_ij = <HO_i(ω)|HO_j(ω')>

    Args:
        pbas_bra (PrimBas_HO or HarmonicOscillator) : bra of HO primitive basis.
        pbas_ket (PrimBas_HO or HarmonicOscillator) : ket of HO primitive basis.

    Returns:
        numpy.ndarray : The overlap integral (density) matrix of HO primitive basis.

    """

    assert isinstance(pbas_bra, HO) or isinstance(pbas_bra, _HO)
    assert isinstance(pbas_ket, HO) or isinstance(pbas_ket, _HO)

    # |χ> = U^†|φ>
    # <χ|O|χ> = <φ|UOU^†|φ> = U<φ|O|φ>U^†
    return (
        np.conjugate(pbas_bra.get_unitary().T)
        @ ovi_HO_FBR_matrix(pbas_bra, pbas_ket)
        @ pbas_ket.get_unitary()
    )


def ovi_HO_FBR(
    v0: int, v1: int, pbas_bra: PrimBas_HO, pbas_ket: PrimBas_HO
) -> float:
    """ Get HO overlap integral matrix element (scalar !!) <HO_v0(ω)|HO_v1(ω')>

    See also J.-L. Chang (2005) J.Mol.Spec. 232, 102-104 \
            https://doi.org/10.1016/j.jms.2005.03.004
    This Python code is too slow. You can use pybind option with `primints_in_cpp.cpp`.
    (to be implemented.)

    Args:
        v0 (int) : The order of bra HO primitive.
        v1 (int) : The order of ket HO primitive.
        pbas_bra (PrimBas_HO) : bra of HO primitive basis.
        pbas_ket (PrimBas_HO) : ket of HO primitive basis.

    Returns:
        float : The overlap integral (density) matrix element of HO primitive basis.

    """
    try:
        from pytdscf.basis._primints import ovi_HO_FBR_cpp

        return ovi_HO_FBR_cpp(
            v0,
            v1,
            pbas_bra.freq_cm1,
            pbas_ket.freq_cm1,
            pbas_bra.origin,
            pbas_ket.origin,
        )
    except ModuleNotFoundError:
        print(
            "You cannot use the fast C++ version. If you want to use it, please reinstall with scikit-build."
        )
        pass
    a0 = (pbas_bra.freq_cm1 / units.au_in_cm1) / 1.0  # ω / hbar
    a1 = (pbas_ket.freq_cm1 / units.au_in_cm1) / 1.0  # ω' / hbar
    x0 = pbas_bra.origin / math.sqrt(a0)  # ζ
    x1 = pbas_ket.origin / math.sqrt(a1)  # ζ'

    d = x1 - x0
    b0 = -a1 * math.sqrt(a0) * d / (a0 + a1)
    b1 = +a0 * math.sqrt(a1) * d / (a0 + a1)

    val = 0.0
    for k0 in range(v0 + 1):
        for k1 in range(v1 + 1):
            if (k0 + k1) % 2 == 0:
                K = (k0 + k1) // 2
                val += (
                    scipy.special.comb(v0, k0)
                    * scipy.special.comb(v1, k1)
                    * scipy.special.eval_hermite(v0 - k0, b0)
                    * scipy.special.eval_hermite(v1 - k1, b1)
                    * pow(2 * math.sqrt(a0), k0)
                    * pow(2 * math.sqrt(a1), k1)
                    * float(scipy.special.factorial2(2 * K - 1))
                    / pow(a0 + a1, K)
                )

    S = (a0 * a1 * d * d) / (a0 + a1)
    A = 2.0 * math.sqrt(a0 * a1) / (a0 + a1)
    C = math.sqrt(
        (A * math.exp(-S))
        / (pow(2, v0 + v1) * math.factorial(v0) * math.factorial(v1))
    )

    return C * val


def poly_HO_FBR(v0: int, v1: int, pbas_bra, pbas_ket, order: int) -> float:
    """ Get HO q^n integral matrix element (scalar !!) <HO_v0(ω)|q^n|HO_v1(ω')>

    See also J.-L. Chang (2005) J.Mol.Spec. 232, 102-104 \
            https://doi.org/10.1016/j.jms.2005.03.004

    Args:
        v0 (int) : The order of bra HO primitive.
        v1 (int) : The order of ket HO primitive.
        pbas_bra (PrimBas_HO) : bra of HO primitive basis.
        pbas_ket (PrimBas_HO) : ket of HO primitive basis.
        order (int) : The order of operator q^n.

    Returns:
        float : The integral (density) of HO primitive with q^n <v0|q^n|v1>.

    To Do:
        Implement return matrix (like ``ovi_HO_FBR_matrix``).

    """
    try:
        from pytdscf.basis._primints import poly_HO_FBR_cpp

        return poly_HO_FBR_cpp(
            v0,
            v1,
            pbas_bra.freq_cm1,
            pbas_ket.freq_cm1,
            pbas_bra.origin,
            pbas_ket.origin,
            order,
        )
    except ModuleNotFoundError:
        print(
            "You cannot use the fast C++ version. If you want to use it, please reinstall with scikit-build."
        )
        pass
    a0 = (pbas_bra.freq_cm1 / units.au_in_cm1) / 1.0  # ω / hbar
    a1 = (pbas_ket.freq_cm1 / units.au_in_cm1) / 1.0  # ω' / hbar
    x0 = pbas_bra.origin / math.sqrt(a0)  # ζ
    x1 = pbas_ket.origin / math.sqrt(a1)  # ζ'

    assert x0 == 0.0, "bra HO should be 0-origin for <v|x^n|v'>"

    d = x1 - x0
    b0 = -a1 * math.sqrt(a0) * d / (a0 + a1)
    b1 = +a0 * math.sqrt(a1) * d / (a0 + a1)

    r = -a1 * d / (a0 + a1)

    val = 0.0
    for k2 in range(order + 1):
        for k0 in range(v0 + 1):
            for k1 in range(v1 + 1):
                if (k0 + k1 + k2) % 2 == 0:
                    K = (k0 + k1 + k2) // 2
                    val += (
                        scipy.special.comb(v0, k0)
                        * scipy.special.comb(v1, k1)
                        * scipy.special.comb(order, k2)
                        * scipy.special.eval_hermite(v0 - k0, b0)
                        * scipy.special.eval_hermite(v1 - k1, b1)
                        * pow(r, order - k2)
                        * pow(2 * math.sqrt(a0), k0)
                        * pow(2 * math.sqrt(a1), k1)
                        * float(scipy.special.factorial2(2 * K - 1))
                        / pow(a0 + a1, K)
                    )

    S = (a0 * a1 * d * d) / (a0 + a1)
    A = 2.0 * math.sqrt(a0 * a1) / (a0 + a1)
    C = math.sqrt(
        (A * math.exp(-S))
        / (pow(2, v0 + v1) * math.factorial(v0) * math.factorial(v1))
    )

    return C * val


class PrimInts:
    """ The primitive integrals class.

    The format is similar to ``dict``. \
        In grid-based DVR, only 'ovlp' and 'auto' will be set.

    Args:
        model (model_cls.Model) : Your input information, such as primitive, SPF, operator etc.

    Examples:
        >>> print(type(model))
        <class 'model_cls.Model'>
        >>> ints_prim = PrimInts(model)
        >>> print(ints_prim.keys())
        dict_keys(['ovlp', 'd^2', 'q^1', 'q^2', 'q^3', 'q^4', 'onesite', (0, 0)])
        >>> print(ints_prim['q^2'].keys())
        dict_keys([(0, 0)])
        >>> bra_state, ket_state, mode = 0, 0, 0
        >>> shape = ints_prim['q^2'][(bra_state,ket_state)][mode].shape
        >>> print(shape)
        (4, 4)
        >>> nprim = model.basinfo.prim_info[bra_state][mode].nprim
        >>> assert nprim == shape[bra_state]
        True

    To Do:
        * Support arbitary polynomial order.
    """

    def __init__(self, model):
        self.set_ovi(model.basinfo)
        assert isinstance(model.basinfo, pytdscf.BasInfo)
        if model.basinfo.need_primints:
            self.set_poly_diag(model.basinfo)
            self.set_poly_nondiag(model.basinfo)
            for matOp in model.observables.values():
                self.set_onesite(model.basinfo, matOp)
            self.set_onesite(model.basinfo, model.hamiltonian)
        # NewIndx-BGN
        for state_pair in itertools.product(
            range(model.basinfo.get_nstate()), repeat=2
        ):
            ints_prim_state_pair = {}
            for op_key in self.op_keys():
                if state_pair in self[op_key]:
                    ints_prim_state_pair[op_key] = self[op_key][state_pair]
            self[state_pair] = ints_prim_state_pair

    # NewIndx-END
    def __len__(self):
        return len(self.__dict__)

    def __repr__(self):
        return repr(self.__dict__)

    def __str__(self):
        return str(self.__dict__)

    def __iter__(self):
        return self.__dict__.iteritems()

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def keys(self):
        """Get operator keys in `PrimInts`

        See also operator key definition in ``hamiltonian_cls.TermProductForm.op_keys``.

        Returns:
            dict_keys :

        Examples:
            >>> print(type(model))
            <class 'model_cls.Model'>
            >>> ints_prim = PrimInts(model)
            >>> print(ints_prim.keys())
            dict_keys(['ovlp', 'd^2', 'q^1', 'q^2', 'q^3', 'q^4', 'onesite', (0, 0)])

        """
        return self.__dict__.keys()

    def values(self):
        """Get operator values in PrimInts

        Returns:
            dict_values :

        Examples:
            >>> print(type(model))
            <class 'model_cls.Model'>
            >>> ints_prim = PrimInts(model)
            >>> print(ints_prim.values())
            dict_values([{(0, 0): [array([[     1.0000000000000,      0.0000000000000,\
                    0.0000000000000,      0.0000000000000], ........
        """
        return self.__dict__.values()

    def items(self):
        """Get operator items in PrimInts

        Returns:
            dict_items :

        Examples:
            >>> print(type(model))
            <class 'model_cls.Model'>
            >>> ints_prim = PrimInts(model)
            >>> print(ints_prim.items())
            dict_items([('ovlp', {(0, 0): [array([[     1.0000000000000,      0.0000000000000,\
                    0.0000000000000,      0.0000000000000], ........
        """
        return self.__dict__.items()

    #
    #    @classmethod
    def op_keys(self):
        """Get operator keys (not contain overlaps between different states).

        Returns:
            list : operator key list

        Examples:
            >>> print(type(model))
            <class 'model_cls.Model'>
            >>> ints_prim = PrimInts(model)
            >>> print(ints_prim.op_keys())
            ['ovlp', 'd^2', 'q^1', 'q^2', 'q^3', 'q^4', 'onesite']

        """
        _op_keys = []
        for key in self.keys():
            if isinstance(key, str):
                _op_keys.append(key)
        return _op_keys

    def statepair_keys(self):
        """Get keys of overlaps between different electronic states.

        Returns:
            list : operator key list

        Examples:
            >>> print(type(model))
            <class 'model_cls.Model'>
            >>> ints_prim = PrimInts(model)
            >>> print(ints_prim.op_keys())
            [(0,0)]

        """
        _statepair_keys = []
        for key in self.keys():
            if isinstance(key, tuple):
                assert len(key) == 2
                _statepair_keys.append(key)
        return _statepair_keys

    def set_onesite(self, basinfo, matH):
        """Set 'onesite' operator by summing up <v0|q^n|v1>

        Args:
            basinfo (model_cls.BasInfo) : Primitive basis information.
            matH (hamiltonian_cls.Hamiltonian) : The operator containing q^1~q^n.

        """
        if matH.onesite:
            matH_onesite = matH.onesite
            self[matH.onesite_name] = {}
            for istate, jstate in itertools.product(
                range(basinfo.get_nstate()), repeat=2
            ):
                terms = matH_onesite[istate][jstate]
                onesite_dofs = [None for idof in range(basinfo.get_ndof())]
                for term_onesite in terms:
                    coef = term_onesite.coef
                    op_dof = term_onesite.op_dof
                    op_key = term_onesite.op_key
                    if onesite_dofs[op_dof] is not None:
                        onesite_dofs[op_dof] += (
                            coef * self[op_key][(istate, jstate)][op_dof]
                        )
                    else:
                        onesite_dofs[op_dof] = (
                            coef * self[op_key][(istate, jstate)][op_dof]
                        )
                self[matH.onesite_name][(istate, jstate)] = copy.deepcopy(
                    onesite_dofs
                )

    def set_ovi(self, basinfo):
        """ Set overlap Integrals information on ``self['ovlp']``.

        Args:
            basinfo (model_cls.BasInfo) : The primitive and SPF information. \
                    This is also stored in ``model_cls.Model.basinfo``.

        """

        def eval_ovi_dof_matrix(state0, state1):
            ovi_dof = [None for idof in range(basinfo.get_ndof())]
            for idof in range(basinfo.get_ndof()):
                pbas_bra = basinfo.get_primbas(state0, idof)
                pbas_ket = basinfo.get_primbas(state1, idof)
                if (
                    is_same_basis := (type(pbas_bra) is type(pbas_ket))
                ) and type(pbas_bra) is PrimBas_HO:
                    ovi_dof[idof] = ovi_HO_FBR_matrix(pbas_bra, pbas_ket)
                elif is_same_basis and isinstance(pbas_bra, _HO):
                    ovi_dof[idof] = ovi_HO_DVR_matrix(pbas_bra, pbas_ket)
                elif is_same_basis and isinstance(pbas_bra, HO):
                    ovi_dof[idof] = ovi_HO_DVR_matrix(pbas_bra, pbas_ket)
                elif is_same_basis:
                    if len(pbas_bra) != len(pbas_ket):
                        raise ValueError(
                            f"Number of {type(pbas_bra)} DVR basis "
                            + "must be the same between bra and ket"
                        )
                    ovi_dof[idof] = np.identity(len(pbas_bra))
                else:
                    raise NotImplementedError(
                        f"Not implemented overlap integral between \
                            {type(pbas_bra)} and {type(pbas_ket)}"
                    )

            return copy.deepcopy(ovi_dof)

        # -- BODY ------------------------#
        self["ovlp"] = {}
        for istate, jstate in itertools.product(
            range(basinfo.get_nstate()), repeat=2
        ):
            self["ovlp"][(istate, jstate)] = eval_ovi_dof_matrix(istate, jstate)

    def set_poly_nondiag(self, basinfo):
        """ Set non-diagonal (=between different electronic states) polynomial \
                q^n term of primitive integral matrix.

        This is also needed for complementary operator.

        Args:
            basinfo (model_cls.BasInfo) : The primitive and SPF information. \
                    This is also stored in ``model_cls.Model.basinfo``.

        """

        def set_poly_dof(state0, state1, norder):
            poly_dof = [None for idof in range(basinfo.get_ndof())]
            for idof in range(basinfo.get_ndof()):
                pbas_bra = basinfo.get_primbas(state0, idof)
                pbas_ket = basinfo.get_primbas(state1, idof)
                nprim_bra = pbas_bra.nprim
                if pbas_bra.origin != 0.0 and pbas_ket.origin != 0:
                    raise ValueError(
                        "Either bra or ket primitive basis must be centered at 0.0"
                    )
                elif pbas_bra.origin != 0.0:
                    continue
                nprim_ket = pbas_ket.nprim
                dum = np.zeros((nprim_bra, nprim_ket))
                for ij_prim in itertools.product(
                    range(nprim_bra), range(nprim_ket)
                ):
                    v0 = ij_prim[0]
                    v1 = ij_prim[1]
                    dum[v0, v1] = poly_HO_FBR(
                        v0, v1, pbas_bra, pbas_ket, norder
                    )
                poly_dof[idof] = dum.copy()

            return copy.deepcopy(poly_dof)

        # -- BODY ------------------------#

        for n_ord_minus1, op_key in enumerate(["q^1"]):
            # In non-diag case, 'q^1' is sufficient for LVC model
            if op_key not in self.keys():
                self[op_key] = {}
            for istate in range(basinfo.get_nstate()):
                for jstate in range(istate + 1, basinfo.get_nstate()):
                    self[op_key][(istate, jstate)] = set_poly_dof(
                        istate, jstate, n_ord_minus1 + 1
                    )
                    self[op_key][(jstate, istate)] = set_poly_dof(
                        jstate, istate, n_ord_minus1 + 1
                    )
                    for idof in range(len(self[op_key][(istate, jstate)])):
                        if self[op_key][(istate, jstate)][idof] is not None:
                            self[op_key][(jstate, istate)][idof] = self[op_key][
                                (istate, jstate)
                            ][idof].T
                        if self[op_key][(jstate, istate)][idof] is not None:
                            self[op_key][(istate, jstate)][idof] = self[op_key][
                                (jstate, istate)
                            ][idof].T

    def set_ham1(self, basinfo):
        """ Set Hamiltonian integral matrix for primitive basis.

        Args:
            basinfo (model_cls.BasInfo) : The primitive and SPF information. \
                    This is also stored in ``model_cls.Model.basinfo``.

        """

        def set_ham1_dof(state0, state1):
            assert (
                state0 == state1
            ), "Hamiltonian integral matrix must be the same electronic states"
            ham1_dof = [None for idof in range(basinfo.get_ndof())]
            for idof in range(basinfo.get_ndof()):
                pbas_bra = basinfo.get_primbas(state0, idof)
                pbas_ket = basinfo.get_primbas(state1, idof)
                nprim_bra = pbas_bra.nprim
                nprim_ket = pbas_ket.nprim
                dum = np.zeros((nprim_bra, nprim_ket))
                for v0, v1 in itertools.product(
                    range(nprim_bra), range(nprim_ket)
                ):
                    if v0 == v1:
                        dum[v0, v1] = (pbas_bra.freq_cm1 / units.au_in_cm1) * (
                            v0 + 0.5
                        )

                ham1_dof[idof] = dum.copy()

            return ham1_dof

        # -- BODY ------------------------#
        self["ham1"] = {}
        for istate in range(basinfo.get_nstate()):
            self["ham1"][(istate, istate)] = set_ham1_dof(istate, istate)
        # ->self.ham1[(0,0)] = set_ham1_dof(0,0)
        # ->self.ham1[(1,1)] = set_ham1_dof(1,1)

    def set_poly_diag(self, basinfo, kinetic=True):
        """ Set diagonal (=between the same electronic state) polynomial q^n term \
                and kinetic d^2 term of primitive integral matrix.

        This is also needed for complementary operator.

        Args:
            basinfo (model_cls.BasInfo) : The primitive and SPF information. \
                    This is also stored in ``model_cls.Model.basinfo``.
            kinetic (bool) : Include kinetic term or not. Defaults to ``True``.

        """

        def eval_kine_diag(state0):
            """
            (d/dq)^2 = w/2 * (cre*cre + des*des - cre*des - des*cre)
            because
            (-1/2)*(d/dq)^2 = w/2 (-(d/dQ)^2) = w/2 (P^2) = w/2 (i/sqrt(2) (cre-des))^2
                            = w/2 * (-1/2) (cre*cre + des*des - cre*des - des*cre)

            <v0|cre cre|v0-2> = <v0|cre |v0-1>*sqrt(v0-1) = <v0|v0>*sqrt(v0  )*sqrt(v0-1)
            <v0|des des|v0+2> = <v0|des |v0+1>*sqrt(v0+2) = <v0|v0>*sqrt(v0+1)*sqrt(v0+2)
            <v0|des cre|v0  > = <v0|des |v0+1>*sqrt(v0+1) = <v0|v0>*sqrt(v0+1)*sqrt(v0+1)
            <v0|cre des|v0  > = <v0|cre |v0-1>*sqrt(v0  ) = <v0|v0>*sqrt(v0  )*sqrt(v0  )
            """
            np.set_printoptions(
                formatter={"float": "{: 20.13f}".format}, linewidth=200
            )

            kine_dof = [None for idof in range(basinfo.get_ndof())]
            for idof in range(basinfo.get_ndof()):
                pbas = basinfo.get_primbas(state0, idof)
                nprim = pbas.nprim
                dum = np.zeros((nprim, nprim))
                for v0, v1 in itertools.product(range(nprim), range(nprim)):
                    # old diffinition->                    fac = pbas.freq_au / 2 * (-1/2)
                    fac = pbas.freq_au / 2
                    fac *= (
                        math.sqrt(v0) * math.sqrt(v0 - 1)
                        if v1 == v0 - 2
                        else math.sqrt(v0 + 1) * math.sqrt(v0 + 2)
                        if v1 == v0 + 2
                        else -math.sqrt(v0 + 1) * math.sqrt(v0 + 1)
                        - math.sqrt(v0) * math.sqrt(v0)
                        if v1 == v0
                        else 0.0
                    )
                    if fac != 0.0:
                        dum[v0, v1] = fac

                kine_dof[idof] = dum.copy()

            return kine_dof

        def eval_poly_diag(state0, *, norder=2):
            r"""scipy.special.comb(10, 5, exact=True)
            <v0(q-q0)|q^norder|v1(q-q0)> = <v0(q)|(q+q0)^norder|v1(q)>
                                         = \sum_i comb(n,i) <v0(q)|q^i|v1(q)>
                                         = \sum_i comb(n,i) <v0(q)|((cre+des)*sqrt(1/2w))**i|v1(q)>
            """

            def apply_cre_plus_des_to_ket(ket_terms):
                """|ket_terms_out> = (cre+des) |ket_terms>"""
                ket_terms_out = [0.0 for i in range(len(ket_terms) + 1)]
                for v1, coef in enumerate(ket_terms):
                    """ |out> = cre c1|v1> = sqrt(v1+1) c1|v1+1>"""
                    ket_terms_out[v1 + 1] += math.sqrt(v1 + 1) * coef
                    """ |out> = des c1|v1> = sqrt(v1  ) c1|v1-1>"""
                    if v1 > 0:
                        ket_terms_out[v1 - 1] += math.sqrt(v1) * coef
                return ket_terms_out

            poly_dofs = [
                [None for idof in range(basinfo.get_ndof())]
                for i in range(norder + 1)
            ]
            for idof in range(basinfo.get_ndof()):
                pbas = basinfo.get_primbas(state0, idof)
                nprim = pbas.nprim
                """ ints_orig[v0,v1,i] = <v0(q)|q^i|v1(q)>
                                       = <v0(q)|((cre+des) * sqrt(1/2w))**i|v1(q)>
                """
                ints_orig = np.zeros((nprim, nprim, norder + 1))
                for v0, v1 in itertools.product(range(nprim), range(nprim)):
                    ket_terms = [0.0 for i in range(v1 + 1)]
                    ket_terms[v1] = 1.0
                    for i in range(norder + 1):
                        ints_orig[v0, v1, i] = (
                            0.0 if v0 > len(ket_terms) - 1 else ket_terms[v0]
                        )
                        ket_terms = apply_cre_plus_des_to_ket(ket_terms)
                        # map(lambda x: x * math.sqrt(1/(2*pbas.freq_au)), ket_terms)
                        ket_terms = [
                            x * math.sqrt(1 / (2 * pbas.freq_au))
                            for x in ket_terms
                        ]
                r"""<v0(q-q0)|q^n|v1(q-q0)> = <v0(q)|(q+q0)^n|v1(q)>
                                     = \sum_i comb(n,i) ints_orig[v0,v1,i] * q0**(n-i)
                """
                ints_disp = [
                    np.zeros((nprim, nprim)) for n in range(norder + 1)
                ]
                for v0, v1 in itertools.product(range(nprim), range(nprim)):
                    for n in range(norder + 1):
                        for i in range(n + 1):
                            ints_disp[n][v0, v1] += (
                                scipy.special.comb(n, i, exact=True)
                                * ints_orig[v0, v1, i]
                                * pbas.origin_mwc ** (n - i)
                            )

                for i in range(norder + 1):
                    poly_dofs[i][idof] = ints_disp[i]

            return poly_dofs

        # -- BODY ------------------------#
        if kinetic:
            self["d^2"] = {}
            for istate in range(basinfo.get_nstate()):
                self["d^2"][(istate, istate)] = eval_kine_diag(istate)

        poly_terms = ["q^1", "q^2", "q^3", "q^4"]
        for n_ord_minus1, op_key in enumerate(poly_terms):
            self[op_key] = {}
            for istate in range(basinfo.get_nstate()):
                ints_poly_diag = eval_poly_diag(istate, norder=len(poly_terms))
                self[op_key][(istate, istate)] = ints_poly_diag[
                    n_ord_minus1 + 1
                ]
