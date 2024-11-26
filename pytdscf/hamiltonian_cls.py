"""
The operator modules consists Hamiltonian.
"""

import itertools
import math
import random
from logging import getLogger

import jax
import jax.numpy as jnp
import numpy as np

from pytdscf import units
from pytdscf._const_cls import const
from pytdscf._mpo_cls import MatrixProductOperators
from pytdscf.dvr_operator_cls import TensorOperator

logger = getLogger("main").getChild(__name__)


class TermProductForm:
    """

    Hold a product form of operators, such as :math:`1.0 q_1 q_2^2`.

    Attributes:
        coef (float or complex) : The coefficient of the product operator, e.g. ``1.0``.
        op_dofs (List[int]) : The MPS lattice (0-)index of operator, e.g. ``[0, 1]``.
        op_keys (List[str]) : The type of operators for each legs, e.g. ``['q', 'q^2']``.
        mode_ops (Dict[Tuple[int], str]) : The pair of ``op_dofs`` and ``op_keys``.
        blockop_key_sites (Dict[str, List[str]]) : Operator legs divided by central site. E.g. \
                3 DOFs, ``'q_0q^2_1'`` gives \
                ``{left:['ovlp','q_0', 'q_0q^2_1'], centr:['q_0', 'q^2_1', 'ovlp'], right:['q^2_1', 'ovlp', 'ovlp']}``.

    """

    def __init__(
        self, coef: float | complex, op_dofs: list[int], op_keys: list[str]
    ):
        self.coef = coef
        self.op_dofs = op_dofs
        self.op_keys = op_keys
        self.mode_ops = {
            op_dof: op_key
            for op_dof, op_key in zip(op_dofs, op_keys, strict=True)
        }
        assert len(op_dofs) == len(op_keys)

    def __str__(self) -> str:
        if isinstance(self.coef, float):
            sign = "+" if self.coef > 0.0 else "-"
            return (
                sign
                + f"{abs(self.coef):.4e}"
                + " "
                + " * ".join(
                    [
                        op_key + "_" + str(op_dof)
                        for (op_dof, op_key) in zip(
                            self.op_dofs, self.op_keys, strict=True
                        )
                    ]
                )
            )
        else:
            assert isinstance(self.coef, complex)
            real_part = f"{self.coef.real:.4e}"
            imag_part = f"{self.coef.imag:.4e}"
            sign_real = "+" if self.coef.real > 0.0 else "-"
            sign_imag = "+" if self.coef.imag > 0.0 else "-"
            return (
                sign_real
                + real_part
                + sign_imag
                + imag_part
                + "j "
                + " * ".join(
                    [
                        op_key + "_" + str(op_dof)
                        for (op_dof, op_key) in zip(
                            self.op_dofs, self.op_keys, strict=True
                        )
                    ]
                )
            )

    @staticmethod
    def convert_key(op_dofs: list[int], op_keys: list[str]) -> str:
        """

        convert product operator to string key

        Args:
            op_dofs (List[int]) : The MPS lattice (0-)index of operator, e.g. ``[0, 1]``.
            op_keys (List[str]) : The type of operators for each legs, e.g. ``['q', 'q^2']``.

        Returns:
            str : string key, e.g. ``q_0 q^2_1``.

        Examples:
            >>> TermProductForm.convert_key([1, 2], ['q', 'q^2'])
                'q_1 q^2_2'

        """
        key = ""
        for op_dof, op_key in zip(op_dofs, op_keys, strict=True):
            key += op_key + "_" + str(op_dof) + " "
        return key.rstrip()

    def term_key(self) -> str:
        """

        call `convert_key` attributes in this class.

        Returns:
            str : string key, e.g. ``q_1 q^2_2``.

        """
        return self.convert_key(self.op_dofs, self.op_keys)

    def is_op_ovlp(self, block_lcr_list: list[str], psite: int) -> bool:
        """

        check MPS lattice block is ``'ovlp'`` operator or not.

        Args:
            block_lcr_list (List[str]) : ``'left'``, or ``'centr'``, or ``'right'``
            psite (int) : (0-) index of center site. LCR of C.

        Returns:
            bool : Return ``True`` if all operator types with legs in the given block\
                    are ``'ovlp'`` when regard centered site as the ``psite``-th site.

        """
        assert hasattr(
            self, "blockop_key_sites"
        ), "blockop_key_sites attribute \
                has not set. Call set_blockp_key attribute before call this attribute."
        for block_lcr in block_lcr_list:
            if self.blockop_key_sites[block_lcr][psite] != "ovlp":
                return False
        return True

    def set_blockop_key(self, ndof: int, *, print_out: bool = False):
        """

        For the complementary operator, the product operator is set to an attribute\
                `blockop_key_sites` that is classified according to the center site.

        Args:
            ndof (int) : The length of MPS lattice, that is degree of freedoms.
            print_out (bool) : Print ``blockop_key_sites`` or not. Defaults to ``False``.

        """
        blockop_key_l = []
        blockop_key_c = []
        blockop_key_r = []
        for centr_dof in range(ndof):
            op_l_key = ""
            op_r_key = ""
            for kdof in self.op_dofs:
                if kdof < centr_dof:
                    op_l_key += (
                        ("" if len(op_l_key) == 0 else " ")
                        + self.mode_ops[kdof]
                        + "_"
                        + str(kdof)
                    )
                if kdof > centr_dof:
                    op_r_key += (
                        ("" if len(op_r_key) == 0 else " ")
                        + self.mode_ops[kdof]
                        + "_"
                        + str(kdof)
                    )
            blockop_key_l.append("ovlp" if len(op_l_key) == 0 else op_l_key)
            blockop_key_c.append(
                "ovlp"
                if centr_dof not in self.op_dofs
                else self.mode_ops[centr_dof]
            )
            blockop_key_r.append("ovlp" if len(op_r_key) == 0 else op_r_key)
        self.blockop_key_sites = {
            "left": blockop_key_l,
            "centr": blockop_key_c,
            "right": blockop_key_r,
        }
        if print_out and const.verbose == 4:
            for idof in range(ndof):
                logger.debug(
                    "("
                    + self.blockop_key_sites["left"][idof]
                    + ", "
                    + self.blockop_key_sites["centr"][idof]
                    + ", "
                    + self.blockop_key_sites["right"][idof]
                    + ")"
                )


class TermOneSiteForm:
    """Operator with legs in only one degree of freedom

    Attributes:
        coef (float or complex) : The coefficient of the product operator, e.g. ``1.0``.
        op_dofs (int) : The MPS lattice (0-)index of operator, e.g. ``0``.
        op_keys (str) : The type of operators for each legs, e.g. ``'q'``, ``'d^2'``.

    """

    def __init__(self, coef: float | complex, op_dof: int, op_key: str):
        self.coef = coef
        self.op_dof = op_dof
        self.op_key = op_key

    def __str__(self) -> str:
        if isinstance(self.coef, complex):
            real_part = f"{self.coef.real:.4e}"
            imag_part = f"{self.coef.imag:.4e}"
            sign_real = "+" if self.coef.real > 0.0 else "-"
            sign_imag = "+" if self.coef.imag > 0.0 else "-"
            return (
                sign_real
                + real_part
                + sign_imag
                + imag_part
                + "j "
                + self.op_key
                + "_"
                + str(self.op_dof)
            )
        else:
            sign = "+" if self.coef > 0.0 else "-"
            return (
                sign
                + f"{abs(self.coef):.4e}"
                + " "
                + self.op_key
                + "_"
                + str(self.op_dof)
            )


def truncate_terms(
    terms: list[TermProductForm], cut_off: float = 0.0
) -> list[TermProductForm]:
    """
    Truncate operators by a certain cut_off.

    Args:
        terms (List[TermProductForm]) : Terms list of sum of products form before truncation.
        cut_off (float) : The threshold of truncation. unit is a.u.

    Returns:
        terms (List[TermProductForm]) : Terms list of sum of products form after truncation.
    """
    terms_after = []
    for term in terms:
        if abs(term.coef) >= cut_off:
            terms_after.append(term)
    return terms_after


def _accumulate_q0_terms(terms: list[TermProductForm]):
    """Regard ovlp operator as q_0 operator"""
    coef_q0 = 0.0
    terms_wo_const = []
    for term in terms:
        if "ovlp" in term.op_keys:
            coef_q0 += term.coef  # type: ignore
        else:
            terms_wo_const.append(term)
    return (terms_wo_const, coef_q0)


def _accumulate_q0_terms_with_const(terms: list[TermProductForm], ndof: int):
    """Regard ovlp operator as random dofs operator"""
    coef_q0 = 0.0
    terms_with_const = []
    for term in terms:
        if "ovlp" in term.op_keys:
            coef_q0 += term.coef  # type: ignore
        else:
            terms_with_const.append(term)
    dof_any = random.choice(range(ndof))
    terms_with_const.append(TermProductForm(coef_q0, [dof_any], ["ovlp"]))
    return (terms_with_const, 0.0)


def _extract_onesite(
    terms_general: list[TermProductForm],
) -> tuple[list[TermProductForm], list[TermOneSiteForm]]:
    """
    Extract onesite operators, such as 'q^2_1', 'd^2_3' from all terms.

    Args:
        terms_general (List[TermProductForm]) : \
            List of all sume of products operators.

    Returns:
         Tuple(List[TermProductForm], List[TermOneSiteForm])) : \
            terms_multisite, terms_onesite

    """
    terms_multisite = []
    terms_onesite = []
    for term in terms_general:
        assert (
            len(term.op_dofs) > 0
        ), 'define PolynomialHamiltonian scalar term as _matH attribute (not in "general" type operator)'
        if len(term.op_dofs) == 1:
            terms_onesite.append(
                TermOneSiteForm(term.coef, term.op_dofs[0], term.op_keys[0])
            )
        else:
            terms_multisite.append(term)
    return (terms_multisite, terms_onesite)


class HamiltonianMixin:
    """Hamiltonian abstract class.

    Attributes:
        name (str) : The name of operator.
        onesite_name (str) : The name of onesite term. Defaults tp ``'onesite'``.
        nstete (int) : The number of electronic states.
        ndof (int) : The degree of freedoms.
        coupleJ (List[List[complex]]) : The couplings of electronic states. \
                ``coupleJ[i][j]`` denotes the couplings between `i`-states (bra) \
                and `j`-states (ket). This contains scalar operator.
    """

    def __init__(
        self,
        name: str,
        nstate: int,
        ndof: int,
        matJ: list[list[complex | float]] | None = None,
    ):
        self.name = name
        self.nstate = nstate
        self.ndof = ndof
        if matJ is None:
            self.coupleJ = [
                [complex(0.0) for j in range(nstate)] for i in range(nstate)
            ]
        else:
            assert (
                len(matJ) == nstate and len(matJ[0]) == nstate
            ), "matJ must be square matrix"
            self.coupleJ = [
                [complex(matJ[i][j]) for j in range(nstate)]
                for i in range(nstate)
            ]


class PolynomialHamiltonian(HamiltonianMixin):
    """
    PolynomialHamiltonian package, contains some model Hamiltonian. Sometimes, this \
            class also manage observable operators, such as dipole operator

    Attributes:
        coupleJ (List[List[complex]]) : The couplings of electronic states. \
                ``coupleJ[i][j]`` denotes the couplings between `i`-states (bra) \
                and `j`-states (ket). This contains scalar operator.
        onesite (List[List[List[TermProductForm or TermOneSiteForm]]]) : \
                The onesite operators. ``onesite[i][j][k]`` \
                denotes `k`-th onesite operator (TermOneSiteForm) between \
                `i`-states (bra) and `j`-states(ket).
        general (List[List[List[TermProductForm or TermOneSiteForm]]]) : \
                The multisite operators. The definition is \
                almost the same as ``onesite``.
    """

    def __init__(
        self, ndof: int, nstate: int = 1, name: str = "hamiltonian", matJ=None
    ):
        super().__init__(name, nstate, ndof, matJ)
        self.onesite_name = (
            "onesite" if name == "hamiltonian" else "onesite_" + name
        )
        self.onesite: list[list[list[TermProductForm | TermOneSiteForm]]] = [
            [[] for j in range(nstate)] for i in range(nstate)
        ]
        self.general: list[list[list[TermProductForm | TermOneSiteForm]]] = [
            [[] for j in range(nstate)] for i in range(nstate)
        ]

    def set_HO_potential_ham1(self, basinfo):
        for istate in range(self.nstate):
            terms = []
            for idof in range(self.ndof):
                terms.append(TermOneSiteForm(1.0, idof, "ham1"))
            self.onesite[istate][istate] += terms

    def set_HO_potential(self, basinfo, *, enable_onesite=True) -> None:
        """Setting Harmonic Oscillator polynomial-based potential"""

        """Intra-state terms |i><i|"""
        for istate in range(self.nstate):
            terms = []
            for idof in range(self.ndof):
                pbas = basinfo.get_primbas(istate, idof)

                """V(Q) = (w**2/2) (q-q0)**2 [= (w/2) (Q-Q0)**2]
                        = (w**2/2) (q**2 -(2*q0)*q +(q0**2))
                """
                q0 = pbas.origin_mwc
                w = pbas.freq_au
                terms.append(TermProductForm(-1 / 2, [idof], ["d^2"]))
                terms.append(TermProductForm(w**2 / 2, [idof], ["q^2"]))
                if q0 != 0.0:
                    terms.append(
                        TermProductForm(w**2 / 2 * (-2 * q0), [idof], ["q^1"])
                    )
                    terms.append(
                        TermProductForm(w**2 / 2 * (q0**2), [idof], ["ovlp"])
                    )

            terms = truncate_terms(terms)
            terms_general, coupleJ = _accumulate_q0_terms_with_const(
                terms, self.ndof
            )

            if enable_onesite:
                terms_general, terms_onesite = _extract_onesite(terms_general)
                self.onesite[istate][istate] += terms_onesite

            self.coupleJ[istate][istate] += coupleJ
            self.general[istate][istate] += terms_general

    def set_LVC(
        self,
        bas_info,
        first_order_coupling: dict[tuple[int, int], dict[int, float]],
    ):
        """Setting Linear Vibronic Coupling (LVC) Model

        Args:
            bas_info (BasisInfo): Basis information
            first_order_coupling (Dict[Tuple[int, int], Dict[int, float]]): First order coupling
                if {(0,1): {2: 0.1}} is given, then the coupling between 0-state and 1-state is 0.1*Q_2


        """
        self.set_HO_potential(bas_info, enable_onesite=True)
        for (istate, jstate), coupling in first_order_coupling.items():
            for idof, coef in coupling.items():
                self.onesite[istate][jstate].append(
                    TermOneSiteForm(coef, idof, "q^1")
                )

    def is_hermitian(self):
        raise NotImplementedError

    def set_ConIns_potential(self, basinfo):
        """Intra-state terms i><i"""
        for istate in range(self.nstate):
            terms = []
            for idof in range(self.ndof):
                pbas = basinfo.get_primbas(istate, idof)

                """V(Q) = (w**2/2) (q-q0)**2 [= (w/2) (Q-Q0)**2]
                        = (w**2/2) (q**2 -(2*q0)*q +(q0**2))
                """
                q0 = pbas.origin_mwc
                w = pbas.freq_au
                # pbas.nprim
                terms.append(TermProductForm(-1 / 2, [idof], ["d^2"]))
                terms.append(TermProductForm(w**2 / 2, [idof], ["q^2"]))
                terms.append(
                    TermProductForm(w**2 / 2 * (-2 * q0), [idof], ["q^1"])
                )
                terms.append(
                    TermProductForm(w**2 / 2 * (q0**2), [idof], ["ovlp"])
                )

            terms = truncate_terms(terms)
            terms_general, coupleJ = _accumulate_q0_terms_with_const(
                terms, self.ndof
            )

            terms_general, terms_onesite = _extract_onesite(terms_general)
            self.coupleJ[istate][istate] += coupleJ
            self.general[istate][istate] += terms_general
            self.onesite[istate][istate] += terms_onesite

        """|al><be|op terms """
        for istate, jstate in itertools.product(range(self.nstate), repeat=2):
            if istate != jstate:
                terms_onesite = []
                for idof in [0]:
                    coupleL2 = (50 / units.au_in_cm1) * (
                        206.0 / units.au_in_cm1
                    ) ** 1.0
                    terms_onesite.append(TermOneSiteForm(coupleL2, idof, "q^2"))
                self.onesite[istate][jstate] += terms_onesite

    def set_henon_heiles(
        self,
        omega: float,
        lam: float,
        f: int,
        omega_unit: str = "cm-1",
        lam_unit: str = "a.u.",
    ) -> list[list[TermProductForm]]:
        r"""Setting Henon-Heiles potential

        In dimensionless form, the Hamiltonian is given by

        .. math::
           \hat{H} = \frac{\omega}{2}\sum_{i=1}^{f} \left( - \frac{\partial^2}{\partial q_i^2} + q_i^2 \right) \
           + \lambda \left( \sum_{i=1}^{f-1} q_i^2 q_{i+1} - \frac{1}{3} q_{i+1}^3 \right)

        But PyTDSCF adopts mass-weighted coordinate, thus the Hamiltonian is given by

        .. math::
           \hat{H} = \frac{1}{2}\sum_{i=1}^{f} \left( - \frac{\partial^2}{\partial Q_i^2} + \omega^2 Q_i^2 \right) \
           + \lambda \omega^{\frac{3}{2}} \left( \sum_{i=1}^{f-1} Q_i^2 Q_{i+1} - \frac{1}{3} Q_{i+1}^3 \right)


        Args:
            omega (float): Harmonic frequency in a.u.
            lam (float): Coupling constant in a.u.
            f (int): Number of degrees of freedom

        Returns:
            List[List[TermProductForm]]: List of Hamiltonian terms

        """
        terms = []
        if omega_unit == "cm-1":
            omega /= units.au_in_cm1
        elif omega_unit.lower() in ["au", "a.u.", "hartree"]:
            pass
        else:
            raise ValueError("omega_unit must be cm-1 or a.u.")

        if lam_unit == "cm-1":
            lam = lam / units.au_in_cm1
        elif lam_unit.lower() in ["au", "a.u.", "hartree"]:
            pass
        else:
            raise ValueError("lam_unit must be cm-1 or a.u.")

        for idof in range(f):
            terms.append(TermProductForm(-1 / 2, [idof], ["d^2"]))
            terms.append(TermProductForm(pow(omega, 2) / 2, [idof], ["q^2"]))
        for idof in range(f - 1):
            terms.append(
                TermProductForm(
                    lam * pow(omega, 1.5), [idof, idof + 1], ["q^2", "q^1"]
                )
            )
            terms.append(
                TermProductForm(-lam * pow(omega, 1.5) / 3, [idof + 1], ["q^3"])
            )

        istate = 0
        terms_general, terms_onesite = _extract_onesite(terms)
        self.coupleJ[istate][istate] = 0.0
        self.general[istate][istate] += terms_general
        self.onesite[istate][istate] += terms_onesite
        return [terms]

    def set_henon_heiles_2D_4th(
        self, lam: float = 0.2
    ) -> list[list[TermProductForm]]:
        r"""

        .. math::
           H(x,y) = -\frac{1}{2}\left(\frac{d^2}{dx^2} + \frac{d^2}{dy^2}\right) \
           + \frac{1}{2}(x^2 + y^2) + \lambda \left(xy^2 - \frac{1}{3}x^3\right)\
           + \lambda^2 \left(\frac{1}{16}(x^4 + y^4) + \frac{1}{8}x^2y^2\right)

        """
        terms = []
        xdof = 0
        ydof = 1
        """-(1/2)(d^2_x + d^2_y)"""
        terms.append(TermProductForm(-1 / 2, [xdof], ["d^2"]))
        terms.append(TermProductForm(-1 / 2, [ydof], ["d^2"]))
        """(1/2)(x^2 + y^2)"""
        terms.append(TermProductForm(+1 / 2, [xdof], ["q^2"]))
        terms.append(TermProductForm(+1 / 2, [ydof], ["q^2"]))
        """lam (x*y^2 - (1/3)*x^3)"""
        terms.append(TermProductForm(lam, [xdof, ydof], ["q^1", "q^2"]))
        terms.append(TermProductForm(-(1 / 3) * lam, [xdof], ["q^3"]))
        """lam^2 ((x^4 + y^4)/16 + (x^2*y^2)/8"""
        terms.append(TermProductForm((1 / 16) * lam**2, [xdof], ["q^4"]))
        terms.append(TermProductForm((1 / 16) * lam**2, [ydof], ["q^4"]))
        terms.append(
            TermProductForm((1 / 8) * lam**2, [xdof, ydof], ["q^2", "q^2"])
        )
        """"""
        for term_prod in terms:
            term_prod.set_blockop_key(self.ndof)

        istate = 0
        terms_general, terms_onesite = _extract_onesite(terms)
        self.coupleJ[istate][istate] = 0.0
        self.onesite[istate][istate] += terms_onesite
        self.general[istate][istate] += terms_general

        return [terms]


class TensorHamiltonian(HamiltonianMixin):
    """Hamiltonian in tensor formulation.

    Attributes:
        mpo (List[List[MatrixProductOperators]]) : MPO between i-state and j-state

    """

    def __init__(
        self,
        ndof: int,
        potential: list[
            list[dict[tuple[int | tuple[int, int], ...], TensorOperator]]
        ],
        name: str = "hamiltonian",
        kinetic: list[list[dict[tuple[int, int], TensorOperator] | None]]
        | None = None,
        decompose_type: str = "QRD",
        rate: float | None = None,
        bond_dimension: list[int] | int | None = None,
        backend="jax",
    ):
        """
        Args:
            ndof (int): degree of freedom, equals to number os sites
            potential (List[List[Dict[Tuple[int], TensorOperator]]]): [istate][jstate] PES
            name (str, optional): Defaults to 'hamiltonian'.
            kinetic (Dict[Tuple[int], TensorOperator], optional): kinetic term. Defaults to None. \
                kinetic term are the same for all states.
            decompose_type (Optional[str], optional): MPO decompose algorithm. Defaults to 'QRD'.
            rate (Optional[float], optional): SVD-MPO contribution rate. Defaults to None.
            bond_dimension (Optional[List[int] or int], optional): SVD-MPO bond dimension. Defaults to None.
        """
        nstate = len(potential)
        super().__init__(name, nstate, ndof)
        self.mpo = [[None for j in range(nstate)] for i in range(nstate)]
        if const.verbose > 2:
            logger.info(f"Start tensor decomposition: type = {decompose_type}")
        for i, j in itertools.product(range(nstate), range(nstate)):
            operators: dict[
                tuple[int | tuple[int, int], ...],
                list[np.ndarray] | list[jax.Array],
            ] = {}
            if potential[i][j] is not None:
                for key, tensor in potential[i][j].items():
                    if key == ():
                        if not (isinstance(tensor, float | complex | int)):
                            raise ValueError(
                                f"scalar term must be scalar but {tensor} is {type(tensor)}"
                            )
                        self.coupleJ[i][j] = tensor
                        continue
                    else:
                        # key is (0,1,...) or ((0,0),(1,1),...)
                        # if key is (0,1,...), then return (0,1,...)
                        # if key is ((0,0),(1,1),...), then return (0,0,1,1,...)
                        flatten_key = tuple()
                        for k in key:
                            if type(k) is tuple:
                                flatten_key += k
                            else:
                                flatten_key += (k,)
                        if flatten_key != tensor.legs:
                            raise ValueError(
                                f"Given potential key {key} is not consistent with tensor legs {tensor.legs}"
                            )
                    if backend.lower() == "jax":
                        if tensor.dtype in [jnp.complex128, np.complex128]:
                            dtype = jnp.complex128
                        elif tensor.dtype in [jnp.float64, np.float64]:
                            dtype = jnp.float64
                        else:
                            raise ValueError(
                                f"core dtype must be complex128 or float64 but {tensor.dtype} is given"
                            )
                        operators[key] = [
                            jnp.array(core, dtype=dtype)
                            for core in tensor.decompose(
                                bond_dimension=bond_dimension,
                                decompose_type=decompose_type,
                                rate=rate,
                            )
                        ]
                    elif backend.lower() == "numpy":
                        operators[key] = [
                            core
                            for core in tensor.decompose(
                                bond_dimension=bond_dimension,
                                decompose_type=decompose_type,
                                rate=rate,
                            )
                        ]
                    else:
                        raise ValueError(
                            f"backend must be jax, or numpy but {backend} is given"
                        )

            if kinetic is not None and kinetic[i][j] is not None:
                assert isinstance(kinetic, list)
                K_ij = kinetic[i][j]
                assert isinstance(K_ij, dict)
                for key, d2 in K_ij.items():
                    if backend.lower() == "jax":
                        if d2.dtype in [jnp.complex128, np.complex128]:
                            dtype = jnp.complex128
                        elif d2.dtype in [jnp.float64, np.float64]:
                            dtype = jnp.float64
                        if key in operators:
                            raise ValueError(
                                f"key {key} is already set in potential. "
                                + "Concatenate KEO and PEO or set KEO as SOP"
                            )
                        operators[key] = [
                            jnp.array(core, dtype=dtype)
                            for core in d2.decompose()
                        ]
                    else:
                        operators[key] = d2.decompose()
            self.mpo[i][j] = MatrixProductOperators(
                nsite=ndof, operators=operators, backend=backend
            )


def read_potential_nMR(
    potential_emu: dict[tuple[int, ...], float | complex],
    *,
    active_modes: list[int] | None = None,
    name: str = "hamiltonian",
    cut_off: float | None = None,
    dipole_emu: dict[tuple[int, ...], tuple[float, float, float]] | None = None,
    print_out: bool = False,
    active_momentum=None,
    div_factorial: bool = True,
    efield: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> PolynomialHamiltonian:
    """ Construct polynomial potential, such as n-Mode Representation form. \
        multi-state nMR is not yet implemented.

    Args:
        potential_emu (Dict[Tuple[int], float or complex]) : The coeffcients of operators. \
                The index is start from `1`. Factorial factor is not included in the coefficients.\
                E.g. V(q_1) = 0.123 * 1/(2!) * q_1^2 gives ``{(1,1):0.123}``.\n
                units are all a.u.:\n
                    k_orig{ij} in [Hartree/emu/bohr**2]\n
                    k_orig{ijk} in [Hartree/emu**i(3/2)/BOHR**3]\n
                    k_orig{ijkl} in [Hartree/emu**(2)/BOHR**4]\n
        active_modes (List[int]) : The acctive degree of freedoms.\
                Note that, The order of the MPS lattice follows this list.\
                The index is start from `1`.
        name (str) : The name of operator.
        cut_off (float) : The threshold of truncatig terms by coeffcients after \
                factorial division.
        dipole_emu (Dict[Tuple[int], float or complex]) : Dipole moment operator to generate a vibrational \
                excited states. Defaults to ``None``. The definition is almost the \
                same as `potential_emu`, but this value is 3d vector (list).\n
                    mu{ij} in [Debye/emu/bohr**2]\n
                    mu{ijk} in [Debye/emu**(3/2)/BOHR**3]\n
                    mu{ijkl} in [Debye/emu**(2)/BOHR**4]\n
        print_out (bool) : Defaults to ``False``
        div_factorial(bool) : Whether or not divided by factorial term. \
            Defaults to ``True``.
        active_momentum (List[Tuple[int, float]]) : [(DOF, coef)]. Defaults to ``None``\
                When active_momentum are set, unselected kinetic energy operators \
                are excluded. Default option include all kinetic energy operators.
        efield (List[float]) : The electric field in [Hartree Bohr^-1 e^-1]. Defaults to ``[1.0, 1.0, 1.0]``

    Returns:
        PolynomialHamiltonian : operator
    """
    if active_modes is None:
        if dipole_emu is not None:
            active_modes = sorted(
                list(set(itertools.chain.from_iterable(dipole_emu.keys())))
            )
        elif potential_emu is not None:
            active_modes = sorted(
                list(set(itertools.chain.from_iterable(potential_emu.keys())))
            )
        else:
            raise ValueError("active_modes must be set")
    logger.info("Construct anharmonic polynomial operator")
    k_orig = potential_emu
    scalar_term: float = 0.0

    if dipole_emu is not None:
        mu = dipole_emu
        active_momentum = False
        k_orig = {}
        for key, val in mu.items():
            if key == ():
                scalar_term += float(np.dot(val, efield))
            else:
                k_orig[key] = np.dot(val, efield)

    # mode_set = set(itertools.chain.from_iterable([key for key in k_orig.keys()]))
    dic = {}
    for i in range(len(active_modes)):
        dic[active_modes[i]] = i + 1
    nmode = len(active_modes)  # max([len(key) for key in k_orig.keys()])

    nstate = 1
    ndof = nmode

    k = {}
    for key, value in k_orig.items():
        """
        e.g. k_orig[(1,2,3,3)] -means-> coef[(1,1,2)] q_0 q_1 (q_2)^2
             k_orig[(3,3,1)]   -means-> coef[(1,0,2)] q_0     (q_2)^2
        """
        assert isinstance(value, float)
        if key == ():
            scalar_term = value
            continue

        if set(key) & set(active_modes) != set(key):
            continue

        degree_of_q = [0 for i in range(nmode)]
        for imode in key:
            imode = dic[imode]
            degree_of_q[imode - 1] += 1

        if print_out:
            logger.debug(degree_of_q)
        key_new = tuple(degree_of_q)
        assert key_new not in k, "duplicated keys in k_orig"
        k[key_new] = value

    matJ = [[scalar_term]]
    hamiltonian = PolynomialHamiltonian(ndof, nstate, name, matJ)

    terms = []

    """kinetic terms"""
    if active_momentum is None:
        for idof in range(nmode):
            terms.append(TermProductForm(-1 / 2, [idof], ["d^2"]))
    elif active_momentum:
        for idof, coef in active_momentum.items():
            idof = dic[idof] - 1
            terms.append(TermProductForm(coef, [idof], ["d^2"]))

    """potential terms"""
    for key, value in k.items():
        op_dofs = []
        op_keys = []
        fac = 1.0
        for idof, order in enumerate(key):
            if order > 0:
                op_dofs.append(idof)
                op_keys.append("q^" + str(order))
                if div_factorial:
                    fac /= math.factorial(order)
        coef = fac * value
        terms.append(TermProductForm(coef, op_dofs, op_keys))

    if cut_off is not None:
        terms = truncate_terms(terms, cut_off=cut_off)
    if const.verbose > 3:
        logger.debug(f"{name} sum of products term = {len(terms)}")

    terms_multisite, terms_onesite = _extract_onesite(terms)
    for term_prod in terms_multisite:
        term_prod.set_blockop_key(hamiltonian.ndof, print_out=print_out)

    hamiltonian.onesite[0][0] += terms_onesite
    hamiltonian.general[0][0] += terms_multisite

    return hamiltonian
