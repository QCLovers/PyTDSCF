"""
The model class for the Time-Dependent Schrodinger Equation (TDSE) calculation.
"""

from __future__ import annotations

import copy
from typing import Literal

import discvar
import jax
import numpy as np
from discvar.abc import DVRPrimitivesMixin
from loguru import logger

import pytdscf
from pytdscf._const_cls import const
from pytdscf.hamiltonian_cls import (
    HamiltonianMixin,
    PolynomialHamiltonian,
    TensorHamiltonian,
)


class Model:
    """ The wavefunction and operator information class

    Args:
        basinfo (BasInfo) : The wavefunction basis information
        operators (Dict) : The operators. ``operators[name]`` gives a operator \
                named ``name``, such as ``hamiltonian``.
        build_td_hamiltonian (PolynomialHamiltonian) : \
                Time dependent PolynomialHamiltonian. Defaults to ``None``.

    Attributes:
        init_weight_VIBSTATE (List[List[float]]) : \
            Initial weight of VIBSTATE. List length is [nstate, ndof].
        init_weight_GRID (List[List[float]]) : \
            Initial weight of GRID. List length is [nstate, ndof].
        init_weight_ESTATE (List[float]) : Initial weight of ESTATE. List length is nstate.
        ints_prim_file (str) : The file name of the primitive integrals
        m_aux_max (int) : The maximum number of auxiliary basis.
        basinfo (BasInfo) : The wavefunction basis information
        hamiltonian (PolynomialHamiltonian) : PolynomialHamiltonian
        observables (Dict) : Observable operators, such as PolynomialHamiltonian, \
                dipole moment, occupation number etc.
        build_td_hamiltonian (PolynomialHamiltonian) : Time-dependent PolynomialHamiltonian.

    """

    init_weight_VIBSTATE: list[list[float]] | None = None
    init_weight_ESTATE: list[float] | None = None
    init_HartreeProduct: list[list[list[float]]] | None = (
        None  # [state][dof][basis]
    )
    ints_prim_file: str | None = None
    m_aux_max: int | None = None
    subspace_inds: dict[int, tuple[int, ...]] | None

    def __init__(
        self,
        basinfo: BasInfo,
        operators: dict[str, HamiltonianMixin],
        *,
        build_td_hamiltonian: PolynomialHamiltonian | None = None,
        space: Literal["hilbert", "liouville"] = "hilbert",
        subspace_inds: dict[int, tuple[int, ...]] | None = None,
        one_gate_to_apply: TensorHamiltonian | None = None,
        kraus_op: dict[tuple[int, ...], np.ndarray | jax.Array] | None = None,
    ):
        self.basinfo = basinfo
        self.hamiltonian = operators.pop("hamiltonian")
        self.observables = operators
        self.build_td_hamiltonian = build_td_hamiltonian
        if self.hamiltonian.nstate != basinfo.get_nstate():
            raise ValueError(
                "The number of states in Hamiltonian and BasInfo are different."
            )
        self.nstate = self.hamiltonian.nstate
        self.use_mpo = isinstance(self.hamiltonian, TensorHamiltonian)
        if space.lower() not in ["hilbert", "liouville"]:
            raise ValueError(
                f"space must be 'hilbert' or 'liouville' but got {space}"
            )
        self.space: Literal["hilbert", "liouville"] = space.lower()  # type: ignore
        self.one_gate_to_apply = one_gate_to_apply
        if self.space == "liouville" and subspace_inds is not None:
            assert isinstance(subspace_inds, dict)
            self.subspace_inds = subspace_inds
            for operator in self.observables.values():
                assert isinstance(operator, TensorHamiltonian)
                operator.project_subspace(subspace_inds)
            self.hamiltonian.project_subspace(subspace_inds)
            if self.one_gate_to_apply is not None:
                self.one_gate_to_apply.project_subspace(subspace_inds)
        else:
            self.subspace_inds = None
        self.kraus_op = kraus_op

    def get_nstate(self) -> int:
        """
        Returns:
            int : Number of electronic states
        """
        return self.basinfo.get_nstate()

    def get_ndof(self) -> int:
        """
        Returns:
            int : Degree of freedoms
        """
        return self.basinfo.get_ndof()

    def get_ndof_per_sites(self):
        """N.Y.I ?"""
        return self.basinfo.get_ndof_per_sites()

    def get_primbas(
        self, istate: int, idof: int
    ) -> pytdscf.basis._primints_cls.PrimBas_HO:
        """
        Args:
            istate (int) : index of electronic states
            idof (int) : index of degree of freedom
        Returns:
            primints_cls.PrimBas_HO : The primitive basis in istate, idof.
        """
        return self.basinfo.get_primbas(istate, idof)

    def get_nspf(self, istate: int, idof: int) -> int:
        """
        Args:
            istate (int) : index of electronic states
            idof (int) : index of degree of freedom
        Returns:
            int : The number of SPF in istate, idof.
        """
        return self.basinfo.get_nspf(istate, idof)

    def get_nprim(self, istate: int, idof: int) -> int:
        """
        Args:
            istate (int) : index of electronic states
            idof (int) : index of degree of freedom
        Returns:
            int : The number of primitive basis in istate, idof.
        """
        return self.basinfo.get_nprim(istate, idof)

    def get_nspf_list(self, istate: int) -> list[int]:
        """
        Args:
            istate (int) : index of electronic states
        Returns:
            List(int) : Number of SPFs. e.g. ``[2, 2]``
        """
        return self.basinfo.get_nspf_list(istate)

    def get_nspf_rangelist(self, istate: int) -> list[list[int]]:
        """
        Args:
            istate (int) : index of electronic states
        Returns:
            List[List[int]] : the indices of SPFs. e.g. \
                    ``[[0,1,2],[0,1,2]]``
        """
        return self.basinfo.get_nspf_rangelist(istate)


class BasInfo:
    """ The Wave function basis information class

    Args:
        prim_info (List[List[PrimBas_HO or DVRPrimitivesMixin]]) : \
                ``prim_info[istate][idof]`` gives ``PrimBas_HO``.
        spf_info (List[List[int]]) : ``spf_info[istate][idof]`` gives \
                the number of SPF.
        ndof_per_sites (bool) : Defaults to ``None``. N.Y.I.

    Attributes:
        prim_info (List[List[PrimBas_HO or DVRPrimitivesMixin]]) : \
                ``prim_info[istate][idof]`` gives ``PrimBas_HO``.
        spf_info (List[List[int]]) : ``spf_info[istate][idof]`` gives \
                the number of SPF.
        nstate (int) : The number of electronic states.
        ndof (int) : The degree of freedoms.
        nspf (int) : The number of SPF.
        nspf_list (List[int]) : The number of SPFs.
        nspf_rangelist (List[List[int]]) : The indices of SPFs.

    """

    def __init__(self, prim_info, spf_info=None, ndof_per_sites=None):
        self.prim_info = copy.deepcopy(prim_info)
        self.is_DVR = any(
            isinstance(basis, pytdscf.basis.abc.DVRPrimitivesMixin)
            or isinstance(basis, DVRPrimitivesMixin)
            for basis in prim_info[0]
        )
        self.need_primints = any(
            isinstance(basis, pytdscf.PrimBas_HO | discvar.ho.PrimBas_HO)
            for basis in prim_info[0]
        )
        if spf_info is None:
            if const.verbose > 1:
                logger.info("The layer of SPF is not used.")
            self.spf_info = [
                [
                    len(self.prim_info[istate][idof])
                    for idof in range(self.get_ndof())
                ]
                for istate in range(self.get_nstate())
            ]
            self.is_standard_method = True
        else:
            self.spf_info = copy.deepcopy(spf_info)
            self.is_standard_method = False
        if ndof_per_sites:
            raise NotImplementedError
            self.ndof_per_sites = ndof_per_sites

    def get_nstate(self) -> int:
        """Get ``nstate`` attributes

        Returns:
            int : Number of electronic states

        """
        if not hasattr(self, "nstate"):
            self.nstate = len(self.prim_info)
        return self.nstate

    def get_ndof(self) -> int:
        """Get ``ndof`` attributes

        Returns:
            int : Degree of freedom

        """
        if not hasattr(self, "ndof"):
            self.ndof = len(self.prim_info[0])
        return self.ndof

    def get_ndof_per_sites(self) -> list[int]:
        """Get ``ndof_per_sites`` attributes"""
        raise NotImplementedError
        # return self.ndof_per_sites

    def get_primbas(
        self, istate: int, idof: int
    ) -> pytdscf.basis._primints_cls.PrimBas_HO:
        """Get ``prim_info[istate][idof]`` attributes
        Args:
            istate (int) : index of electronic states
            idof (int) : index of degree of freedom

        Returns:
            PrimBas_HO : The primitive basis of istate, idof.

        """
        # NYI->i_set = self.state_label[istate][idof]
        return self.prim_info[istate][idof]

    def get_nspf(self, istate: int, idof: int) -> int:
        """Get number of SPF

        Args:
            istate (int) : index of electronic states
            idof (int) : index of degree of freedom
        Returns:
            int : Number of SPF

        """
        # NYI->i_set = self.state_label[istate][idof]
        return self.spf_info[istate][idof]

    def get_nprim(self, istate: int, idof: int) -> int:
        """Get number of primitive basis

        Args:
            istate (int) : index of electronic states
            idof (int) : index of degree of freedom
        Returns:
            int : Number of primitive basis

        """
        return self.prim_info[istate][idof].nprim

    def get_ngrid(self, istate: int, idof: int) -> int:
        return self.get_nprim(istate, idof)

    def get_nspf_list(self, istate: int) -> list[int]:
        """Get number of SPFs list ``nspf_list`` attributes

        Args:
            istate (int) : index of electronic states
        Returns:
            list(int) : Number of SPFs. e.g. ``[2, 2]``

        """
        if not hasattr(self, "nsfp_list"):
            self.nspf_list = []
            for idof in range(self.get_ndof()):
                self.nspf_list.append(self.get_nspf(istate, idof))
        return self.nspf_list

    def get_nspf_rangelist(self, istate: int) -> list[list[int]]:
        """ Get number of SPFs list ``nspf_rangelist`` attributes

        Args:
            istate (int) : index of electronic states
        Returns:
            List[List[int]] : the indices of SPFs. e.g. \
                    ``[[0,1,2],[0,1,2]]``

        """
        if not hasattr(self, "nsfp_rangelist"):
            self.nspf_rangelist = []
            for idof in range(self.get_ndof()):
                self.nspf_rangelist.append(
                    list(range(self.get_nspf(istate, idof)))
                )
        return self.nspf_rangelist
