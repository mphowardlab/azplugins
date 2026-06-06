# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2025, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""Updaters."""

import hoomd

from hoomd.azplugins import _azplugins
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import OnlyTypes


class DynamicBond(hoomd.operation.Updater):
    R"""Update bonds dynamically during simulation.

    Args:
        r_cut (float): Distance cutoff for making bonds between particles
        probability (float): Probability of bond formation, between 0 and 1, default = 1
        bond_type (str): Type of bond to be formed
        group_1 (:py:mod: `hoomd.filter.ParticleFilter`): First particle group
        group_2 (:py:mod:`hoomd.filter.ParticleFilter`): Second particle group
        max_bonds_1 (int): Maximum number of bonds a particle in group_1 can have
        max_bonds_2 (int): Maximum number of bonds a particle in group_2 can have
        seed (int): Seed to the pseudo-random number generator
        nlist (:py:mod:`hoomd.md.nlist`): NeighborList (optional) to update exclusions
        period (int): Particle types will be updated every *period* time steps
        phase (int): When -1, start on the current time step. Otherwise, execute
                     on steps where *(step + phase) % period* is 0.

        Forms bonds of type bond_type between particles in group_1 and group_2 during
        the simulation, if particle distances are shorter than r_cut. If the
        neighborlist used for the pair potential in the simulation is given as a
        parameter nlist, the neighbor list exclusions will be updated to include the
        newly formed bonds. Each particle has a number of maximum bonds which it can
        form, given by max_bonds_1 for particles in group_1 and max_bonds_2 for group_2.

        The particles in the two groups group_1 and group_2 should be completely
        separate with no common elements, e.g. two different types, or the two
        groups should be identical, where now max_bonds_1 needs to be equal to
        max_bonds_2.


        .. warning::
            If the groups group_1 and group_2 are modified during the simulation, this
            Updater will not be updated to reflect the changes. It is the user's
            responsibility to ensure that the groups do not change as long as this
            updater is active.

        Examples::

            azplugins.update.dynamic_bond(
                nlist=nl,
                r_cut=1.0,
                bond_type=0,
                groups=[hoomd.group.type(type="A"),hoomd.group.type(type="B")],
                max_bonds=[2,3]
            )
    """

    _ext_module = _azplugins
    _cpp_class_name = "DynamicBondUpdater"

    def __init__(
        self,
        trigger,
        groups,
        max_bonds,
        bond_type,
        nlist = None,
        r_cut=1.0,
        seed=0,
        probability=1.0,
    ):
        super().__init__(trigger)

        params = ParameterDict(
            r_cut=OnlyTypes(float, allow_none=False),
            nlist=OnlyTypes(hoomd.md.nlist.NeighborList, strict=True, allow_none=True),
            groups=list(groups),
            max_bonds =list(max_bonds),
            bond_type=OnlyTypes(int, strict=True, allow_none=False),
            seed=OnlyTypes(int, strict=True, allow_none=False),
            probability=OnlyTypes(float, strict=False,allow_none=True),
        )

        params.update(
            dict(
                r_cut=r_cut,
                nlist=nlist,
                groups=groups,
                max_bonds=max_bonds,
                bond_type=bond_type,
                seed=seed,
                probability=probability,
            )
        )

        self._param_dict.update(params)

    def _parse_groups(self,vec_groups,vec_max_bonds,sim):
        """Converts Groups depending if there is one or two given.

        Args:
            vec_groups (Sequence[filter]): A sequence of length 2 or 1 of type ``hoomd.filter.ParticleFilter``.
            vec_max_bonds(Sequence[int]): A sequence of length 2 or 1 of type ``int``.
        """
        try:
            l_vec = len(vec_groups)
        except:
            raise ValueError("Expected array of hoomd.filter.ParticleFilter for argument `groups`.")
        if l_vec == 1:
            if isinstance(vec_groups[0], hoomd.filter.ParticleFilter):
                self.group_1 = sim.state._get_group(vec_groups[0])
                self.group_2 = sim.state._get_group(vec_groups[0])
                self._param_dict["group_1"] = self.group_1
                self._param_dict["group_2"] = self.group_2
                if len(vec_max_bonds) !=1:
                    raise ValueError("Expected array[int] of same length as `groups` for argument `max_bonds`.")
                else:
                    self.max_bonds_group_1 = vec_max_bonds[0]
                    self._param_dict["max_bonds_group_1"] = vec_max_bonds[0]
                    self.max_bonds_group_2 = vec_max_bonds[0]
                    self._param_dict["max_bonds_group_2"] = vec_max_bonds[0]
            else:
                raise ValueError("Expected array of hoomd.filter.ParticleFilter for argument `groups`.")
        elif l_vec == 2:
            if isinstance(vec_groups[0], hoomd.filter.ParticleFilter) and\
               isinstance(vec_groups[1], hoomd.filter.ParticleFilter):
                self.group_1 = sim.state._get_group(vec_groups[0])
                self.group_2 = sim.state._get_group(vec_groups[1])
                self._param_dict["group_1"] = self.group_1
                self._param_dict["group_2"] = self.group_2
                if len(vec_max_bonds) != 2:
                    raise ValueError("Expected array[int] of same length as `groups` for argument `max_bonds`.")
                else:
                    self.max_bonds_group_1 = vec_max_bonds[0]
                    self._param_dict["max_bonds_group_1"] = vec_max_bonds[0]
                    self.max_bonds_group_2 = vec_max_bonds[1]
                    self._param_dict["max_bonds_group_2"] = vec_max_bonds[1]
            else:
                raise ValueError("Expected array of hoomd.filter.ParticleFilter for argument `groups`.")
        else:
            raise ValueError(
                "Expected an array of one or two hoomd.filter.ParticleFilter for argument `groups`.")

    @property
    def bond_type(self):
        """bond_type (int): Type of the bonds that are made by this updater."""
        return self._cpp_obj.bond_type

    @bond_type.setter
    def bond_type(self, value):
        if value is not None:
            self._param_dict["bond_type"] = value
            self._cpp_obj.setBondType(value)

    @property
    def probability(self):
        """Probability (float): Probability of forming bonds. Value between 0 and 1."""
        return self._cpp_obj.probability

    @probability.setter
    def probability(self, value):
        self._param_dict["probability"] = value
        self._cpp_obj.probability = value

    @property
    def r_cut(self):
        """r_cut (float): Distance cutoff for making bonds between particles."""
        return self._cpp_obj.r_cut

    @r_cut.setter
    def r_cut(self, value):
        self._cpp_obj.r_cut = value
        self._param_dict["r_cut"] = value

    def _attach_hook(self):
        """Create the c++ mirror class."""

        sim = self._simulation

        if isinstance(self._simulation.device, hoomd.device.CPU):
            cpp_class = getattr(self._ext_module, self._cpp_class_name)
        else:
            cpp_class = getattr(self._ext_module, self._cpp_class_name + "GPU")

        self._parse_groups(self.groups,self.max_bonds,sim)

        self._cpp_obj = cpp_class(
            sim.state._cpp_sys_def,
            self.trigger,
            self.group_1,
            self.group_2,
            self.seed,
            self.r_cut,
            self.probability,
            self.max_bonds_group_1,
            self.max_bonds_group_2,
            self.bond_type
        )

        if self.nlist is not None:
            self.nlist._attach(sim)
            self._cpp_obj.setNlist(self.nlist._cpp_obj)


        super()._attach_hook()
