# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2022, Auburn University
# This file is part of the azplugins project, released under the Modified BSD License.

""" Updaters. """

import hoomd

from hoomd.azplugins import _azplugins
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import OnlyTypes

class dynamic_bond(hoomd.operation.Updater):
    R""" Update bonds dynamically during simulation.

    Args:
        r_cut (float): Distance cutoff for making bonds between particles
        probability (float): Probability of bond formation, between 0 and 1, default = 1
        bond_type (str): Type of bond to be formed
        group_1 (:py:mod: `hoomd.filter.ParticleFilter`): First particle group to form bonds between
        group_2 (:py:mod:`hoomd.filter.ParticleFilter`): Second particle group to form bonds between
        max_bonds_1 (int): Maximum number of bonds a particle in group_1 can have
        max_bonds_2 (int): Maximum number of bonds a particle in group_2 can have
        seed (int): Seed to the pseudo-random number generator
        nlist (:py:mod:`hoomd.md.nlist`): NeighborList (optional) for updating the exclusions
        period (int): Particle types will be updated every *period* time steps
        phase (int): When -1, start on the current time step. Otherwise, execute
                     on steps where *(step + phase) % period* is 0.

        Forms bonds of type bond_type between particles in group_1 and group_2 during
        the simulation, if particle distances are shorter than r_cut. If the neighborlist
        used for the pair potential in the simulation is given as a parameter nlist, the
        neighbor list exclusions will be updated to include the newly formed bonds.
        Each particle has a number of maximum bonds which it can form, given by
        max_bonds_1 for particles in group_1 and max_bonds_2 for group_2.

        The particles in the two groups group_1 and group_2 should be completely
        separate with no common elements, e.g. two different types, or the two
        groups should be identical, where now max_bonds_1 needs to be equal to max_bonds_2.


        .. warning::
            If the groups group_1 and group_2 are modified during the simulation, this
            Updater will not be updated to reflect the changes. It is the user's
            responsibility to ensure that the groups do not change as long as this
            updater is active.

        Examples::

            azplugins.update.dynamic_bond(nlist=nl,r_cut=1.0,probability=1.0, bond_type='bond',
                group_1=hoomd.group.type(type='A'),group_2=hoomd.group.type(type='B'),max_bonds_1=3,max_bonds_2=2)
    """
    _ext_module = _azplugins
    _cpp_class_name = "DynamicBondUpdater"

    def __init__(self,
                 trigger,
                 nlist,
                 group_1,
                 group_2,
                 bond_type=None,
                 max_bonds_group_1=None,
                 max_bonds_group_2=None,
                 r_cut=None,
                 seed=0,
                 probability=1):
        super().__init__(trigger)

        params = ParameterDict(
            r_cut=OnlyTypes(float, allow_none=True),
            nlist =OnlyTypes(hoomd.md.nlist.NeighborList,strict=True, allow_none=False),
            group_1=OnlyTypes(hoomd.filter.ParticleFilter, allow_none=False),
            group_2=OnlyTypes(hoomd.filter.ParticleFilter, allow_none=False),
            max_bonds_group_1=OnlyTypes(int, allow_none=True),
            max_bonds_group_2=OnlyTypes(int, allow_none=True),
            bond_type=OnlyTypes(int,strict=True,allow_none=True),
            seed = OnlyTypes(int,strict=True,allow_none=False),
            probability = float(probability)
        )

        params.update(
            dict(
                r_cut = r_cut,
                nlist=nlist,
                group_1=group_1,
                group_2=group_2,
                max_bonds_group_1=max_bonds_group_1,
                max_bonds_group_2=max_bonds_group_2,
                bond_type = bond_type,
                seed = seed,
                probability = probability
            )
        )

        self._param_dict.update(params)
        #self.set_params(r_cut,probability,bond_type,max_bonds_1,max_bonds_2,nlist)

    @property
    def bond_type(self):
        return self._cpp_obj.bond_type

    @bond_type.setter
    def bond_type(self,value):
        if value is not None:
            self._param_dict['bond_type']=value
            self._cpp_obj.setBondType(value)

    @property
    def probability(self):
        return self._cpp_obj.probability

    @probability.setter
    def probability(self,value):
        self._param_dict['probability']=value
        self._cpp_obj.probability = value

    @property
    def max_bonds_group_1(self):
        """
         max_bonds_1 (int)
        """
        return self._cpp_obj.max_bonds_group_1

    @max_bonds_group_1.setter
    def max_bonds_group_1(self, value):
        self._cpp_obj.max_bonds_group_1 = value
        self._param_dict['max_bonds_group_1']=value

    @property
    def r_cut(self):
        """
         r_cut (float): Distance cutoff for making bonds between particles
        """
        return self._cpp_obj.r_cut

    @r_cut.setter
    def r_cut(self, value):
        self._cpp_obj.r_cut = value
        self._param_dict['r_cut']=value

    @property
    def max_bonds_group_2(self):
        """
         max_bonds_group_2 (int)
        """
        return self._cpp_obj.max_bonds_group_2

    @max_bonds_group_2.setter
    def max_bonds_group_2(self, value):
        self._cpp_obj.max_bonds_group_2 = value
        self._param_dict['max_bonds_group_2']=value


    def _attach_hook(self):
        sim = self._simulation
        """Create the c++ mirror class."""
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cpp_class = getattr(self._ext_module, self._cpp_class_name)
        else:
            cpp_class = getattr(self._ext_module, self._cpp_class_name + "GPU")

        if self.group_1 is not None:
            group_1 = sim.state._get_group(self.group_1)
        else:
            group_1 = None

        if self.group_2 is not None:
            group_2 = sim.state._get_group(self.group_2)
        else:
            group_2 = None

        self.nlist._attach(sim)

        self._cpp_obj = cpp_class(
            sim.state._cpp_sys_def,
            self.trigger,
            self.nlist._cpp_obj,
            group_1,
            group_2,
            self.seed
        )

        super()._attach_hook()



