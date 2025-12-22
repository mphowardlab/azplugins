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
    #_ext_module = _azplugins

    def __init__(self,trigger,r_cut,nlist,bond_type,group_1, group_2, max_bonds_1,max_bonds_2,seed,probability=1,period=1, phase=0):
        super().__init__(trigger)

        params = ParameterDict(
            r_cut=float(r_cut),
            nlist =OnlyTypes(hoomd.md.nlist.NeighborList,strict=True, allow_none=False),
            group_1=OnlyTypes(hoomd.filter.ParticleFilter, allow_none=True),
            group_2=OnlyTypes(hoomd.filter.ParticleFilter, allow_none=True),
            max_bonds_1=OnlyTypes(int, allow_none=True),
            max_bonds_2=OnlyTypes(int, allow_none=True),
            bond_type=OnlyTypes(str,strict=True),
            seed = int(seed),
            probability = float(probability)
        )
        params.update(
            dict(
                r_cut = r_cut,
                nlist=nlist,
                group_1=group_1,
                group_2=group_2,
                max_bonds_1=max_bonds_1,
                max_bonds_2=max_bonds_2,
                bond_type = bond_type,
                seed = seed,
                probability = probability
            )
        )
        print("parsed param dict")
        self._param_dict.update(params)


    def _attach_hook(self):
        sim = self._simulation
        print("in attach hook dynamic bonds")
        print(_azplugins)
        if isinstance(sim.device, hoomd.device.GPU):
            cpp_class = _azplugins.DynamicBondUpdaterGPU
        else:
            cpp_class = _azplugins.DynamicBondUpdater

        if self.group_1 is not None:
            group_1 = sim.state._get_group(self.group_1)
        else:
            group_1 = None

        if self.group_2 is not None:
            group_2 = sim.state._get_group(self.group_2)
        else:
            group_2 = None

        #todo: incomplete
        self._cpp_obj = cpp_class(
            sim.state._cpp_sys_def,
            group_1,
            group_2
        )

        super()._attach_hook()



