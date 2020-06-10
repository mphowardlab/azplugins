# Copyright (c) 2018-2020, Michael P. Howard
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: mphoward / Everyone is free to add additional updaters

import hoomd
from hoomd import _hoomd
from hoomd.md import _md


from . import _azplugins

class types(hoomd.update._updater):
    def __init__(self, inside, outside, lo, hi, period=1, phase=0):
        R""" Update particle types based on region

        Args:
            inside (str): Particle type name inside region
            outside (str): Particle type name outside region
            lo (float): *z* coordinate of region lower bound
            hi (float): *z* coordinate of region upper bound
            period (int): Particle types will be updated every *period* time steps
            phase (int): When -1, start on the current time step. Otherwise, execute
                         on steps where *(step + phase) % period* is 0.

        Updates particle types based on their position in a slab with coordinates
        along the *z*-axis. Every *period* time steps, the positions of all
        particles are checked. If particles of type *inside* are no longer in the
        slab, their types are changed to *outside*. If particles of type outside
        are in the slab, their types are changed to *inside*. Other particle types
        are ignored.

        .. note::

            The thermostatted region must lie inside the simulation box. An
            error will be raised at runtime if the region lies outside the box.
            In simulations where the size of the simulation box changes, the
            size of the region is not rescaled, and could eventually end up
            outside the simulation box if not chosen appropriately.

        .. warning::

            When multiple :py:class:`update.types` instances exist, no checks are
            made to guarantee that regions do not overlap each other. This may lead
            to unexpected behavior depending on the update order and particle types.
            It is the user's responsibility to ensure a sensible choice of regions is made.

        Examples::

            azplugins.update.types(inside='A', outside='B', lo=-5.0, hi=5.0)
            azplugins.update.types(inside='C', outside='B', lo=-15.0, hi=-10.0, period=10)

        :py:class:`update.types` may be useful in conjunction with the Langevin
        thermostat (:py:class:`hoomd.integrate.langevin`) for implementing a thermostatted
        region of the simulation box. Particles of type *inside* can have the
        thermostat applied (:math:`\gamma > 0`), while particles of type *outside*
        have :math:`\gamma = 0`.

        Thermostatted region example::

            azplugins.update.types(inside='A', outside='B', lo=-5.0, hi=5.0, period=10)
            ig = hoomd.md.integrate.langevin(group=all, kT=1.0, seed=5)
            ig.set_gamma('A', gamma=0.1)
            ig.set_gamma('B', gamma=0.0)

        """
        hoomd.util.print_status_line()

        hoomd.update._updater.__init__(self)

        if not hoomd.context.exec_conf.isCUDAEnabled():
            cpp_class = _azplugins.TypeUpdater
        else:
            cpp_class = _azplugins.TypeUpdaterGPU
        self.cpp_updater = cpp_class(hoomd.context.current.system_definition)
        self.setupUpdater(period, phase)

        self.metadata_fields = ['inside','outside','lo','hi']

        hoomd.util.quiet_status()
        self.set_params(inside, outside, lo, hi)
        hoomd.util.unquiet_status()

    def set_params(self, inside=None, outside=None, lo=None, hi=None):
        R""" Set the type updater parameters.

        Args:
            inside (str): Particle type name inside region
            outside (str): Particle type name outside region
            lo (float): *z* coordinate of region lower bound
            hi (float): *z* coordinate of region upper bound

        Examples::

            updt = azplugins.update.types(inside='A', outside='B', lo=-5.0, hi=5.0)
            updt.set_params(inside='B', outside='A')
            updt.set_params(lo=-8.0)
            updt.set_params(hi=4.0)

        """
        hoomd.util.print_status_line()

        if inside is not None:
            self.inside = inside
            try:
                type_id = hoomd.context.current.system_definition.getParticleData().getTypeByName(inside)
            except RuntimeError:
                hoomd.context.msg.error('update.type: inside type ' + self.inside + ' not recognized\n')
                raise ValueError('update.type: inside type ' + self.inside + ' not recognized')
            self.cpp_updater.inside = type_id

        if outside is not None:
            self.outside = outside
            try:
                type_id = hoomd.context.current.system_definition.getParticleData().getTypeByName(outside)
            except RuntimeError:
                hoomd.context.msg.error('update.type: outside type ' + self.outside + ' not recognized\n')
                raise ValueError('update.type: outside type ' + self.outside + ' not recognized')
            self.cpp_updater.outside = type_id

        if self.inside == self.outside:
            hoomd.context.msg.error('update.type: inside type (' + self.inside + ') cannot be the same as outside type\n')
            raise ValueError('update.type: inside type (' + self.inside + ') cannot be the same as outside type')

        if lo is not None:
            self.lo = lo
            self.cpp_updater.lo = lo

        if hi is not None:
            self.hi = hi
            self.cpp_updater.hi = hi

        if self.lo >= self.hi:
            hoomd.context.msg.error('update.type: lower z bound ' + str(self.lo) + ' >= upper z bound ' + str(self.hi) + '.\n')
            raise ValueError('update.type: upper and lower bounds are inverted')


class dynamic_bond(hoomd.update._updater):
    def __init__(self, nlist, r_cut,bond_type, bond_reservoir_type,group_1, group_2, max_bonds_1,max_bonds_2,period=1, phase=0):

        hoomd.util.print_status_line()

        hoomd.update._updater.__init__(self)
        self.nlist = nlist

        if not hoomd.context.exec_conf.isCUDAEnabled():
            cpp_class = _azplugins.DynamicBondUpdater
            self.nlist.cpp_nlist.setStorageMode(_md.NeighborList.storageMode.half)
        else:
            hoomd.context.msg.error('update.dynamic_bond not implemented on the GPU \n')
            raise ValueError('update.dynamic_bond not implemented on the GPU ')
            #cpp_class = _azplugins.TypeUpdaterGPU

        # look up the bond ids based on the given names - this will throw an error if the bond types do not exist
        bond_type_id = hoomd.context.current.system_definition.getBondData().getTypeByName(bond_type)
        bond_reservoir_type_id  = hoomd.context.current.system_definition.getBondData().getTypeByName(bond_reservoir_type)

        self.rcutsq = r_cut**2.0

        # we need to check that the groups have no overlap if the max_bonds_1 and max_bonds_2 are different
        new_cpp_group = _hoomd.ParticleGroup.groupIntersection(group_1.cpp_group, group_2.cpp_group)
        if new_cpp_group.getNumMembersGlobal()>0 and max_bonds_1 != max_bonds_2:
            hoomd.context.msg.error('update.dynamic_bond: groups are overlapping with ' + str(new_cpp_group.getNumMembersGlobal())
                                    + ' common members, but maximum bonds formed by each is different ' + str(max_bonds_1)
                                    + ' != '+  str(max_bonds_2)+ '.\n')
            raise ValueError('update.dynamic_bond: groups are overlapping with different number of maximum bonds')

        #it doesn't really make sense to allow partially overlapping groups?

        self.cpp_updater = cpp_class(hoomd.context.current.system_definition,
        self.nlist.cpp_nlist,group_1.cpp_group,group_2.cpp_group,self.rcutsq,bond_type_id,bond_reservoir_type_id,max_bonds_1,max_bonds_2)
        self.setupUpdater(period, phase)

        # how to do handling of exclusions in the neighborlist correctly?
        # neighbor list is ordered by particle id, does this create artifacts? (CPU)
        # what happens if bond reservoir is empty? should we throw a warning?

    def set_params(self, bond_type=None, max_bonds_1=None, max_bonds_2=None,group_1=None, group_2=None):
        # todo - cpp class right now doesn't have any set/get functions
        hoomd.util.print_status_line()
