# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2022, Auburn University
# This file is part of the azplugins project, released under the Modified BSD License.
"""
Updaters
========

.. autosummary::
    :nosignatures:

    types
    dynamic_bond

.. autoclass:: types
.. autoclass:: dynamic_bond
"""
import hoomd
from hoomd import _hoomd
from hoomd.md import _md


from . import _azplugins

class types(hoomd.update._updater):
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
    def __init__(self, inside, outside, lo, hi, period=1, phase=0):
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
    R""" Update bonds dynamically during simulation.

    Args:
        r_cut (float): Distance cutoff for making bonds between particles
        probability (float): Probability of bond formation, between 0 and 1, default = 1
        bond_type (str): Type of bond to be formed
        group_1 (:py:mod:`hoomd.group`): First particle group to form bonds between
        group_2 (:py:mod:`hoomd.group`): Second particle group to form bonds between
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

            azplugins.update.dynamic_bond(nlist=nl,r_cut=1.0,bond_type='bond',
                group_1=hoomd.group.all(),group_2=hoomd.group.all(), max_bonds_1=3,max_bonds_2=3)
            azplugins.update.types(r_cut=1.0,probability=1.0, bond_type='bond',
                group_1=hoomd.group.type(type='A'),group_2=hoomd.group.type(type='B'),max_bonds_1=3,max_bonds_2=2)
    """

    def __init__(self,r_cut,bond_type,group_1, group_2, max_bonds_1,max_bonds_2,seed,probability=1,nlist=None,period=1, phase=0):

        hoomd.util.print_status_line()
        hoomd.update._updater.__init__(self)

        if not hoomd.context.exec_conf.isCUDAEnabled():
            cpp_class = _azplugins.DynamicBondUpdater
        else:
            cpp_class = _azplugins.DynamicBondUpdaterGPU

        self.cpp_updater = cpp_class(hoomd.context.current.system_definition,group_1.cpp_group,group_2.cpp_group,seed)

        self.metadata_fields = ['r_cut','probability','bond_type','group_1', 'group_2', 'max_bonds_1','max_bonds_2','nlist']
        self.setupUpdater(period, phase)

        hoomd.util.quiet_status()
        self.set_params(r_cut,probability,bond_type,max_bonds_1,max_bonds_2,nlist)
        hoomd.util.unquiet_status()


    def set_params(self, r_cut=None, probability=None, bond_type=None, max_bonds_1=None, max_bonds_2=None, nlist=None):
        R""" Set the dynamic_bond parameters.

        Args:
            r_cut (float): Distance cutoff for making bonds between particles
            probability (float): Probability of bond formation
            bond_type (str): Type of bond to be formed
            max_bonds_1 (int): Maximum number of bonds a particle in group_1 can have
            max_bonds_2 (int): Maximum number of bonds a particle in group_2 can have
            nlist (:py:mod:`hoomd.md.nlist`): NeighborList (optional) for updating the exclusions

        Examples::

            bonds =  azplugins.update.dynamic_bond(nlist=nl,r_cut=1.0,bond_type='bond',
                        group_1=hoomd.group.all(),group_2=hoomd.group.all(), max_bonds_1=3,max_bonds_2=3)
            bonds.set_params(r_cut=2.0)
            bonds.set_params(max_bonds_1=5,max_bonds_2=5)

        """
        if r_cut is not None:
            if r_cut <=0:
                hoomd.context.msg.error('update.dynamic_bond: cutoff ' + str(r_cut) + ' <=0 .\n')
                raise ValueError('update.dynamic_bond: cutoff is smaller or equal to zero.')
            self.r_cut = r_cut
            self.cpp_updater.r_cut = self.r_cut

        if probability is not None:
            if probability <=0:
                hoomd.context.msg.error('update.dynamic_bond: probability ' + str(probability) + ' <=0 .\n')
                raise ValueError('update.dynamic_bond: probability is smaller than zero.')
            if probability >1:
                hoomd.context.msg.error('update.dynamic_bond: probability ' + str(probability) + ' >1 .\n')
                raise ValueError('update.dynamic_bond: probability is larger than one.')
            self.probability = probability
            self.cpp_updater.probability = self.probability

        if bond_type is not None:
            # look up the bond id based on the given name - this will throw an error if the bond type does not exist
            bond_type_id = hoomd.context.current.system_definition.getBondData().getTypeByName(bond_type)
            self.bond_type_id = bond_type_id
            self.cpp_updater.bond_type = self.bond_type_id

        if max_bonds_1 is not None:
            if max_bonds_1 <=0:
                hoomd.context.msg.error('update.dynamic_bond: number of maximum bonds for group 1 is ' + str(max_bonds_1) + ' <=0 .\n')
                raise ValueError('update.dynamic_bond: number of maximum bonds for group 1 is smaller or equal to zero.')
            self.max_bonds_1 = max_bonds_1
            self.cpp_updater.max_bonds_group_1 = self.max_bonds_1

        if max_bonds_2 is not None:
            if max_bonds_2 <=0:
                hoomd.context.msg.error('update.dynamic_bond: number of maximum bonds for group 2 is ' + str(max_bonds_2) + ' <=0 .\n')
                raise ValueError('update.dynamic_bond: number of maximum bonds for group 2 is smaller or equal to zero.')
            self.max_bonds_2 = max_bonds_2
            self.cpp_updater.max_bonds_group_2 = self.max_bonds_2

        if nlist is not None:
            self.nlist = nlist
            self.cpp_updater.setNeighbourList(self.nlist.cpp_nlist)
