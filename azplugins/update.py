# Copyright (c) 2018-2019, Michael P. Howard
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: mphoward / Everyone is free to add additional updaters

import hoomd

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
