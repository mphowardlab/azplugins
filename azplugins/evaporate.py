# Copyright (c) 2016, Panagiotopoulos Group, Princeton University
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: mphoward

import hoomd
from hoomd import _hoomd
import _azplugins

class particles(hoomd.update._updater):
    def __init__(self, solvent, evaporated, lo, hi, seed, Nmax=False, period=1, phase=0):
        R""" Evaporate particles from a region.

        Args:
            solvent (str): Solvent particle type
            evaporated (str): Evaporated particle
            lo (float): *z* coordinate of evaporation region lower bound
            hi (float): *z* coordinate of evaporation region upper bound
            seed (int): Seed to the pseudo-random number generator
            Nmax (int): Maximum number of particles to evaporate
            period (int): Particle types will be updated every *period* time steps
            phase (int): When -1, start on the current time step. Otherwise, execute
                         on steps where *(step + phase) % period* is 0.

        Evaporate particles of type *solvent* from the slab defined by coordinates
        *lo* and *hi* along the *z*-azis. Every *period* time steps, the positions of all
        particles are checked. Up to *Nmax* particles of type *solvent* in the
        slab have their types changed to *evaporated*. If *Nmax* is None, all
        *solvent* particles have their types changed. Other particle types
        are ignored.

        .. note::

            The evaporation region must lie inside the simulation box. An
            error will be raised at runtime if the region lies outside the box.
            In simulations where the size of the simulation box changes, the
            size of the region is not rescaled, and could eventually end up
            outside the simulation box if not chosen appropriately.

        Evaporated particles are not actually removed from the simulation
        due to the inefficiency of resizing particle arrays frequently.
        All potential interactions for the *evaporated* type should be
        switched off to simulate full deletion of the particles from the box.
        Periodically, it is advisable to stop the simulation, remove the
        evaporated particles, and restart the simulation. This can be achieved
        efficiently using the snapshot API.

        .. warning::

            Because evaporated particles are not removed from the simulation box,
            the temperature, pressure, etc. reported by :py:class:`hoomd.compute.thermo`
            will not be meaningful. (Their degrees of freedom are still included
            in calculations.) This is OK because evaporation is a nonequilibrium
            process, and care should be taken in defining these quantities anyway.
            If necessary, make sure that you compute these properties in post-processing.

        Because of the nonequilibrium simulation, it is ill-advised to couple
        the entire system to a thermostat. However, evaporation can
        cause the system to cool throughout the simulation. It is recommended
        to couple another slab of the box to a Langevin thermostat to help
        control the temperature. See :py:class:`update.types` for an example
        of how to do this.

        The flux of particles out of the box is controlled by *Nmax*. If the
        simulation box has a cross sectional area *A*, then the flux
        *j* is:

        .. math::
            :nowrap:

            j = \frac{N_{\rm max}}{A \times \Delta t \times {\rm period}}

        It should be emphasized that this is the **maximum** flux attainable.
        If the evaporation process becomes diffusion-limited (there are fewer
        than *Nmax* particles in the evaporation region), then the actual
        flux obtained will be lower.

        Examples::

            azplugins.evaporate.particles(solvent='S', evaporated='Z', lo=-5.0, hi=5.0, seed=42, Nmax=5)
            azplugins.evaporate.particles(solvent='S', evaporated='Z', lo=-15.0, hi=-10.0, seed=77, period=10)

        """
        hoomd.update._updater.__init__(self)

        if not hoomd.context.exec_conf.isCUDAEnabled():
            cpp_class = _azplugins.ParticleEvaporator
        else:
            cpp_class = _azplugins.ParticleEvaporatorGPU
        self.cpp_updater = cpp_class(hoomd.context.current.system_definition, seed)
        self.setupUpdater(period, phase)

        self.metadata_fields = ['solvent','evaporated','lo','hi','seed','Nmax']
        self.seed = seed

        hoomd.util.quiet_status()
        self.set_params(solvent, evaporated, lo, hi, Nmax)
        hoomd.util.unquiet_status()

    def set_params(self, solvent=None, evaporated=None, lo=None, hi=None, Nmax=None):
        R""" Set the particle evaporation parameters.

        Args:
            solvent (str): Solvent particle type
            evaporated (str): Evaporated particle
            lo (float): *z* coordinate of evaporation region lower bound
            hi (float): *z* coordinate of evaporation region upper bound
            Nmax (int): Maximum number of particles to evaporate

        Examples::

            evap = azplugins.evaporate.particles(solvent='S', evaporated='Z', lo=-5.0, hi=5.0, Nmax=5)
            evap.set_params(solvent='Q', evaporated='R')
            updt.set_params(lo=-8.0)
            updt.set_params(hi=4.0, Nmax=10)

        """
        if solvent is not None:
            self.solvent = solvent
            try:
                type_id = hoomd.context.current.system_definition.getParticleData().getTypeByName(solvent)
            except RuntimeError:
                hoomd.context.msg.error('evaporate.particles: solvent type ' + self.solvent + ' not recognized\n')
                raise ValueError('evaporate.particles: solvent type ' + self.solvent + ' not recognized')
            self.cpp_updater.outside = type_id

        if evaporated is not None:
            self.evaporated = evaporated
            try:
                type_id = hoomd.context.current.system_definition.getParticleData().getTypeByName(evaporated)
            except RuntimeError:
                hoomd.context.msg.error('evaporate.particles: evaporated type ' + self.evaporated + ' not recognized\n')
                raise ValueError('evaporate.particles: evaporated type ' + self.evaporated + ' not recognized')
            self.cpp_updater.inside = type_id

        if self.solvent == self.evaporated:
            hoomd.context.msg.error('evaporate.particles: evaporated type (' + self.evaporated + ') cannot be the same as solvent type\n')
            raise ValueError('evaporate.particles: evaporated type (' + self.evaporated + ') cannot be the same as solvent type')

        if lo is not None:
            self.lo = lo
            self.cpp_updater.lo = lo

        if hi is not None:
            self.hi = hi
            self.cpp_updater.hi = hi

        if self.lo >= self.hi:
            hoomd.context.msg.error('evaporate.particles: lower z bound ' + str(self.lo) + ' >= upper z bound ' + str(self.hi) + '.\n')
            raise ValueError('evaporate.particles: upper and lower bounds are inverted')

        if Nmax is not None:
            self.Nmax = Nmax
            # cast Nmax to an integer if it is false
            if Nmax is False:
                Nmax = 0xffffffff
            self.cpp_updater.Nmax = Nmax
