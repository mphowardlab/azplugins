# Copyright (c) 2016, Panagiotopoulos Group, Princeton University
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: mphoward

import hoomd
from hoomd import _hoomd
from hoomd.md import _md
import _azplugins

class implicit(hoomd.md.force._force):
    R""" Implicit model for particle evaporation

    Args:
        interface (:py:mod:`hoomd.variant` or :py:obj:`float`): *z* position of interface
        name (str): Name of the model instance

    An evaporating solvent front is modeled implicitly by a purely repulsive
    harmonic interface that pushes down on nonvolatile solutes. The potential
    is truncated at its minimum, and has the following form:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V(z) = & 0 & z < H \\
               & \frac{\kappa}{2} (z-H)^2 & H \le z < H_{\rm c} \\
               & \frac{\kappa}{2} (H_{\rm c} - H)^2 - F_g (z - H_{\rm c}) & z \ge H_{\rm c}
        \end{eqnarray*}

    Here, the interface is located at *z* height *H*, and may change with time.
    The effective interface position *H* may be modified per-particle-type using a *offset*
    (*offset* is added to *H* to determine the effective *H*).
    :math:`\kappa` is a spring constant setting the strength of the interface
    (:math:`\kappa` is a proxy for the surface tension). The harmonic potential
    is truncated above a height :math:`H_{\rm c} = H + \Delta`, at which point a
    constant force :math:`F_g` acts on the particle. This is meant to model the
    effect of gravity once the interface moves below the particle.

    Typically, good choices would be to set :math:`\kappa` to scale with the particle
    radius squared, :math:`\Delta` equal to the particle radius, and
    :math:`F_g = -\kappa \Delta` so that the potential is continued linearly
    (the force is continuous) and also that :math:`F_g` scales with the cube of the
    particle radius.

    The following coefficinets must be set per unique particle type:

    - :math:`\kappa` - *k* (energy per distance squared) - spring constant
    - *offset* (distance) - per-particle-type amount to shift *H*, default: 0.0
    - :math:`F_g` - *g* (force) - force to apply above :math:`H_{\rm c}`
    - :math:`\Delta` - *cutoff* (distance) - sets cutoff at :math:`H_{\rm c} = H + \Delta`

    .. note::
        If *cutoff* is set to None, False, or a negative number, the interaction is
        ignored for the particle type.

    Example::

        # moving interface from H = 100. to H = 50.
        interf = hoomd.variant.linear_interp([[0,100.],[1e6,50.]],zero=0)
        evap = azplugins.evaporate.implicit(interface=interf)

        # small particle has diameter 1.0
        evap.force_coeff.set('S', k=50.0, offset=0.0, g=50.0*0.5, cutoff=0.5)
        # big particle is twice as large (diameter 2.0), so all coefficients are scaled
        evap.force_coeff.set('B', k=50.0*2**2, offset=0.0, g=50.0*2**3/2., cutoff=1.0)

    .. warning::
        Virial calculation has not been implemented for this model because it is
        nonequilibrium. A warning will be raised if any calls to
        :py:class:`hoomd.analyze.log` are made because the logger always requests
        the virial flags. However, this warning can be safely ignored if the
        pressure (tensor) is not being logged or the pressure is not of interest.

    """
    def __init__(self, interface, name=""):
        hoomd.util.print_status_line()

        # initialize the base class
        hoomd.md.force._force.__init__(self,name)

        # setup the (moving) interface variant
        self.interface = hoomd.variant._setup_variant_input(interface)

        # setup the coefficient vector
        self.force_coeff = hoomd.md.external.coeff();
        self.force_coeff.set_default_coeff('offset', 0.0)
        self.required_coeffs = ['k','offset','g','cutoff']
        self.metadata_fields = ['force_coeff','interface']

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            cpp_class = _azplugins.ImplicitEvaporator
        else:
            cpp_class = _azplugins.ImplicitEvaporatorGPU
        self.cpp_force = cpp_class(hoomd.context.current.system_definition, self.interface.cpp_variant)

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

    def update_coeffs(self):
        # check that the force coefficients are valid
        if not self.force_coeff.verify(self.required_coeffs):
           hoomd.context.msg.error("Not all force coefficients are set\n")
           raise RuntimeError("Error updating force coefficients")

        # set all the params
        pdata = hoomd.context.current.system_definition.getParticleData()
        ntypes = pdata.getNTypes()
        type_list = []
        for i in range(0,ntypes):
            type_list.append(pdata.getNameByType(i))

        for i in range(0,ntypes):
            coeff = {}
            for c in self.required_coeffs:
                coeff[c] = self.force_coeff.get(type_list[i], c)

            if coeff['cutoff'] is None or coeff['cutoff'] is False:
                coeff['cutoff'] = -1.0

            self.cpp_force.setParams(i, coeff['k'], coeff['offset'], coeff['g'], coeff['cutoff'])

    def get_metadata(self):
        data = hoomd.md.force._force.get_metadata(self)

        # make sure coefficients are up-to-date
        self.update_coeffs()

        data['force_coeff'] = self.force_coeff
        data['interface'] = self.interface

        return data

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
