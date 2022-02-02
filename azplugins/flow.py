# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2022, Auburn University
# This file is part of the azplugins project, released under the Modified BSD License.
"""
Flow
====

Flow fields and profilers

.. autosummary::
    :nosignatures:

    quiescent
    constant
    parabolic
    reverse_perturbation
    FlowProfiler

Flow integrators

.. autosummary::
    :nosignatures:

    brownian
    langevin

.. autoclass:: quiescent
.. autoclass:: constant
.. autoclass:: parabolic
.. autoclass:: reverse_perturbation
.. autoclass:: FlowProfiler
.. autoclass:: brownian
.. autoclass:: langevin

"""

import numpy as np
import hoomd
from hoomd import _hoomd

from . import _azplugins

class quiescent(object):
    r""" Quiescent fluid profile

    Example::

        u = azplugins.flow.quiescent()

    """
    def __init__(self):
        hoomd.util.print_status_line()
        self._cpp = _azplugins.QuiescentFluid()

    def __call__(self, r):
        """ Computes the velocity profile

        Args:
            r (list): Position to evaluate profile

        Example::

            >>> u = azplugins.flow.quiescent()
            >>> print u([0.0, 0.1, 0.3])
            (0,0,0)
            >>> print u((0.5, -0.2, 1.6))
            (0,0,0)

        """
        hoomd.util.print_status_line()

        u = self._cpp(_hoomd.make_scalar3(r[0], r[1], r[2]))
        return (u.x, u.y, u.z)

class constant(object):
    r""" Constant flow profile

    Args:
        U (tuple): Flow field.

    This flow corresponds to a constant vector field, e.g., a constant
    backflow in bulk or a plug flow in a channel. The flow field is
    independent of the position it is evaluated at.

    Example::

        u = azplugins.flow.constant(U=(1,0,0))

    """
    def __init__(self, U):
        hoomd.util.print_status_line()
        _U = _hoomd.make_scalar3(U[0],U[1],U[2])
        self._cpp = _azplugins.ConstantFlow(_U)

    def __call__(self, r):
        """ Computes the velocity profile

        Args:
            r (list): Position to evaluate profile

        Example::

            >>> u = azplugins.flow.constant(U=(1,2,3))
            >>> print u([0.0, 0.1, 0.3])
            (1,2,3)
            >>> print u((0.5, -0.2, 1.6))
            (1,2,3)

        """
        hoomd.util.print_status_line()

        u = self._cpp(_hoomd.make_scalar3(r[0], r[1], r[2]))
        return (u.x, u.y, u.z)

class parabolic(object):
    r""" Parabolic flow profile between parallel plates

    Args:
         U (float): Mean velocity
         H (float): Channel half-width

    This flow field generates the parabolic flow profile in a slit geomtry:

    .. math::

        u_x(z) = \frac{3}{2}U \left[1 - \left(\frac{z}{H}\right)^2 \right]

    The flow is along *x* with the gradient in *z*. The distance between the
    two plates is :math:`2H`, and the channel is centered around :math:`z=0`.
    The mean flow velocity is *U*.

    Example::

        u = azplugins.flow.parabolic(U = 2.0, H = 0.5)

    Note:
        Creating a flow profile does **not** imply anything about the simulation
        box boundaries. It is the responsibility of the user to construct
        appropriate bounding walls commensurate with the flow profile geometry.

    """
    def __init__(self, U, H):
        hoomd.util.print_status_line()
        self._cpp = _azplugins.ParabolicFlow(U, H)

    def __call__(self, r):
        """ Computes the velocity profile

        Args:
            r (list): Position to evaluate profile

        Example::

            >>> u = azplugins.flow.parabolic(U = 2.0, H = 0.5)
            >>> print u([0.0, 0.1, 0.3])
            (2,0,0)
            >>> print u((0.5, -0.2, 1.6))
            (0,0,0)

        """
        hoomd.util.print_status_line()

        u = self._cpp(_hoomd.make_scalar3(r[0], r[1], r[2]))
        return (u.x, u.y, u.z)

class brownian(hoomd.md.integrate._integration_method):
    R""" Brownian dynamics.

    Args:
        group (:py:mod:`hoomd.group`): Group of particles to apply this method to.
        kT (:py:mod:`hoomd.variant` or :py:obj:`float`): Temperature of the simulation (in energy units).
        flow (:py:class:`parabolic` or :py:class:`quiescent`): Flow profile
        seed (int): Random seed to use for generating :math:`\vec{F}_\mathrm{R}`.
        dscale (bool): Control :math:`\lambda` options. If 0 or False, use :math:`\gamma` values set per type. If non-zero, :math:`\gamma = \lambda d_i`.
        noiseless (bool): If set true, there will be no noise (random force)

    :py:class:`brownian` integrates particles forward in time according to the overdamped Langevin equations of motion,
    sometimes called Brownian dynamics, or the diffusive limit.

    .. math::

        \frac{d\vec{x}}{dt} = \vec{v}_0 + \frac{\vec{F}_\mathrm{C} + \vec{F}_\mathrm{R}}{\gamma}

        \langle \vec{F}_\mathrm{R} \rangle = 0

        \langle |\vec{F}_\mathrm{R}|^2 \rangle = 2 d k T \gamma / \delta t


    where :math:`\vec{F}_\mathrm{C}` is the force on the particle from all potentials and constraint forces,
    :math:`\gamma` is the drag coefficient, :math:`\vec{F}_\mathrm{R}`
    is a uniform random force, :math:`\vec{v}` is the particle's velocity, :math:`\vec{v}_0` is an
    imposed flow field and :math:`d` is the dimensionality
    of the system. The magnitude of the random force is chosen via the fluctuation-dissipation theorem
    to be consistent with the specified drag and temperature, :math:`T`.
    When :math:`kT=0`, the random force :math:`\vec{F}_\mathrm{R}=0`.

    :py:class:`brownian` generates random numbers by hashing together the particle tag, user seed, and current
    time step index. See `C. L. Phillips et. al. 2011 <http://dx.doi.org/10.1016/j.jcp.2011.05.021>`_ for more
    information.

    .. attention::
        Change the seed if you reset the simulation time step to 0. If you keep the same seed, the simulation
        will continue with the same sequence of random numbers used previously and may cause unphysical correlations.

    :py:class:`brownian` uses the integrator from `I. Snook, The Langevin and Generalised Langevin Approach to the Dynamics of
    Atomic, Polymeric and Colloidal Systems, 2007, section 6.2.5 <http://dx.doi.org/10.1016/B978-0-444-52129-3.50028-6>`_,
    with the exception that :math:`\vec{F}_\mathrm{R}` is drawn from a uniform random number distribution.

    .. warning::
        In Brownian dynamics, particle velocities are completely decoupled from positions
        and the particles are approximately massless. Accordingly, :py:class:`brownian` does
        not modify the velocities of particles. :py:class:`hoomd.compute.thermo` will
        **not** report appropriate temperatures and pressures if logged or needed by other
        commands. This behavior differs from the HOOMD Brownian dynamics implementation.

    Brownian dynamics neglects the acceleration term in the Langevin equation. This assumption is valid when
    overdamped: :math:`\frac{m}{\gamma} \ll \delta t`. Use :py:class:`langevin` if your system is not overdamped.

    You can specify :math:`\gamma` in two ways:

    1. Use :py:class:`set_gamma()` to specify it directly, with independent values for each particle type in the system.
    2. Specify :math:`\lambda` which scales the particle diameter to :math:`\gamma = \lambda d_i`. The units of
       :math:`\lambda` are mass / distance / time.

    :py:class:`brownian` must be used with integrate.mode_standard.

    *T* can be a variant type, allowing for temperature ramps in simulation runs.

    A :py:class:`hoomd.compute.thermo` is automatically created and associated with *group*.

    Examples::

        group_all = hoomd.group.all()
        u = azplugins.flow.parabolic(U=2.0, H=1.0)
        azplugins.flow.brownian(group=group_all, kT=1.0, flow=u, seed=5)
        azplugins.flow.brownian(group=group_all, kT=hoomd.variant.linear_interp([(0, 4.0), (1e6, 1.0)]), flow=u, seed=10)

    """
    def __init__(self, group, kT, flow, seed, dscale=False, noiseless=False):
        hoomd.util.print_status_line()

        # initialize base class
        super(brownian, self).__init__()

        # setup the variant inputs
        kT = hoomd.variant._setup_variant_input(kT)

        # create the compute thermo
        hoomd.compute._get_unique_thermo(group=group)

        if dscale is False or dscale == 0:
            use_lambda = False
        else:
            use_lambda = True

        # construct the correct flow field
        use_gpu = hoomd.context.exec_conf.isCUDAEnabled()
        if type(flow) is constant:
            if not use_gpu:
                cpp_class = _azplugins.BrownianConstantFlow
            else:
                cpp_class = _azplugins.BrownianConstantFlowGPU
        elif type(flow) is parabolic:
            if not use_gpu:
                cpp_class = _azplugins.BrownianParabolicFlow
            else:
                cpp_class = _azplugins.BrownianParabolicFlowGPU
        elif type(flow) is quiescent:
            if not use_gpu:
                cpp_class = _azplugins.BrownianQuiescentFluid
            else:
                cpp_class = _azplugins.BrownianQuiescentFluidGPU
        else:
            hoomd.context.msg.error('flow.brownian: flow field type not recognized\n')
            raise TypeError('Flow field type not recognized')

        self.cpp_method = cpp_class(hoomd.context.current.system_definition,
                                    group.cpp_group,
                                    kT.cpp_variant,
                                    flow._cpp,
                                    seed,
                                    use_lambda,
                                    float(dscale),
                                    noiseless)
        self.cpp_method.validateGroup()

        # store metadata
        self.group = group
        self.kT = kT
        self.flow = flow
        self.seed = seed
        self.dscale = dscale
        self.noiseless = noiseless
        self.metadata_fields = ['group', 'kT', 'seed', 'dscale','noiseless']

    def set_params(self, kT=None, flow=None, noiseless=None):
        R""" Change langevin integrator parameters.

        Args:
            kT (:py:mod:`hoomd.variant` or :py:obj:`float`): New temperature (if set) (in energy units).
            flow (object): Flow field object
            noiseless (bool): If true, do not apply the random noise in the equations of motion

        Examples::

            brownian.set_params(kT=2.0)

        Note:
            Because of the way flow fields are implemented, the type of *flow*
            is not permitted to change after the integrator is constructed.
            If you need to change the flow field type, disable the current
            integrator and create a new one.

        """
        hoomd.util.print_status_line()
        self.check_initialization()

        # change the parameters
        if kT is not None:
            # setup the variant inputs
            kT = hoomd.variant._setup_variant_input(kT)
            self.cpp_method.setT(kT.cpp_variant)
            self.kT = kT

        if flow is not None:
            if type(flow) is type(self.flow):
                self.cpp_method.setFlowField(flow._cpp)
                self.flow = flow
            else:
                hoomd.context.msg.error('flow.langevin: flow profile type cannot change after construction')
                raise TypeError('Flow profile type cannot change after construction')

        if noiseless is not None:
            self.cpp_method.setNoiseless(noiseless)
            self.noiseless = noiseless

    def set_gamma(self, a, gamma):
        R""" Set gamma for a particle type.

        Args:
            a (str): Particle type name
            gamma (float): :math:`\gamma` for particle type a (in units of force/velocity)

        :py:meth:`set_gamma()` sets the coefficient :math:`\gamma` for a single particle type, identified
        by name. The default is 1.0 if not specified for a type.

        It is not an error to specify gammas for particle types that do not exist in the simulation.
        This can be useful in defining a single simulation script for many different types of particles
        even when some simulations only include a subset.

        Examples::

            langevin.set_gamma('A', gamma=2.0)

        """
        hoomd.util.print_status_line()
        self.check_initialization()
        a = str(a)

        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes()
        type_list = []
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i))

        # change the parameters
        for i in range(0,ntypes):
            if a == type_list[i]:
                self.cpp_method.setGamma(i,gamma)

class langevin(hoomd.md.integrate._integration_method):
    R""" Langevin dynamics.

    Args:
        group (:py:mod:`hoomd.group`): Group of particles to apply this method to.
        kT (:py:mod:`hoomd.variant` or :py:obj:`float`): Temperature of the simulation (in energy units).
        flow (:py:class:`parabolic` or :py:class:`quiescent`): Flow profile
        seed (int): Random seed to use for generating :math:`\vec{F}_\mathrm{R}`.
        dscale (bool): Control :math:`\lambda` options. If 0 or False, use :math:`\gamma` values set per type. If non-zero, :math:`\gamma = \lambda d_i`.
        noiseless (bool): If set true, there will be no noise (random force)

    :py:class:`langevin` integrates particles forward in time according to the Langevin equations of motion:

    .. math::

        m \frac{d\vec{v}}{dt} = \vec{F}_\mathrm{C} - \gamma \cdot (\vec{v} - \vec{v}_0) + \vec{F}_\mathrm{R}

        \langle \vec{F}_\mathrm{R} \rangle = 0

        \langle |\vec{F}_\mathrm{R}|^2 \rangle = 2 d kT \gamma / \delta t

    where :math:`\vec{F}_\mathrm{C}` is the force on the particle from all potentials and constraint forces,
    :math:`\gamma` is the drag coefficient, :math:`\vec{v}` is the particle's velocity,
    :math:`\vec{v}_0` is an impose flow field, :math:`\vec{F}_\mathrm{R}`
    is a uniform random force, and :math:`d` is the dimensionality of the system (2 or 3).  The magnitude of
    the random force is chosen via the fluctuation-dissipation theorem
    to be consistent with the specified drag and temperature, :math:`T`.
    When :math:`kT=0`, the random force :math:`\vec{F}_\mathrm{R}=0`.

    :py:class:`langevin` generates random numbers by hashing together the particle tag, user seed, and current
    time step index. See `C. L. Phillips et. al. 2011 <http://dx.doi.org/10.1016/j.jcp.2011.05.021>`_ for more
    information.

    .. attention::
        Change the seed if you reset the simulation time step to 0. If you keep the same seed, the simulation
        will continue with the same sequence of random numbers used previously and may cause unphysical correlations.

    Langevin dynamics includes the acceleration term in the Langevin equation and is useful for gently thermalizing
    systems using a small gamma. This assumption is valid when underdamped: :math:`\frac{m}{\gamma} \gg \delta t`.
    Use :py:class:`brownian` if your system is not underdamped.

    :py:class:`langevin` uses the same integrator as :py:class:`nve` with the additional force term
    :math:`- \gamma \cdot \vec{v} + \vec{F}_\mathrm{R}`. The random force :math:`\vec{F}_\mathrm{R}` is drawn
    from a uniform random number distribution.

    You can specify :math:`\gamma` in two ways:

    1. Use :py:class:`set_gamma()` to specify it directly, with independent values for each particle type in the system.
    2. Specify :math:`\lambda` which scales the particle diameter to :math:`\gamma = \lambda d_i`. The units of
       :math:`\lambda` are mass / distance / time.

    :py:class:`langevin` must be used with :py:class:`mode_standard`.

    *T* can be a variant type, allowing for temperature ramps in simulation runs.

    A :py:class:`hoomd.compute.thermo` is automatically created and associated with *group*.

    Examples::

        group_all = hoomd.group.all()
        u = azplugins.flow.parabolic(U=2.0, H=1.0)
        azplugins.flow.langevin(group=group_all, kT=1.0, flow=u, seed=5)
        azplugins.flow.langevin(group=group_all, kT=hoomd.variant.linear_interp([(0, 4.0), (1e6, 1.0)]), flow=u, seed=10)

    """
    def __init__(self, group, kT, flow, seed, dscale=False, noiseless=False):
        hoomd.util.print_status_line()

        # initialize base class
        super(langevin, self).__init__()

        # setup the variant inputs
        kT = hoomd.variant._setup_variant_input(kT)

        # create the compute thermo
        hoomd.compute._get_unique_thermo(group=group)

        if dscale is False or dscale == 0:
            use_lambda = False
        else:
            use_lambda = True

        # construct the correct flow field
        use_gpu = hoomd.context.exec_conf.isCUDAEnabled()
        if type(flow) is constant:
            if not use_gpu:
                cpp_class = _azplugins.LangevinConstantFlow
            else:
                cpp_class = _azplugins.LangevinConstantFlowGPU
        elif type(flow) is parabolic:
            if not use_gpu:
                cpp_class = _azplugins.LangevinParabolicFlow
            else:
                cpp_class = _azplugins.LangevinParabolicFlowGPU
        elif type(flow) is quiescent:
            if not use_gpu:
                cpp_class = _azplugins.LangevinQuiescentFluid
            else:
                cpp_class = _azplugins.LangevinQuiescentFluidGPU
        else:
            hoomd.context.msg.error('flow.langevin: flow field type not recognized\n')
            raise TypeError('Flow field type not recognized')

        self.cpp_method = cpp_class(hoomd.context.current.system_definition,
                                    group.cpp_group,
                                    kT.cpp_variant,
                                    flow._cpp,
                                    seed,
                                    use_lambda,
                                    float(dscale),
                                    noiseless)
        self.cpp_method.validateGroup()

        # store metadata
        self.group = group
        self.kT = kT
        self.flow = flow
        self.seed = seed
        self.dscale = dscale
        self.noiseless = noiseless
        self.metadata_fields = ['group', 'kT', 'seed', 'dscale','noiseless']

    def set_params(self, kT=None, flow=None, noiseless=None):
        R""" Change langevin integrator parameters.

        Args:
            kT (:py:mod:`hoomd.variant` or :py:obj:`float`): New temperature (if set) (in energy units).
            flow (object): Flow field object
            noiseless (bool): If true, do not apply the random noise in the equations of motion

        Examples::

            langevin.set_params(kT=2.0)

        Note:
            Because of the way flow fields are implemented, the type of *flow*
            is not permitted to change after the integrator is constructed.
            If you need to change the flow field type, disable the current
            integrator and create a new one.

        """
        hoomd.util.print_status_line()
        self.check_initialization()

        # change the parameters
        if kT is not None:
            # setup the variant inputs
            kT = hoomd.variant._setup_variant_input(kT)
            self.cpp_method.setT(kT.cpp_variant)
            self.kT = kT

        if flow is not None:
            if type(flow) is type(self.flow):
                self.cpp_method.setFlowField(flow._cpp)
                self.flow = flow
            else:
                hoomd.context.msg.error('flow.langevin: flow profile type cannot change after construction')
                raise TypeError('Flow profile type cannot change after construction')

        if noiseless is not None:
            self.cpp_method.setNoiseless(noiseless)
            self.noiseless = noiseless

    def set_gamma(self, a, gamma):
        R""" Set gamma for a particle type.

        Args:
            a (str): Particle type name
            gamma (float): :math:`\gamma` for particle type a (in units of force/velocity)

        :py:meth:`set_gamma()` sets the coefficient :math:`\gamma` for a single particle type, identified
        by name. The default is 1.0 if not specified for a type.

        It is not an error to specify gammas for particle types that do not exist in the simulation.
        This can be useful in defining a single simulation script for many different types of particles
        even when some simulations only include a subset.

        Examples::

            langevin.set_gamma('A', gamma=2.0)

        """
        hoomd.util.print_status_line()
        self.check_initialization()
        a = str(a)

        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes()
        type_list = []
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i))

        # change the parameters
        for i in range(0,ntypes):
            if a == type_list[i]:
                self.cpp_method.setGamma(i,gamma)

class reverse_perturbation(hoomd.update._updater):
    R"""Reverse nonequilibrium shear flow in MD simulations.

    Implements an algorithm to generate shear flow, originally published by Mueller-Plathe:

    Florian Mueller-Plathe. "Reversing the perturbation in nonequilibrium
    molecular dynamics:  An easy way to calculate the shear viscosity of
    fluids." `Phys. Rev. E, 59:4894-4898, May 1999 <https://doi.org/10.1103/PhysRevE.59.4894>`_.

    The method swaps up to `Nswap` particle velocities every `period`
    timesteps to introduce a momentum flow.  While the swapping is
    unphysical, the resulting momentum flow is physical. The x component
    of the particle velocities are swapped in slabs in z-direction,
    creating a flow in x direction.

    Two distinct slabs in z direction of thickness `width` are chosen at
    -Lz/4 and +Lz/4.  The bottom slab at Lz/4 searched for up to `Nswap`
    particles for the x-component in momentum closest to the target momentum `target_momentum`
    in flow direction, the top slab is searched for up to `Nswap` particles with
    momentum x-component against flow direction closest to -target momentum `target_momentum`.
    Afterwards, both momentum components are swapped for up to `Nswap`
    particles.

    The velocity profile needs to be measured and can be influenced by
    picking the right values for `Nswap`, `target_momentum` and `period`. Too large flows
    will result in non-linear velocity profiles.

    The updater registers a variable to the logger called `rp_momentum`
    which saves the momentum excange for each execution. Because of the
    order of execution, the value reported in `rp_momentum` will be one
    time step out of sync. (The value reported on timestep 0 will
    always be zero, the value at timestep 1 will be the momentum exchanged
    on timestep 0, etc.)

    Args:
        group (:py:mod:`hoomd.group`): Group for which the update will be set
        Nswap (int): maximum number of particle pairs to swap velocities of.
        width (float): thickness of swapping slabs. `width` should be as
            small as possible for small disturbed volume, where the
            unphysical swapping is done. But each slab has to contain a
            sufficient number of particles. The default value is 1.0.
        target_momentum (float): target momentum for particles to pick.
        period (int): velocities are swapped every `period` of steps.
        phase (int): When -1, start on the current time step. When >= 0,
            execute on steps where *(step + phase) % period == 0*.
            The default value is 0.

    .. warning::

        This updater is intended to work with NVE integration only.

    Examples::

        f = azplugins.flow.reverse_perturbation(group=hoomd.group.all(),width=1.0,Nswap=1,period=100,target_momentum=1.0)
        f.set_period(1e3)
        f.disable()

    """
    def __init__(self,group,Nswap,period,target_momentum,width=1.0,phase=0):
        hoomd.util.print_status_line()

        # initialize the base class
        hoomd.update._updater.__init__(self)

        # Error out in MPI simulations
        if hoomd.comm.get_num_ranks() > 1:
            hoomd.context.msg.error("azplugins.flow.reverse_perturbation is not supported in multi-processor simulations.\n\n")
            raise RuntimeError("Error initializing updater.")

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_updater = _azplugins.ReversePerturbationFlow(hoomd.context.current.system_definition, group.cpp_group, Nswap, width,target_momentum)
        else:
            self.cpp_updater = _azplugins.ReversePerturbationFlowGPU(hoomd.context.current.system_definition, group.cpp_group, Nswap, width,target_momentum)

        self.setupUpdater(period,phase)

        self.metadata_fields = ['group','Nswap','width','period','phase','target_momentum']

        hoomd.util.quiet_status()
        self.set_params(group, Nswap, width,target_momentum)
        hoomd.util.unquiet_status()


    def set_params(self, group=None, Nswap=None, width=None, target_momentum=None):
        R""" Set the reverse perturbation flow updater parameters.

        Args:
            group (:py:mod:`hoomd.group`): Group for which the update will be set
            Nswap (int): maximum number of particle pairs to swap velocities of
            width (float): thickness of swapping slabs.
            target_momentum (float): target momentum for pairs to swap.

        Examples::

            f = azplugins.flow.reverse_perturbation(group=hoomd.group.all(),width=1.0,Nswap=1,period=100,target_momentum=1.0)
            f.set_params(width=0.5,Nswap=10)
            f.set_params(width=0.3)
            f.set_period(1000)

        """
        if group is not None:
            self.group = group
            self.cpp_updater.group = group.cpp_group

        if Nswap is not None:
            if Nswap<=0:
                hoomd.context.msg.error('reverse_perturbation.flow: Nswap: ' + str(Nswap) + ' needs to be bigger than zero.\n')
                raise ValueError('reverse_perturbation.flow: Nswap negative')
            self.Nswap = int(Nswap)
            self.cpp_updater.Nswap = int(Nswap)

        if width is not None:
            self.width = width
            self.cpp_updater.width = width

        if target_momentum is not None:
            if target_momentum<=0:
                hoomd.context.msg.error('reverse_perturbation.flow: target_momentum: ' + str(target_momentum) + ' needs to be bigger than zero.\n')
                raise ValueError('reverse_perturbation.flow: target_momentum negative')
            self.target_momentum = target_momentum
            self.cpp_updater.target_momentum =target_momentum

class FlowProfiler:
    R"""Measure average profiles along a spatial dimension.

    The average density, velocity, and temperature profiles are computed along
    a given spatial dimension. Both number and mass densities and velocities are
    available.

    Args:
        system: :py:mod:`hoomd` or :py:mod:`hoomd.mpcd` system (e.g., returned by :py:func:`hoomd.init.read_gsd`).
        axis (int): direction for binning (0=*x*, 1=*y*, 2=*z*).
        bins (int): Number of bins to use along ``axis``.
        range (tuple): Lower and upper spatial bounds to use along ``axis`` like ``(lo,hi)``.
        area (float): Cross-sectional area of bins to normalize density  (default: 1.0).

    Examples::

        f = azplugins.flow.FlowProfiler(system=system, axis=2, bins=20, range=(-10,10), area=20**2)
        hoomd.analyze.callback(f, period=1e3)
        hoomd.run(1e4)
        if hoomd.comm.get_rank() == 0:
            np.savetxt('profiles.dat', np.column_stack((f.centers, f.number_density, f.number_velocity[:,2], f.kT)))

    .. note::

        In MPI simulations, the profiles are **only** valid on the root rank. Accessing them on
        other ranks will give ``None``.

    .. note::

        If ``range`` is smaller than the box, paritcles outside of  ``range`` will be ignored.
        If ``range`` is larger than the box, the bins outside of the box will be filled with zeros.

    """
    def __init__(self, system, axis, bins, range, area=1.):
        self.system = system
        self.axis = axis

        # setup bins with edges that span the range
        self.edges = np.linspace(range[0], range[1], bins+1)
        self.centers = 0.5*(self.edges[:-1]+self.edges[1:])
        self._dx = self.edges[1:]-self.edges[:-1]
        self.area = area
        self.range = range
        self.bins = bins

        # profiles are initially empty
        self.reset()

        if self.axis not in (0,1,2):
            hoomd.context.msg.error('flow.FlowProfiler: axis needs to be 0, 1, or 2.\n')
            raise ValueError('Axis not recognized.')

    def __call__(self, timestep):
        r"""Evaluate the profiles at the current step.

        The profiles are computed for the current system. This call signature is
        intended for compatibility with `hoomd.analyze.callback`.

        Args:
            timestep (int): Current timestep.

        """
        hoomd.util.quiet_status()
        snap = self.system.take_snapshot()
        hoomd.util.unquiet_status()

        if hoomd.comm.get_rank() == 0:
            x = snap.particles.position[:,self.axis]
            v = snap.particles.velocity
            m = snap.particles.mass

            # bin particles, shifting by -1 due to numpy binning convention
            binids = np.digitize(x, self.edges)-1

            # filter particles outside of range
            flags = np.logical_and(binids > -1, binids < self.bins)
            binids = binids[flags]
            x = x[flags]
            v = v[flags]
            m = m[flags]

            # number density (counts)
            counts = np.bincount(binids,minlength=self.bins)
            self._counts += counts

            # mass density (mass)
            mass = np.bincount(binids,weights=m,minlength=self.bins)
            self._bin_mass += mass

            # velocity profiles
            # need to do each dimension separeately because of how bincount works
            num_vel = np.zeros((self.bins,3))
            mass_vel = np.zeros((self.bins,3))
            for dim in range(3):
               num_vel[:,dim] = np.bincount(binids,v[:,dim],minlength=self.bins)
               mass_vel[:,dim] = np.bincount(binids,m*v[:,dim],minlength=self.bins)
            np.divide(num_vel, counts[:,None], out=num_vel, where=counts[:,None]  > 0)
            np.divide(mass_vel, mass[:,None], out=mass_vel, where=mass[:,None] > 0)
            self._number_velocity += num_vel
            self._mass_velocity += mass_vel

            # temperature
            ke = np.bincount(binids, weights=0.5*m*np.sum(v**2,axis=1), minlength=self.bins)
            ke_cm = 0.5*mass*np.sum(mass_vel**2,axis=1)
            kT = np.zeros(self.bins)
            np.divide(2*(ke-ke_cm), 3*(counts-1), out=kT, where=counts > 1)
            self._kT += kT
            self.samples += 1

    def reset(self):
        r"""Reset the internal averaging counters."""
        self.samples = 0
        self._counts = np.zeros(self.bins)
        self._bin_mass = np.zeros(self.bins)
        self._number_density = np.zeros(self.bins)
        self._mass_density = np.zeros(self.bins)
        self._number_velocity = np.zeros((self.bins,3))
        self._mass_velocity = np.zeros((self.bins,3))
        self._kT = np.zeros(self.bins)

    @property
    def number_density(self):
        r"""The current average number density profile."""
        if hoomd.comm.get_rank() == 0 and self.samples > 0:
            return self._counts/(self._dx*self.area*self.samples)
        else:
            return None

    @property
    def mass_density(self):
        r"""The current mass-averaged density profile."""
        if hoomd.comm.get_rank() == 0 and self.samples > 0:
            return self._bin_mass/(self._dx*self.area*self.samples)
        else:
            return None

    @property
    def number_velocity(self):
        r"""The current number-averaged velocity profile."""
        if hoomd.comm.get_rank() == 0 and self.samples > 0:
            return self._number_velocity / self.samples
        else:
            return None

    @property
    def mass_velocity(self):
        r"""The current mass-averaged velocity profile."""
        if hoomd.comm.get_rank() == 0 and self.samples > 0:
            return self._mass_velocity / self.samples
        else:
            return None

    @property
    def kT(self):
        r"""The current average temperature profile."""
        if hoomd.comm.get_rank() == 0 and self.samples > 0:
            return self._kT / self.samples
        else:
            return None
