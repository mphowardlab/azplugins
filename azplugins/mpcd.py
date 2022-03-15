# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2022, Auburn University
# This file is part of the azplugins project, released under the Modified BSD License.
"""
MPCD methods
============

.. autosummary::
    :nosignatures:

    compute_velocity
    reverse_perturbation
    sinusoidal_channel
    sinusoidal_expansion_constriction

.. autoclass:: compute_velocity
.. autoclass:: reverse_perturbation
.. autoclass:: sinusoidal_channel
.. autoclass:: sinusoidal_expansion_constriction

"""


import hoomd

from . import _azplugins

class compute_velocity(hoomd.compute._compute):
    r"""Compute center-of-mass velocity of MPCD particles

    Args:
        suffix (str): Suffix to attach to logged quantities.

    This computes the center-of-mass velocity of all MPCD particles:

    .. math::

        \mathbf{v}_{\rm cm} = \dfrac{\sum_i m_i \mathbf{v}_i}{\sum_i m_i}

    where :math:`\mathbf{v}_i` is the velocity and and :math:`m_i` is the mass
    of particle *i* in the group. Note that because all MPCD particles currently
    have the same mass, this is equivalent to the number-average velocity.

    The components of the result are exposed as loggable quantities ``mpcd_vx``,
    ``mpcd_vy``, and ``mpcd_vz`` with ``suffix`` appended. By default,
    ``suffix`` is an empty string, but a custom suffix may be specified if needed
    to distinguish the logged quantity from something else. You can save these
    results using :py:class:`hoomd.analyze.log`.

    Example::

        azplugins.mpcd.compute_velocity()
        hoomd.analyze.log(filename='velocity.dat', quantities=['mpcd_vx'], period=10)

    """
    def __init__(self, suffix=None):
        hoomd.util.print_status_line()
        super().__init__()

        # create suffix for logged quantity
        if suffix is None:
            suffix = ''

        if suffix in self._used_suffixes:
            hoomd.context.msg.error('azplugins.mpcd.velocity: Suffix {} already used\n'.format(suffix))
            raise ValueError('Suffix {} already used for MPCD velocity'.format(suffix))
        else:
            self._used_suffixes.append(suffix)

        # create the c++ mirror class
        self.cpp_compute = _azplugins.MPCDVelocityCompute(hoomd.context.current.mpcd.data, suffix)
        hoomd.context.current.system.addCompute(self.cpp_compute, self.compute_name);

    _used_suffixes = []

class reverse_perturbation(hoomd.update._updater):
    R"""Reverse nonequilibrium shear flow in MPCD simulations.

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
    -Lz/4 and +Lz/4 (default) or at the distance specified by the parameter `H`.
    `H` represents  the distance of each slab from z=0, it is the
    half-width between the two slabs. The bottom slab is searched for up to `Nswap`
    particles with a x-velocity component close to the target momentum +`target_momentum`
    in flow direction, the top slab is searched for up to `Nswap` particles
    with  x-velocity component close to the negative target momentum -`target_momentum`
    against flow direction.
    Afterwards, both velocity components are swapped for up to `Nswap`
    particles.

    Please note that a slab distance other than half z-box size has very
    limited user cases, for example if walls are present in the system.
    For a system with periodic boundary conditions it does not make sense
    to change the slab distance  parameter `H` from its default value.

    The velocity profile needs to be measured and can be influenced by
    picking the right values for `Nswap`, `target_momentum` and `period`.
    Too large flows will result in non-linear velocity profiles.

    The updater registers a variable to the logger called `rp_momentum`
    which saves the momentum excange for each execution. Because of the
    order of execution, the value reported in `rp_momentum` will be one
    time step out of sync. (The value reported on timestep 0 will
    always be zero, the value at timestep 1 will be the momentum exchanged
    on timestep 0, etc.)

    Args:
        Nswap (int): maximum number of particle pairs to swap velocities of.
        width (float): thickness of swapping slabs. `width` should be as
            small as possible for small disturbed volume, where the
            unphysical swapping is done. But each slab has to contain a
            sufficient number of particles. The default value is 1.0.
        H (float) : if used, sets half width between slabs. This parameter needs be set to Lz/4
            in periodic boundary conditions (use H=None or leave H unset).
            A different `H` is only useful if walls are present in the system.
        period (int): velocities are swapped every `period` of steps.
        target_momentum (float): target momentum for the swapping slabs.
        phase (int): When -1, start on the current time step. When >= 0,
            execute on steps where *(step + phase) % period == 0*.
            The default value is 0.

    Examples::

        f= azplugins.mpcd.reverse_perturbation(group=hoomd.group.all(),width=1.0,Nswap=1,period=100,target_momentum=0.5)
        f.set_period(1e3)
        f.disable()

    """
    def __init__(self,Nswap,period,target_momentum,width=1.0,phase=0,H=None):
        hoomd.util.print_status_line()

        # initialize the base class
        hoomd.update._updater.__init__(self)

        # Error out in MPI simulations
        if hoomd.comm.get_num_ranks() > 1:
            hoomd.context.msg.error("azplugins.mpcd.reverse_perturbation is not supported in multi-processor simulations.\n\n")
            raise RuntimeError("Error initializing updater.")

        if H==None:
            system = hoomd.data.system_data(hoomd.context.current.system_definition)
            Lz = system.sysdef.getParticleData().getGlobalBox().getL().z
            H=Lz/4.0

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_updater = _azplugins.MPCDReversePerturbationFlow(hoomd.context.current.mpcd.data,  Nswap, width,H,target_momentum)
        else:
            self.cpp_updater = _azplugins.MPCDReversePerturbationFlowGPU(hoomd.context.current.mpcd.data, Nswap, width,H,target_momentum)

        self.setupUpdater(period,phase)

        self.metadata_fields = ['Nswap','width','period','phase','target_momentum','distance']

        hoomd.util.quiet_status()
        self.set_params( Nswap, width,target_momentum,H)
        hoomd.util.unquiet_status()


    def set_params(self, Nswap=None, width=None, target_momentum=None,H=None):
        R"""Set the reverse pertubation flow updater parameters.

        Args:
            Nswap (int): maximum number of particle pairs to swap velocities of
            width (float): thickness of swapping slabs.
            H (float): sets half width between slabs
            target_momentum (float): target momentum for the swapping slabs.

        Examples::

            f= azplugins.mpcd.reverse_pertubation(width=1.0,Nswap=1,period=100,target_momentum=0.5)
            f.set_params(width=0.5,Nswap=10)
            f.set_params(width=0.3)
            f.set_params(H=10)
            f.set_period(1000)

        """

        if Nswap is not None:
            if Nswap<=0:
                hoomd.context.msg.error('mpcd.reverse_perturbation: Nswap: ' + str(Nswap) + ' needs to be bigger than zero.\n')
                raise ValueError('mpcd.reverse_perturbation: Nswap negative')
            self.Nswap = int(Nswap)
            self.cpp_updater.Nswap = int(Nswap)

        if width is not None:
            if width<=0:
                hoomd.context.msg.error('mpcd.reverse_perturbation: H: ' + str(H) + ' needs to be bigger than zero.\n')
                raise ValueError('mpcd.reverse_perturbation: H negative')
            self.width = width
            self.cpp_updater.width = width

        if H is not None:
            if H<=0:
                hoomd.context.msg.error('mpcd.reverse_perturbation: H: ' + str(H) + ' needs to be bigger than zero.\n')
                raise ValueError('mpcd.reverse_perturbation: H negative')
            self.distance = H
            self.cpp_updater.distance = H

        if target_momentum is not None:
            if target_momentum<=0:
                hoomd.context.msg.error('mpcd.reverse_perturbation: target_momentum: ' + str(target_momentum) + ' needs to be bigger than zero.\n')
                raise ValueError('mpcd.reverse_perturbation: target_momentum negative')
            self.target_momentum = target_momentum
            self.cpp_updater.target_momentum = target_momentum


class sinusoidal_channel(hoomd.mpcd.stream._streaming_method):
    r"""Anti-symmetric Sinusoidal channel streaming geometry.

    Args:
        A (float): Amplitude of Cosine wall
        h (float): channel half-width
        p (int):   channel periodicity
        boundary (str): boundary condition at wall ("slip" or "no_slip"", default "no_slip")
        period (int): Number of integration steps between collisions

    The anti-symmetric cosine geometry represents a fluid confined between two
    walls described by a sinusoidal profile with equations
    :math: `(A*cos(2*pi*p*x/Lx) +/- 2h)`, where A is the amplitude, :math: `h` is
    the channel half-width, :math:`Lx` is the BoxDim in *x* direction,
    and :math: `p` is the period of the wall cosine. The channel is
    anti-symmetric around the origin in *z* direction. The cosines of top and
    bottom are running in parallel and create a "wavy" cannel with :math: `p`
    repetitions.

    The "inside" of the :py:class:`sinusoidal_channel` is the space where
    :math:`|z- cos(2*pi*p*x/Lx)| < h`.

    Examples::

        stream.sinusoidal_channel(A=30.,h=1.5, p=1)
        stream.sinusoidal_channel(A=25.,h=2,p=2, boundary="no_slip", period=10)

    """
    def __init__(self, A,h,p, V=0.0, boundary="no_slip", period=1):
        hoomd.util.print_status_line()

        hoomd.mpcd.stream._streaming_method.__init__(self, period)

        self.metadata_fields += ['L','A','h','p','boundary']
        self.A = A
        self.h = h
        self.p = p
        self.boundary = boundary

        bc = self._process_boundary(boundary)
        system = hoomd.data.system_data(hoomd.context.current.system_definition)
        Lx = system.sysdef.getParticleData().getGlobalBox().getL().x
        self.L = Lx
        # create the base streaming class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            stream_class = _azplugins.ConfinedStreamingMethodSinusoidalChannel
        else:
            stream_class = _azplugins.ConfinedStreamingMethodGPUSinusoidalChannel
            hoomd.context.msg.error('mpcd.sinusoidal_channel: no GPU support implemented!\n')
            raise RuntimeError('GPU support not implemented')

        self._cpp = stream_class(hoomd.context.current.mpcd.data,
                                 hoomd.context.current.system.getCurrentTimeStep(),
                                 self.period,
                                 0,
                                 _azplugins.SinusoidalChannel(Lx,A,h,p,bc))

    def set_filler(self, density, kT, seed, type='A'):
        r"""Add virtual particles to symmetric cosine channel.

        Args:
            density (float): Density of virtual particles.
            kT (float): Temperature of virtual particles.
            seed (int): Seed to pseudo-random number generator for virtual particles.
            type (str): Type of the MPCD particles to fill with.

        The virtual particle filler draws particles within the volume *outside* the
        wavy walls that could be overlapped by any cell that is partially *inside*
        the channel. The particles are drawn from the velocity distribution
        consistent with *kT* and with the given *density*.
        The mean of the distribution is zero in *x*, *y*, and *z*. Typically, the
        virtual particle density and temperature are set
        to the same conditions as the solvent.

        The virtual particles will act as a weak thermostat on the fluid, and so energy
        is no longer conserved. Momentum will also be sunk into the walls.

        Example::

            sinusoidal_channel.set_filler(density=5.0, kT=1.0, seed=42)

        """
        hoomd.util.print_status_line()

        if hoomd.comm.get_num_ranks() > 1:
            hoomd.context.msg.error('MPI support for filler not implemented\n')
            raise RuntimeError('MPI support not implemented')

        type_id = hoomd.context.current.mpcd.particles.getTypeByName(type)
        T = hoomd.variant._setup_variant_input(kT)

        if self._filler is None:
            if not hoomd.context.exec_conf.isCUDAEnabled():
                fill_class = _azplugins.SinusoidalChannelFiller
            else:
                fill_class = _azplugins.SinusoidalChannelFillerGPU
            self._filler = fill_class(hoomd.context.current.mpcd.data,
                                      density,
                                      type_id,
                                      T.cpp_variant,
                                      seed,
                                      self._cpp.geometry)
        else:
            self._filler.setDensity(density)
            self._filler.setType(type_id)
            self._filler.setTemperature(T.cpp_variant)
            self._filler.setSeed(seed)

    def remove_filler(self):
        """Remove the virtual particle filler.

        Example::

            sinusoidal_channel.remove_filler()

        """
        hoomd.util.print_status_line()

        self._filler = None

    def set_params(self, A=None, h=None, p=None, boundary=None):
        """Set parameters for the symmetric cosine geometry.

        Args:
            A (float): channel Amplitude
            h (float): channel half-width
            p (int): channel periodicity
            boundary (str): boundary condition at wall ("slip" or "no_slip"", default "no_slip")

        Changing any of these parameters will require the geometry to be
        constructed and validated, so do not change these too often.

        Examples::

            sinusoidal_channel.set_params(A=15.0)
            sinusoidal_channel.set_params(boundary="no_slip")

        """
        hoomd.util.print_status_line()

        if A is not None:
            self.A = A

        if h is not None:
            self.h = h

        if p is not None:
            if type(p) == type(1):
                self.p = p
            else:
                hoomd.context.msg.error('mpcd.stream.sinusoidal_channel: p needs to be an integer!\n')

        if boundary is not None:
            self.boundary = boundary

        bc = self._process_boundary(self.boundary)
        self._cpp.geometry = _azplugins.SinusoidalChannel(self.L,self.A,self.h,self.p,bc)
        if self._filler is not None:
            self._filler.setGeometry(self._cpp.geometry)

class sinusoidal_expansion_constriction(hoomd.mpcd.stream._streaming_method):
    r"""Symmetric Sinusoidal streaming geometry, i.e a constriction expansion channel.

    Args:
        H (float): channel half-width at its widest point
        h (float): channel half-width at its narrowest point
        p (int): channel periodicity
        boundary (str): boundary condition at wall ("slip" or "no_slip"", default "no_slip")
        period (int): Number of integration steps between collisions

    The symmetric cosine geometry represents a fluid confined between two walls
    described by a sinusoidal profile with equations
    :math:`+/-(A cos(2*pi*p*x/Lx) + A + h)`,
    where A = 0.5*(H-h) is the amplitude, :math:`Lx` is the BoxDim in *x*
    direction, and :math: `p` is the
    period of the wall cosine. The channel is axis-symmetric around the origin in
    *z* direction. The two symmetric cosine walls create a periodic series of
    :math: `p` constrictions and expansions. The parameter :math: `H` gives the
    channel half-width at its widest and :math: `h` is the channel half-width at
    its narrowest point.

    The "inside" of the :py:class:`sinusoidal_expansion_constriction` is the space where
    :math:`|z| < (A cos(2pi*p*x/Lx) + A + h)`.

    Examples::

        stream.sinusoidal_expansion_constriction(H=30.,h=1.5, p=1)
        stream.sinusoidal_expansion_constriction(H=25.,h=2,p=2, boundary="no_slip", period=10)

    """
    def __init__(self, H,h,p,boundary="no_slip", period=1):
        hoomd.util.print_status_line()

        hoomd.mpcd.stream._streaming_method.__init__(self, period)

        self.metadata_fields += ['L','H','h','p','boundary']
        self.H = H
        self.h = h
        self.p = p
        self.boundary = boundary

        bc = self._process_boundary(boundary)
        system = hoomd.data.system_data(hoomd.context.current.system_definition)
        Lx = system.sysdef.getParticleData().getGlobalBox().getL().x
        self.L = Lx
        # create the base streaming class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            stream_class = _azplugins.ConfinedStreamingMethodSinusoidalExpansionConstriction
        else:
            stream_class = _azplugins.ConfinedStreamingMethodGPUSinusoidalExpansionConstriction
            hoomd.context.msg.error('mpcd.stream.sinusoidal_expansion_constriction: no GPU support implemented!\n')
            raise RuntimeError('GPU support not implemented')

        self._cpp = stream_class(hoomd.context.current.mpcd.data,
                                 hoomd.context.current.system.getCurrentTimeStep(),
                                 self.period,
                                 0,
                                 _azplugins.SinusoidalExpansionConstriction(Lx,H,h,p,bc))

    def set_filler(self, density, kT, seed, type='A'):
        r"""Add virtual particles to symmetric cosine channel.

        Args:
            density (float): Density of virtual particles.
            kT (float): Temperature of virtual particles.
            seed (int): Seed to pseudo-random number generator for virtual particles.
            type (str): Type of the MPCD particles to fill with.

        The virtual particle filler draws particles within the volume *outside* the
        cosine walls that could be overlapped by any cell that is partially *inside*
        the channel. The particles are drawn from
        the velocity distribution consistent with *kT* and with the given *density*.
        The mean of the distribution is zero in *x*, *y*, and *z*. Typically, the
        virtual particle density and temperature are set
        to the same conditions as the solvent.

        The virtual particles will act as a weak thermostat on the fluid, and so energy
        is no longer conserved. Momentum will also be sunk into the walls.

        Example::

            sinusoidal_expansion_constriction.set_filler(density=5.0, kT=1.0, seed=42)

        """
        hoomd.util.print_status_line()

        if hoomd.comm.get_num_ranks() > 1:
            hoomd.context.msg.error('MPI support for filler not implemented\n')
            raise RuntimeError('MPI support not implemented')

        type_id = hoomd.context.current.mpcd.particles.getTypeByName(type)
        T = hoomd.variant._setup_variant_input(kT)

        if self._filler is None:
            if not hoomd.context.exec_conf.isCUDAEnabled():
                fill_class = _azplugins.SinusoidalExpansionConstrictionFiller
            else:
                fill_class = _azplugins.SinusoidalExpansionConstrictionFillerGPU
            self._filler = fill_class(hoomd.context.current.mpcd.data,
                                      density,
                                      type_id,
                                      T.cpp_variant,
                                      seed,
                                      self._cpp.geometry)
        else:
            self._filler.setDensity(density)
            self._filler.setType(type_id)
            self._filler.setTemperature(T.cpp_variant)
            self._filler.setSeed(seed)

    def remove_filler(self):
        """Remove the virtual particle filler.

        Example::

            sinusoidal_expansion_constriction.remove_filler()

        """
        hoomd.util.print_status_line()

        self._filler = None

    def set_params(self, H=None, h=None, p=None,boundary=None):
        """Set parameters for the symmetric cosine geometry.

        Args:
            H (float): channel half-width at its widest point
            h (float): channel half-width at its narrowest point
            p (int):   channel periodicity
            boundary (str): boundary condition at wall ("slip" or "no_slip"", default "no_slip")

        Changing any of these parameters will require the geometry to be
        constructed and validated, so do not change these too often.

        Examples::

            sinusoidal_expansion_constriction.set_params(H=15.0)
            sinusoidal_expansion_constriction.set_params(boundary="no_slip")

        """
        hoomd.util.print_status_line()

        if H is not None:
            self.H = H

        if h is not None:
            self.h = h

        if p is not None:
            if type(p) == type(1):
                self.p = p
            else:
                hoomd.context.msg.error('mpcd.stream.sinusoidal_expansion_constriction: p needs to be an integer!\n')

        if boundary is not None:
            self.boundary = boundary

        bc = self._process_boundary(self.boundary)
        self._cpp.geometry = _azplugins.SinusoidalExpansionConstriction(self.L,self.H,self.h,self.p,bc)
        if self._filler is not None:
            self._filler.setGeometry(self._cpp.geometry)
