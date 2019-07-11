# Copyright (c) 2018-2020, Michael P. Howard
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: astatt / Everyone is free to add additional methods for mpcd

import hoomd

from . import _azplugins

class reverse_perturbation(hoomd.update._updater):
    R""" Updater class for a shear flow according to the algorithm
    published by Mueller-Plathe.:

    "Florian Mueller-Plathe. Reversing the perturbation innonequilibrium
    molecular dynamics:  An easy way to calculate the shear viscosity of
    fluids. Phys. Rev. E, 59:4894-4898, May 1999."
    <http://link.aps.org/pdf/10.1103/PhysRevE.59.4894>_

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
    picking the right values for `Nswap`,`target_momentum` and `period`.
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
        R""" Set the reverse pertubation flow updater parameters.

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


class sine(hoomd.mpcd.stream._streaming_method):
    r""" Parallel plate (slit) streaming geometry.

    Args:
        H (float): channel half-width
        V (float): wall speed (default: 0)
        boundary (str): boundary condition at wall ("slip" or "no_slip"")
        period (int): Number of integration steps between collisions

    The slit geometry represents a fluid confined between two infinite parallel
    plates. The slit is centered around the origin, and the walls are placed
    at :math:`z=-H` and :math:`z=+H`, so the total channel width is *2H*.
    The walls may be put into motion, moving with speeds :math:`-V` and
    :math:`+V` in the *x* direction, respectively. If combined with a
    no-slip boundary condition, this motion can be used to generate simple
    shear flow.

    The "inside" of the :py:class:`slit` is the space where :math:`|z| < H`.

    Examples::

        stream.slit(period=10, H=30.)
        stream.slit(period=1, H=25., V=0.1)

    .. versionadded:: 2.6

    """
    def __init__(self, H, V=0.0, boundary="no_slip", period=1):
        hoomd.util.print_status_line()

        hoomd.mpcd.stream._streaming_method.__init__(self, period)

        self.metadata_fields += ['H','V','boundary']
        self.H = H
        self.V = V
        self.boundary = boundary

        bc = self._process_boundary(boundary)

        # create the base streaming class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            stream_class = _azplugins.ConfinedStreamingMethodSine
        else:
            stream_class = _azplugins.ConfinedStreamingMethodGPUSine
        self._cpp = stream_class(hoomd.context.current.mpcd.data,
                                 hoomd.context.current.system.getCurrentTimeStep(),
                                 self.period,
                                 0,
                                 _azplugins.SineGeometry(H,V,bc))

    def set_filler(self, density, kT, seed, type='A'):
        r""" Add virtual particles to slit channel.

        Args:
            density (float): Density of virtual particles.
            kT (float): Temperature of virtual particles.
            seed (int): Seed to pseudo-random number generator for virtual particles.
            type (str): Type of the MPCD particles to fill with.

        The virtual particle filler draws particles within the volume *outside* the
        slit walls that could be overlapped by any cell that is partially *inside*
        the slit channel (between the parallel plates). The particles are drawn from
        the velocity distribution consistent with *kT* and with the given *density*.
        The mean of the distribution is zero in *y* and *z*, but is equal to the wall
        speed in *x*. Typically, the virtual particle density and temperature are set
        to the same conditions as the solvent.

        The virtual particles will act as a weak thermostat on the fluid, and so energy
        is no longer conserved. Momentum will also be sunk into the walls.

        Example::

            slit.set_filler(density=5.0, kT=1.0, seed=42)

        .. versionadded:: 2.6

        """
        hoomd.util.print_status_line()

        type_id = hoomd.context.current.mpcd.particles.getTypeByName(type)
        T = hoomd.variant._setup_variant_input(kT)

        if self._filler is None:
            if not hoomd.context.exec_conf.isCUDAEnabled():
                fill_class = _azplugins.SineGeometryFiller
            else:
                fill_class = _azplugins.SineGeometryFillerGPU
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
        """ Remove the virtual particle filler.

        Example::

            slit.remove_filler()

        .. versionadded:: 2.6

        """
        hoomd.util.print_status_line()

        self._filler = None

    def set_params(self, H=None, V=None, boundary=None):
        """ Set parameters for the slit geometry.

        Args:
            H (float): channel half-width
            V (float): wall speed (default: 0)
            boundary (str): boundary condition at wall ("slip" or "no_slip"")

        Changing any of these parameters will require the geometry to be
        constructed and validated, so do not change these too often.

        Examples::

            slit.set_params(H=15.0)
            slit.set_params(V=0.2, boundary="no_slip")

        .. versionadded:: 2.6

        """
        hoomd.util.print_status_line()

        if H is not None:
            self.H = H

        if V is not None:
            self.V = V

        if boundary is not None:
            self.boundary = boundary

        bc = self._process_boundary(self.boundary)
        self._cpp.geometry = _azplugins.SineGeometry(self.H,self.V,bc)
        if self._filler is not None:
            self._filler.setGeometry(self._cpp.geometry)
