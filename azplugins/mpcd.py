# Copyright (c) 2018-2019, Michael P. Howard
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: astatt / Everyone is free to add additional methods for mpcd

import hoomd
from azplugins import _azplugins

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

