# Copyright (c) 2016-2018, Panagiotopoulos Group, Princeton University
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: astatt / Everyone is free to add additional updaters

import hoomd
from hoomd import _hoomd
import _azplugins

class reverse_perturbation(hoomd.update._updater):
    R""" Updater class for a shear flow according to the algorithm
    published by Mueller-Plathe.:

    "Florian Mueller-Plathe. Reversing the perturbation innonequilibrium
    molecular dynamics:  An easy way to calculate the shear viscosity of
    fluids. Phys. Rev. E, 59:4894-4898, May 1999."
    <http://dx.doi.org/10.1103/PhysRevE.59.4894>_

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
