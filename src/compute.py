# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2024, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""Flow profiles."""

import hoomd
from hoomd import _hoomd
from hoomd.operation import Compute
from hoomd.logging import log
import numpy as np
from hoomd.data.parameterdicts import ParameterDict

class FlowProfile(Compute):
    R"""Measure average profiles along a spatial dimension.

    The average density, velocity, and temperature profiles are computed along
    a given spatial dimension. Both number and mass densities and velocities are
    available.

    Args:
        system (hoomd.Simulation): The HOOMD simulation object containing the system state.
        axis (int): direction for binning (0=*x*, 1=*y*, 2=*z*).
        bins (int): Number of bins to use along ``axis``.
        range (tuple): Lower and upper spatial bounds to use along ``axis`` like ``(lo,hi)``.
        area (float): Cross-sectional area of bins to normalize density  (default: 1.0).

    Examples::

        flow_profile = hoomd.azplugins.compute.FlowProfile(system=sim.state, axis=2, bins=20, range=(-10, 10), area=100)
        sim.operations.computes.append(flow_profile)
        sim.run(1e4)
        if sim.device.communicator.rank == 0:
            np.savetxt('profiles.dat', np.column_stack((flow_profile.centers, flow_profile.number_density, flow_profile.number_velocity[:, 2], flow_profile.kT)))
    """

    def __init__(self, system, axis, bins, range, area=1.0):
        super().__init__()
        param_dict = ParameterDict(
            system=system,
            axis=int(axis),
            bins=int(bins),
            range=tuple(range),
            area=float(area)
        )
        self._param_dict.update(param_dict)
        
        self.edges = np.linspace(range[0], range[1], bins + 1)
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        self._dx = self.edges[1:] - self.edges[:-1]
        self.reset()
        
        if self.axis not in (0, 1, 2):
            raise ValueError('Axis not recognized.')
        

    def __call__(self, timestep):

        state = self._simulation.state

        if self._simulation.device.communicator.rank == 0:
            positions = state.get_snapshot().particles.position
            velocities = state.get_snapshot().particles.velocity
            masses = state.get_snapshot().particles.mass

            x = positions[:, self.axis]
            v = velocities
            m = masses

            binids = np.digitize(x, self.edges) - 1
            flags = np.logical_and(binids >= 0, binids < self.bins)
            binids = binids[flags]
            x = x[flags]
            v = v[flags]
            m = m[flags]

            counts = np.bincount(binids, minlength=self.bins)
            self._counts += counts

            mass = np.bincount(binids, weights=m, minlength=self.bins)
            self._bin_mass += mass

            num_vel = np.zeros((self.bins, 3))
            mass_vel = np.zeros((self.bins, 3))
            for dim in range(3):
                num_vel[:, dim] = np.bincount(binids, weights=v[:, dim], minlength=self.bins)
                mass_vel[:, dim] = np.bincount(binids, weights=m * v[:, dim], minlength=self.bins)
            np.divide(num_vel, counts[:, None], out=num_vel, where=counts[:, None] > 0)
            np.divide(mass_vel, mass[:, None], out=mass_vel, where=mass[:, None] > 0)
            self._number_velocity += num_vel
            self._mass_velocity += mass_vel

            ke = np.bincount(binids, weights=0.5 * m * np.sum(v ** 2, axis=1), minlength=self.bins)
            ke_cm = 0.5 * mass * np.sum(mass_vel ** 2, axis=1)
            kT = np.zeros(self.bins)
            np.divide(2 * (ke - ke_cm), 3 * (counts - 1), out=kT, where=counts > 1)
            self._kT += kT
            self.samples += 1

    def reset(self):
        """Reset the internal averaging counters."""
        self.samples = 0
        self._counts = np.zeros(self.bins)
        self._bin_mass = np.zeros(self.bins)
        self._number_velocity = np.zeros((self.bins, 3))
        self._mass_velocity = np.zeros((self.bins, 3))
        self._kT = np.zeros(self.bins)

    @log(category='sequence')
    def number_density(self):
        r"""The current average number density profile."""
        if self._simulation.device.communicator.rank == 0 and self.samples > 0:
            return self._counts / (self._dx * self.area * self.samples)
        else:
            return None

    @log(category='sequence')
    def mass_density(self):
        r"""The current mass-averaged density profile."""
        if self._simulation.device.communicator.rank == 0 and self.samples > 0:
            return self._bin_mass / (self._dx * self.area * self.samples)
        else:
            return None

    @log(category='sequence')
    def number_velocity(self):
        r"""The current number-averaged velocity profile."""
        if self._simulation.device.communicator.rank == 0 and self.samples > 0:
            return self._number_velocity / self.samples
        else:
            return None

    @log(category='sequence')
    def mass_velocity(self):
        r"""The current mass-averaged velocity profile."""
        if self._simulation.device.communicator.rank == 0 and self.samples > 0:
            return self._mass_velocity / self.samples
        else:
            return None

    @log(category='sequence')
    def kT(self):
        r"""The current average temperature profile."""
        if self._simulation.device.communicator.rank == 0 and self.samples > 0:
            return self._kT / self.samples
        else:
            return None