# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2024, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""Flow profiles."""

import numpy
from hoomd.data.parameterdicts import ParameterDict
from hoomd.logging import log
from hoomd.custom import Action

class FlowProfile(Action):
    """Measure average profiles along a spatial dimension.

    The average density, velocity, and temperature profiles are computed along
    a given spatial dimension. Both number and mass densities and velocities
    are available.

    Args:
        axis (int): direction for binning (0=*x*, 1=*y*, 2=*z*).
        bins (int): Number of bins to use along ``axis``.
        range (tuple): Lower and upper spatial bounds to use along ``axis`` like
            ``(lo,hi)``.
        area (float): Cross-sectional area of bins to normalize density
            (default: 1.0).

    Examples::

        flow_profile = hoomd.azplugins.FlowProfile(
            axis=2, bins=20, range=(-10, 10), area=100)
        sim.operations.add(hoomd.write.CustomWriter(
            action=flow_profile, trigger=hoomd.trigger.Periodic(1000)))
        sim.run(1e4)
        if sim.device.communicator.rank == 0:
            numpy.savetxt('profiles.dat', numpy.column_stack((
                flow_profile.centers, flow_profile.number_density(),
                flow_profile.number_velocity()[:, 2], flow_profile.kt())))
    """

    def __init__(self, axis, bins, range_, area=1.0):
        super().__init__()
        self._param_dict = ParameterDict(
            axis=int(axis),
            bins=int(bins),
            range=tuple(range_),
            area=float(area),
        )
        self._param_dict.update(self._param_dict)

        self.edges = numpy.linspace(range_[0], range_[1], bins + 1)
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        self._dx = self.edges[1:] - self.edges[:-1]
        self.reset()

        if self._param_dict["axis"] not in (0, 1, 2):
            raise ValueError('Axis not recognized.')

    def attach(self, simulation):
        self._state = simulation.state
        self._simulation = simulation
        super().attach(simulation)

    def act(self, timestep):
        """Compute the flow profile at the given timestep."""
        state = self._state

        if self._simulation.device.communicator.rank == 0:
            snapshot = state.get_snapshot()
            if snapshot.communicator.rank == 0:
                positions = snapshot.particles.position
                velocities = snapshot.particles.velocity
                masses = snapshot.particles.mass

                x = positions[:, self._param_dict["axis"]]
                v = velocities
                m = masses

                binids = numpy.digitize(x, self.edges) - 1
                flags = numpy.logical_and(binids >= 0, binids < self._param_dict["bins"])
                binids = binids[flags]
                x = x[flags]
                v = v[flags]
                m = m[flags]

                counts = numpy.bincount(binids, minlength=self._param_dict["bins"])
                self._counts += counts

                mass = numpy.bincount(binids, weights=m, minlength=self._param_dict["bins"])
                self._bin_mass += mass

                num_vel = numpy.zeros((self._param_dict["bins"], 3))
                mass_vel = numpy.zeros((self._param_dict["bins"], 3))
                for dim in range(3):
                    num_vel[:, dim] = numpy.bincount(
                        binids, weights=v[:, dim], minlength=self._param_dict["bins"]
                    )
                    mass_vel[:, dim] = numpy.bincount(
                        binids, weights=m * v[:, dim], minlength=self._param_dict["bins"]
                    )
                numpy.divide(
                    num_vel, counts[:, None], out=num_vel, where=counts[:, None] > 0
                )
                numpy.divide(mass_vel, mass[:, None], out=mass_vel, where=mass[:, None] > 0)
                self._number_velocity += num_vel
                self._mass_velocity += mass_vel

                ke = numpy.bincount(
                    binids, weights=0.5 * m * numpy.sum(v**2, axis=1), minlength=self._param_dict["bins"]
                )
                ke_cm = 0.5 * mass * numpy.sum(mass_vel**2, axis=1)
                kT = numpy.zeros(self._param_dict["bins"])
                numpy.divide(2 * (ke - ke_cm), 3 * (counts - 1), out=kT, where=counts > 1)
                self._kT += kT
                self.samples += 1

    def reset(self):
        """Reset the internal averaging counters."""
        self.samples = 0
        self._counts = numpy.zeros(self._param_dict["bins"])
        self._bin_mass = numpy.zeros(self._param_dict["bins"])
        self._number_velocity = numpy.zeros((self._param_dict["bins"], 3))
        self._mass_velocity = numpy.zeros((self._param_dict["bins"], 3))
        self._kT = numpy.zeros(self._param_dict["bins"])

    @log(category='sequence', requires_run=True)
    def number_density(self):
        """Return the number density."""
        if self._simulation.device.communicator.rank == 0 and self.samples > 0:
            return self._counts / (self._dx * self._param_dict["area"] * self.samples)
        return None

    @log(category='sequence', requires_run=True)
    def mass_density(self):
        """Return the mass density."""
        if self._simulation.device.communicator.rank == 0 and self.samples > 0:
            return self._bin_mass / (self._dx * self._param_dict["area"] * self.samples)
        return None

    @log(category='sequence', requires_run=True)
    def number_velocity(self):
        """Return the number velocity."""
        if self._simulation.device.communicator.rank == 0 and self.samples > 0:
            return self._number_velocity / self.samples
        return None

    @log(category='sequence', requires_run=True)
    def mass_velocity(self):
        """Return the mass velocity."""
        if self._simulation.device.communicator.rank == 0 and self.samples > 0:
            return self._mass_velocity / self.samples
        return None

    @log(category='sequence', requires_run=True)
    def kt(self):
        """Return the temperature (kT)."""
        if self._simulation.device.communicator.rank == 0 and self.samples > 0:
            return self._kT / self.samples
        return None
