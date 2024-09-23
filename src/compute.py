# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2024, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""Flow profiles."""

import numpy
from hoomd.custom import Action


class FlowFieldProfiler(Action):
    """Measure average profiles in 3D.

    The average density, velocity, and temperature profiles are computed along
    a given spatial dimension. Both number and mass densities and velocities
    are available.

    Args:
        num_bins (int or 3-tuple): Number of bins in all three directions.
        bin_ranges ((3,2) array): Ranges for all three directions.

    Examples:
        flow_field = hoomd.azplugins.compute.FlowFieldProfiler(
            num_bins=(10, 10, 10),
            bin_ranges=([-L/2., L/2.], [-L/2., L/2.], [-L/2., L/2.])
        )
        flow_field_writer = hoomd.write.CustomWriter(
            action=flow_field,
            trigger=hoomd.trigger.Periodic(100)
        )
        simulation.operations.writers.append(flow_field_writer)
    """

    def __init__(self, num_bins, bin_ranges):
        super().__init__()
        self.bin_ranges = numpy.array(bin_ranges, dtype=float)

        if self.bin_ranges.shape != (3, 2):
            error_message = 'bin_ranges must be a (3,2) array'
            raise TypeError(error_message)

        if numpy.any(self.bin_ranges[:, 1] <= self.bin_ranges[:, 0]):
            error_message = 'Bin ranges must be increasing'
            raise ValueError(error_message)

        self.num_bins = numpy.array(num_bins, dtype=int)
        if self.num_bins.ndim == 0:
            self.num_bins = numpy.array([num_bins, num_bins, num_bins], dtype=int)

        if self.num_bins.shape != (3,):
            error_message = 'num_bins must be an int or a 3-tuple'
            raise TypeError(error_message)

        if not numpy.all(self.num_bins > 1):
            error_message = 'At least 1 bin required per dimension'
            raise ValueError(error_message)

        self.bin_sizes = (self.bin_ranges[:, 1] - self.bin_ranges[:, 0]) / self.num_bins
        self.bin_edges = [
            numpy.linspace(
                start=self.bin_ranges[dim, 0],
                stop=self.bin_ranges[dim, 1],
                num=self.num_bins[dim] + 1,
            )
            for dim in range(3)
        ]
        total_bins = numpy.prod(self.num_bins)
        self._counts = numpy.zeros(total_bins, dtype=int)
        self._velocity = numpy.zeros((total_bins, 3), dtype=float)
        self._num_samples = 0

    def act(self, timestep):
        """Execute the action of computing flow fields at the specified timestep."""
        with self._state.cpu_local_snapshot as snap:
            type_filter = snap.particles.typeid != 1
            pos = snap.particles.position[type_filter]
            vel = snap.particles.velocity[type_filter]

            in_range_filter = numpy.all(
                numpy.logical_and(
                    pos >= self.bin_ranges[:, 0], pos < self.bin_ranges[:, 1]
                ),
                axis=1,
            )
            binids = numpy.floor(
                (pos[in_range_filter] - self.bin_ranges[:, 0]) / self.bin_sizes
            ).astype(int)
            binids_1d = numpy.ravel_multi_index(binids.T, self.num_bins)
            numpy.add.at(self._counts, binids_1d, 1)
            numpy.add.at(self._velocity, binids_1d, vel[in_range_filter])

            self._num_samples += 1

    @property
    def density(self):
        """Compute density profile array based on bin counts and number of samples."""
        if self._num_samples > 0:
            bin_volume = numpy.prod(self.bin_sizes)
            density = self._counts / (self._num_samples * bin_volume)
            return numpy.reshape(density, self.num_bins)
        return numpy.zeros(self.num_bins)

    @property
    def velocity(self):
        """Compute and return the velocity profile as an array."""
        return numpy.divide(
            self._velocity,
            self._counts[:, None],
            out=numpy.zeros_like(self._velocity),
            where=self._counts[:, None] > 0,
        )

    def write(self, filename):
        """Write the computed density and velocity profiles to a file."""
        bin_centers = [
            0.5 * (self.bin_edges[dim][:-1] + self.bin_edges[dim][1:])
            for dim in range(3)
        ]
        bin_centers_mesh = numpy.meshgrid(*bin_centers, indexing='ij')
        numpy.savez(
            filename,
            X=bin_centers_mesh[0],
            Y=bin_centers_mesh[1],
            Z=bin_centers_mesh[2],
            density=self.density,
            velocity=self.velocity,
        )
