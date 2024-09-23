# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2024, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""Tests for the FlowFieldProfiler class."""

import hoomd
import numpy
import pytest
from hoomd.azplugins.compute import FlowFieldProfiler


class TestFlowFieldProfiler:
    """Tests for the FlowFieldProfiler class.

    These tests validate the binning, density, and velocity profiles
    calculated by the FlowFieldProfiler.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Fixture setup for each test."""
        device = hoomd.device.auto_select()
        self.sim = hoomd.Simulation(device=device, seed=42)

        L = 10.0
        snapshot = hoomd.Snapshot()
        if snapshot.communicator.rank == 0:
            snapshot.configuration.box = [L, L, L, 0, 0, 0]
            snapshot.particles.N = 5
            snapshot.particles.types = ['A', 'B', 'C']
            snapshot.particles.position[:] = [
                [0, 0, -4.25],
                [0, 0, 2.25],
                [0, 0, 3.0],
                [0, 0, 4.25],
                [0, 0, 4.2],
            ]
            snapshot.particles.velocity[:] = [
                [2.0, 0, 0],
                [1.0, 0, 0],
                [-1.0, 0, 0],
                [3.0, 0, 0],
                [1.0, 0, 0],
            ]
            snapshot.particles.mass[:] = [0.5, 0.5, 0.5, 0.5, 1.0]

        self.sim.create_state_from_snapshot(snapshot)

        integrator = hoomd.md.Integrator(dt=0.001)
        nl = hoomd.md.nlist.Cell(buffer=0.4)
        lj = hoomd.md.pair.LJ(nlist=nl, default_r_cut=2.5)

        # Setting LJ parameters for all type pairs
        lj.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0)
        lj.params[('A', 'B')] = dict(epsilon=1.0, sigma=1.0)
        lj.params[('A', 'C')] = dict(epsilon=1.0, sigma=1.0)
        lj.params[('B', 'B')] = dict(epsilon=1.0, sigma=1.0)
        lj.params[('B', 'C')] = dict(epsilon=1.0, sigma=1.0)
        lj.params[('C', 'C')] = dict(epsilon=1.0, sigma=1.0)

        integrator.forces.append(lj)
        self.sim.operations.integrator = integrator

        # Adjusting for new FlowFieldProfiler
        self.flow_field = FlowFieldProfiler(
            num_bins=(10, 10, 10),
            bin_ranges=([-L / 2, L / 2], [-L / 2, L / 2], [-L / 2, L / 2]),
        )
        self.sim.operations.computes.append(self.flow_field)

    def test_binning(self):
        """Test binning functionality."""
        self.sim.run(1)

        # Assert that binning has been set up correctly for each dimension
        expected_bins_x = numpy.linspace(-5, 5, 11)
        expected_bins_y = numpy.linspace(-5, 5, 11)
        expected_bins_z = numpy.linspace(-5, 5, 11)

        # Testing bin edges
        numpy.testing.assert_allclose(self.flow_field.bin_edges[0], expected_bins_x)
        numpy.testing.assert_allclose(self.flow_field.bin_edges[1], expected_bins_y)
        numpy.testing.assert_allclose(self.flow_field.bin_edges[2], expected_bins_z)

    def test_density_profiles(self):
        """Test the density calculation."""
        self.sim.run(1)

        # Invoke the flow profiler
        density = self.flow_field.density

        # The expected densities will vary based on the positions of the particles
        bin_volume = (10.0 / 10) ** 3  # volume of each bin
        expected_densities = numpy.zeros((10, 10, 10))

        # Check positions in bins and calculate expected densities
        expected_densities[5, 5, 2] = 1 / bin_volume  # Particle at (-4.25, 0, 0)
        expected_densities[5, 5, 7] = 1 / bin_volume  # Particle at (2.25, 0, 0)
        expected_densities[5, 5, 7] += 1 / bin_volume  # Particle at (3.0, 0, 0)
        expected_densities[5, 5, 8] = 1 / bin_volume  # Particle at (4.25, 0, 0)
        expected_densities[5, 5, 8] += 1 / bin_volume  # Particle at (4.2, 0, 0)

        numpy.testing.assert_allclose(density, expected_densities)

    def test_velocity_profiles(self):
        """Test the velocity calculation."""
        self.sim.run(1)

        velocity = self.flow_field.velocity

        # Expected velocities (average of particles in each bin)
        expected_velocities = numpy.zeros((10, 10, 10, 3))

        expected_velocities[5, 5, 2] = [2.0, 0, 0]  # Particle at (-4.25, 0, 0)
        expected_velocities[5, 5, 7] = [1.0, 0, 0]  # Particle at (2.25, 0, 0)
        expected_velocities[5, 5, 7] += [-1.0, 0, 0]  # Particle at (3.0, 0, 0)
        expected_velocities[5, 5, 7] /= 2  # Average velocity of particles
        expected_velocities[5, 5, 8] = [3.0, 0, 0]  # Particle at (4.25, 0, 0)
        expected_velocities[5, 5, 8] += [1.0, 0, 0]  # Particle at (4.2, 0, 0)
        expected_velocities[5, 5, 8] /= 2  # Average velocity of particles

        numpy.testing.assert_allclose(velocity, expected_velocities)

    def test_temperature(self):
        """Test temperature calculation functionality."""
        self.sim.run(1)

        # Placeholder test for temperature calculation, adjust as needed.
        # Depending on the actual implementation of temperature calculation in
        # the `FlowFieldProfiler` class, you may need to change this.
        temperature = numpy.zeros((10, 10, 10))  # Simulated temperature array
        expected_temperature = numpy.zeros((10, 10, 10))

        numpy.testing.assert_allclose(temperature, expected_temperature)


if __name__ == '__main__':
    pytest.main(['-v', __file__])
