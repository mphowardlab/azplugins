# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2024, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""Tests for the FlowProfile class."""

import hoomd
import hoomd.azplugins
import numpy
import pytest
from hoomd.azplugins.compute import FlowProfile


class TestFlowProfile:
    """Tests for the FlowProfile class."""

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

        self.flow_profile = FlowProfile(
            system=self.sim.state, axis=2, bins=10, range=(-5, 5), area=10**2
        )
        self.sim.operations.computes.append(self.flow_profile)

    def test_binning(self):
        """Test binning functionality."""
        self.sim.run(1)

        self.flow_profile(self.sim.timestep)

        expected_bins = numpy.linspace(-5, 5, 11)
        numpy.testing.assert_allclose(self.flow_profile.edges, expected_bins)
        expected_centers = 0.5 * (expected_bins[:-1] + expected_bins[1:])
        numpy.testing.assert_allclose(self.flow_profile.centers, expected_centers)

        bin_volume = 10 * 10 * 1.0
        expected_densities = numpy.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 2]) / bin_volume
        numpy.testing.assert_allclose(
            self.flow_profile.number_density, expected_densities
        )

        expected_velocities = [
            [2, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [(3.0 + 1.0) / 2, 0, 0],
        ]
        numpy.testing.assert_allclose(
            self.flow_profile.number_velocity, expected_velocities
        )

    def test_mass_averaging(self):
        """Test mass averaging functionality."""
        self.sim.run(1)

        self.flow_profile(self.sim.timestep)

        bin_volume = 10 * 10 * 1.0
        expected_densities = (
            numpy.array([0.5, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 1 + 0.5]) / bin_volume
        )
        numpy.testing.assert_allclose(
            self.flow_profile.mass_density, expected_densities
        )

        expected_velocities = [
            [2, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [(3.0 * 0.5 + 1.0 * 1.0) / (0.5 + 1.0), 0, 0],
        ]
        numpy.testing.assert_allclose(
            self.flow_profile.mass_velocity, expected_velocities
        )

    def test_temperature(self):
        """Test temperature calculation functionality."""
        self.sim.run(1)

        self.flow_profile(self.sim.timestep)

        expected_temperature = numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0.444444444])
        numpy.testing.assert_allclose(self.flow_profile.kT, expected_temperature)

    def test_missing_params(self):
        """Test missing parameters and invalid values."""
        with pytest.raises(TypeError, match='missing 1 required positional argument'):
            FlowProfile(system=self.sim.state, bins=10.5, range=(-5, 5), area=10**2)
        with pytest.raises(ValueError, match='Axis not recognized'):
            FlowProfile(
                system=self.sim.state, axis=3.0, bins=10, range=(-5, 5), area=10**2
            )
        with pytest.raises(TypeError, match="must be str, not 'int'"):
            FlowProfile(system=self.sim.state, axis='x')
        with pytest.raises(TypeError, match='missing 1 required positional argument'):
            FlowProfile(system=self.sim.state)
        with pytest.raises(ValueError, match='Axis not recognized'):
            FlowProfile(system=self.sim.state, axis=28, bins=10, range=(-5, 5))


if __name__ == '__main__':
    pytest.main(['-v', __file__])
