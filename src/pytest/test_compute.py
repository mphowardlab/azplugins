# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2024, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""Compute unit tests."""

import hoomd
import numpy

import pytest


@pytest.mark.parametrize(
    "cls,lower_bounds,upper_bounds",
    [
        (
            hoomd.azplugins.compute.CylindricalVelocityField,
            (0, 0, -10),
            (10, 2 * numpy.pi, 10),
        ),
    ],
    ids=[
        "CylindricalVelocityField",
    ],
)
class TestVelocityField:
    def test_attach_detach(
        self,
        simulation_factory,
        two_particle_snapshot_factory,
        cls,
        lower_bounds,
        upper_bounds,
    ):
        # test creation of object
        field = cls(
            num_bins=[2, 0, 1], lower_bounds=lower_bounds, upper_bounds=upper_bounds
        )
        numpy.testing.assert_array_equal(field.num_bins, (2, 0, 1))
        numpy.testing.assert_array_almost_equal(field.lower_bounds, lower_bounds)
        numpy.testing.assert_array_almost_equal(field.upper_bounds, upper_bounds)
        assert field.filter is None
        assert field.include_mpcd_particles is False

        # computed quantities are inaccessible before a run
        with pytest.raises(hoomd.error.DataAccessError):
            getattr(field, "velocities")

        # make simulation and attach compute
        sim = simulation_factory(two_particle_snapshot_factory(L=20))
        sim.operations.add(field)
        assert len(sim.operations.computes) == 1
        sim.run(0)

        # make sure values did not change
        numpy.testing.assert_array_equal(field.num_bins, (2, 0, 1))
        numpy.testing.assert_array_almost_equal(field.lower_bounds, lower_bounds)
        numpy.testing.assert_array_almost_equal(field.upper_bounds, upper_bounds)
        assert field.filter is None
        assert field.include_mpcd_particles is False

        # computed quantities are now accessible
        vel = field.velocities
        assert vel.shape == (2, 1, 3)
        numpy.testing.assert_array_equal(vel, 0)

        # detach from simulation and test properties again
        sim.operations.remove(field)
        assert len(sim.operations.computes) == 0
        with pytest.raises(hoomd.error.DataAccessError):
            getattr(field, "velocities")

    def test_select_particles(
        self,
        simulation_factory,
        two_particle_snapshot_factory,
        cls,
        lower_bounds,
        upper_bounds,
    ):
        # only HOOMD particles
        sim = simulation_factory(two_particle_snapshot_factory(L=20))
        field = cls(
            num_bins=[2, 0, 1],
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            filter=hoomd.filter.All(),
        )
        assert field.filter is not None
        assert field.include_mpcd_particles is False
        sim.operations.add(field)
        sim.run(0)
        assert field.filter is not None
        assert field.include_mpcd_particles is False

        # only MPCD particles
        sim = simulation_factory(two_particle_snapshot_factory(L=20))
        field = cls(
            num_bins=[2, 0, 1],
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            include_mpcd_particles=True,
        )
        assert field.filter is None
        assert field.include_mpcd_particles is True
        sim.operations.add(field)
        sim.run(0)
        assert field.filter is None
        assert field.include_mpcd_particles is True

        # both
        sim = simulation_factory(two_particle_snapshot_factory(L=20))
        field = cls(
            num_bins=[2, 0, 1],
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            filter=hoomd.filter.All(),
            include_mpcd_particles=True,
        )
        assert field.filter is not None
        assert field.include_mpcd_particles is True
        sim.operations.add(field)
        sim.run(0)
        assert field.filter is not None
        assert field.include_mpcd_particles is True

    def test_binning_shape(
        self,
        simulation_factory,
        two_particle_snapshot_factory,
        cls,
        lower_bounds,
        upper_bounds,
    ):
        sim = simulation_factory(two_particle_snapshot_factory(L=20))

        # start with all bins on
        field = cls(
            num_bins=[2, 3, 4], lower_bounds=lower_bounds, upper_bounds=upper_bounds
        )
        sim.operations.add(field)
        sim.run(0)
        assert field.velocities.shape == (2, 3, 4, 3)
        assert field.coordinates.shape == (2, 3, 4, 3)

        # bin only 2 dimensions
        field.num_bins = [2, 3, 0]
        assert field.velocities.shape == (2, 3, 3)
        assert field.coordinates.shape == (2, 3, 2)

        field.num_bins = [2, 0, 4]
        assert field.velocities.shape == (2, 4, 3)
        assert field.coordinates.shape == (2, 4, 2)

        field.num_bins = [0, 3, 4]
        assert field.velocities.shape == (3, 4, 3)
        assert field.coordinates.shape == (3, 4, 2)

        # bin only 1 dimensions
        field.num_bins = [2, 0, 0]
        assert field.velocities.shape == (2, 3)
        assert field.coordinates.shape == (2,)

        field.num_bins = [0, 3, 0]
        assert field.velocities.shape == (3, 3)
        assert field.coordinates.shape == (3,)

        field.num_bins = [0, 0, 4]
        assert field.velocities.shape == (4, 3)
        assert field.coordinates.shape == (4,)

        # bin 0 dimensions (weird case)
        field.num_bins = [0, 0, 0]
        assert field.velocities.shape == (3,)
        assert field.coordinates is None


class TestCylindricalVelocityField:
    def _make_snapshot(self):
        snap = hoomd.Snapshot()
        if snap.communicator.rank == 0:
            snap.configuration.box = [20, 20, 20, 0, 0, 0]

            # HOOMD particle
            snap.particles.N = 1
            snap.particles.types = ["A"]
            snap.particles.position[0] = [1, 1, 0.1]
            snap.particles.velocity[0] = [-1, 1, 2]
            snap.particles.mass[0] = 5

            # MPCD particle
            snap.mpcd.N = 1
            snap.mpcd.types = ["A"]
            snap.mpcd.position[0] = [-0.5, -0.5, -0.1]
            snap.mpcd.velocity[0] = [-numpy.sqrt(4.5), -numpy.sqrt(4.5), -2]
            snap.mpcd.mass = 1
        return snap

    def test_basic(self, simulation_factory):
        sim = simulation_factory(self._make_snapshot())

        field = hoomd.azplugins.compute.CylindricalVelocityField(
            num_bins=[2, 3, 4],
            lower_bounds=(0, 0, -1),
            upper_bounds=(2, 3 * numpy.pi / 2, 1),
            filter=hoomd.filter.All(),
            include_mpcd_particles=True,
        )
        sim.operations.add(field)
        sim.run(0)

        # HOOMD particle is in one bin, MPCD is in another
        hoomd_velocity = [0, numpy.sqrt(2), 2]
        mpcd_velocity = [3, 0, -2]
        vel = field.velocities
        numpy.testing.assert_array_almost_equal(vel[1, 0, 2], hoomd_velocity)
        numpy.testing.assert_array_almost_equal(vel[0, 2, 1], mpcd_velocity)
        # remaining entries should be zero
        mask = numpy.ones(vel.shape, dtype=bool)
        mask[1, 0, 2] = False
        mask[0, 2, 1] = False
        numpy.testing.assert_array_equal(vel[mask], 0)

        # only bin in r
        field.num_bins = [2, 0, 0]
        numpy.testing.assert_array_almost_equal(
            field.velocities, [mpcd_velocity, hoomd_velocity]
        )

        # only bin in theta
        field.num_bins = [0, 3, 0]
        numpy.testing.assert_array_almost_equal(
            field.velocities, [hoomd_velocity, [0, 0, 0], mpcd_velocity]
        )

        # only bin in z
        field.num_bins = [0, 0, 4]
        numpy.testing.assert_array_almost_equal(
            field.velocities, [[0, 0, 0], mpcd_velocity, hoomd_velocity, [0, 0, 0]]
        )

        # reset bin counts and omit particles based on bounds
        field.num_bins = [1, 1, 1]
        field.lower_bounds = [1, numpy.pi, -1]
        field.upper_bounds = [2, 3 * numpy.pi / 2, 1]
        numpy.testing.assert_array_equal(field.velocities, 0)
