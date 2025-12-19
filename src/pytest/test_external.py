# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2025, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

import hoomd
import hoomd.azplugins
import pytest
import numpy


class CustomVariant(hoomd.variant.Variant):
    def __init__(self, z):
        super().__init__()
        self.z = float(z)

    def __call__(self, timestep):
        if timestep <= 1:
            z = self.z
        else:
            z = self.z - 1
        return z

    def _min(self):
        return self.z - 1

    def _max(self):
        return self.z


@pytest.fixture
def custom_variant_fixture():
    return CustomVariant(z=5.0)


@pytest.fixture
def integrator():
    ig = hoomd.md.Integrator(dt=0.0)
    nve = hoomd.md.methods.ConstantVolume(hoomd.filter.All())
    ig.methods = [nve]
    return ig


@pytest.mark.parametrize(
    "cls",
    [
        hoomd.azplugins.external.PlanarHarmonicBarrier,
        hoomd.azplugins.external.SphericalHarmonicBarrier,
    ],
    ids=[
        "PlanarHarmonicBarrier",
        "SphericalHarmonicBarrier",
    ],
)
class TestHarmonicBarrier:
    def test_create(
        self,
        simulation_factory,
        two_particle_snapshot_factory,
        integrator,
        custom_variant_fixture,
        cls,
    ):
        # create object
        barrier = cls(location=3.0)
        barrier.params["A"].update(dict(k=10.0, offset=0.5))

        # make simulation and attach compute to integrator
        sim = simulation_factory(two_particle_snapshot_factory())
        sim.operations.integrator = integrator
        integrator.forces.append(barrier)

        # make sure values are initial ones
        assert isinstance(barrier.location, hoomd.variant.Constant)
        assert barrier.location(0) == 3.0
        assert barrier.params["A"] == dict(k=10.0, offset=0.5)

        # make sure values did not change on run
        sim.run(0)
        assert isinstance(barrier.location, hoomd.variant.Constant)
        assert barrier.location(0) == 3.0
        assert barrier.params["A"] == dict(k=10.0, offset=0.5)


def test_spherical_harmonic_barrier(
    simulation_factory, integrator, custom_variant_fixture
):
    snap = hoomd.Snapshot()
    if snap.communicator.rank == 0:
        snap.configuration.box = [20, 20, 20, 0, 0, 0]
        snap.particles.N = 4
        snap.particles.types = ["A", "B"]
        snap.particles.position[:] = [
            [0, 0, 4.6],
            [0, 0, -5.4],
            [0, 5.6, 0],
            [6.6, 0, 0],
        ]
        snap.particles.typeid[:] = [0, 1, 0, 0]
    sim = simulation_factory(snap)
    sim.operations.integrator = integrator

    barrier = hoomd.azplugins.external.SphericalHarmonicBarrier(
        location=custom_variant_fixture
    )
    kA = 50.0
    dB = 2.0
    kB = kA * dB**2
    barrier.params["A"] = dict(k=kA, offset=0.1)
    barrier.params["B"] = dict(k=kB, offset=-0.1)
    sim.operations.add(barrier)

    #  at first step the barrier will not move
    sim.run(1)
    forces = barrier.forces
    energies = barrier.energies
    if sim.device.communicator.rank == 0:
        # particle 0 is outside interaction range
        assert numpy.isclose(energies[0], 0.0)
        numpy.testing.assert_allclose(forces[0], [0, 0, 0], atol=1e-4)
        # particle 1 (type B) is experiencing the harmonic potential in +z
        assert numpy.isclose(energies[1], 0.5 * kB * 0.5**2, atol=1e-4)
        numpy.testing.assert_allclose(forces[1], [0.0, 0.0, kB * 0.5], atol=1e-4)
        # particle 2 (type A) is also experiencing the harmonic potential but in -y
        assert numpy.isclose(energies[2], 0.5 * kA * 0.5**2, atol=1e-4)
        numpy.testing.assert_allclose(forces[2], [0.0, -kA * 0.5, 0.0], atol=1e-4)
        # particle 3 (type A) is experiencing force in -x
        assert numpy.isclose(energies[3], 0.5 * kA * 1.5**2, atol=1e-4)
        numpy.testing.assert_allclose(forces[3], [-kA * 1.5, 0.0, 0.0], atol=1e-4)

    # disable B interactions and advance the simulation two steps so that now
    # barrier is at 4.0 in both verlet steps
    barrier.params["B"] = dict(k=0.0, offset=-0.1)
    sim.run(2)
    forces = barrier.forces
    energies = barrier.energies
    if sim.device.communicator.rank == 0:
        # particle 0 is now inside the harmonic region, -x
        assert numpy.isclose(energies[0], 0.5 * kA * 0.5**2)
        numpy.testing.assert_allclose(forces[0], [0.0, 0.0, -kA * 0.5], atol=1e-4)
        # particle 1 (type B) should now be ignored because of the K
        assert numpy.isclose(energies[1], 0.0, atol=1e-4)
        numpy.testing.assert_allclose(forces[1], [0, 0, 0], atol=1e-4)
        # particle 2 (type A) is 1.5 distance away from the barrier now
        assert numpy.isclose(energies[2], 0.5 * kA * 1.5**2, atol=1e-4)
        numpy.testing.assert_allclose(forces[2], [0.0, -kA * 1.5, 0.0], atol=1e-4)
        # particle 3 (type A) is still experiencing force in -x but with larger strength
        assert numpy.isclose(energies[3], 0.5 * kA * 2.5**2, atol=1e-4)
        numpy.testing.assert_allclose(forces[3], [-kA * 2.5, 0.0, 0.0], atol=1e-4)


def test_planar_harmonic_barrier(
    simulation_factory, integrator, custom_variant_fixture
):
    snap = hoomd.Snapshot()
    if snap.communicator.rank == 0:
        snap.configuration.box = [20, 20, 20, 0, 0, 0]
        snap.particles.N = 4
        snap.particles.types = ["A", "B"]
        snap.particles.position[:] = [
            [1, 4.6, 1],
            [-1, 5.4, 1],
            [1, 5.6, -1],
            [-1, 6.6, -1],
        ]
        snap.particles.typeid[:] = [0, 1, 0, 0]
    sim = simulation_factory(snap)
    sim.operations.integrator = integrator

    barrier = hoomd.azplugins.external.PlanarHarmonicBarrier(
        location=custom_variant_fixture
    )
    kA = 50.0
    dB = 2.0
    kB = kA * dB**2
    barrier.params["A"] = dict(k=kA, offset=0.1)
    barrier.params["B"] = dict(k=kB, offset=-0.1)
    sim.operations.add(barrier)

    #  at first step the barrier will not move
    sim.run(1)
    forces = barrier.forces
    energies = barrier.energies
    if sim.device.communicator.rank == 0:
        # particle 0 is outside interaction range
        assert numpy.isclose(energies[0], 0.0)
        numpy.testing.assert_allclose(forces[0], [0, 0, 0], atol=1e-4)
        # particle 1 (type B) is experiencing the harmonic potential
        assert numpy.isclose(energies[1], 0.5 * kB * 0.5**2, atol=1e-4)
        numpy.testing.assert_allclose(forces[1], [0.0, -kB * 0.5, 0.0], atol=1e-4)
        # particle 2 (type A) is also experiencing the harmonic potential
        assert numpy.isclose(energies[2], 0.5 * kA * 0.5**2, atol=1e-4)
        numpy.testing.assert_allclose(forces[2], [0.0, -kA * 0.5, 0.0], atol=1e-4)
        # particle 3 (type A) is experiencing the harmonic potential
        assert numpy.isclose(energies[3], 0.5 * kA * 1.5**2, atol=1e-4)
        numpy.testing.assert_allclose(
            forces[3],
            [0.0, -kA * 1.5, 0.0],
            atol=1e-4,
        )

    # disable B interactions and advance the simulation two steps so that now
    # barrier is at 4.0 in both verlet steps
    barrier.params["B"] = dict(k=0.0, offset=-0.1)
    sim.run(2)
    forces = barrier.forces
    energies = barrier.energies
    if sim.device.communicator.rank == 0:
        # particle 0 is now inside the harmonic region, -x
        assert numpy.isclose(energies[0], 0.5 * kA * 0.5**2, atol=1e-4)
        numpy.testing.assert_allclose(forces[0], [0.0, -kA * 0.5, 0.0], atol=1e-4)
        # particle 1 (type B) should now be ignored because of the K
        assert numpy.isclose(energies[1], 0.0, atol=1e-4)
        numpy.testing.assert_allclose(forces[1], [0, 0, 0], atol=1e-4)
        # particle 2 (type A) is 1.5 distance away from the barrier now
        assert numpy.isclose(energies[2], 0.5 * kA * 1.5**2, atol=1e-4)
        numpy.testing.assert_allclose(
            forces[2],
            [0.0, -kA * 1.5, 0.0],
            atol=1e-4,
        )
        # particle 3 (type A) is still experiencing force in -x but with larger strength
        assert numpy.isclose(energies[3], 0.5 * kA * 2.5**2, atol=1e-4)
        numpy.testing.assert_allclose(forces[3], [0.0, -kA * 2.5, 0.0], atol=1e-4)
