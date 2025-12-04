# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2025, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""Wall potential unit tests."""

import collections

import hoomd
import hoomd.azplugins
import numpy

import pytest

PotentialTestCase = collections.namedtuple(
    "PotentialTestCase",
    ["potential", "params", "wall", "position", "energy", "force"],
)

potential_tests = []

# Lennard Jones 9-3
potential_tests += [
    # test the calculation of force and potential for plane
    PotentialTestCase(
        hoomd.azplugins.wall.LJ93,
        {"sigma": 1.0, "epsilon": 1.0, "r_cut": 3.0, "r_extrap": 0.0},
        hoomd.wall.Plane(origin=(0, 0, 0), normal=(0, 0, 1)),
        (0, 0, 1.5),
        -0.2558,
        -0.5718,
    ),
    PotentialTestCase(
        hoomd.azplugins.wall.LJ93,
        {"sigma": 1.0, "epsilon": 1.0, "r_cut": 3.0, "r_extrap": 0.0},
        hoomd.wall.Plane(origin=(0, 0, 0), normal=(0, 0, 1)),
        (0, 0, 3.5),
        0,
        0,
    ),
    # test the calculation of force and potential for sphere
    PotentialTestCase(
        hoomd.azplugins.wall.LJ93,
        {"sigma": 1.0, "epsilon": 1.0, "r_cut": 3.0, "r_extrap": 0.0},
        hoomd.wall.Sphere(radius=5, origin=(0, 0, 0), inside=True),
        (0, 0, 3.5),
        -0.2558,
        0.5718,
    ),
    PotentialTestCase(
        hoomd.azplugins.wall.LJ93,
        {"sigma": 1.0, "epsilon": 1.0, "r_cut": 3.0, "r_extrap": 0.0},
        hoomd.wall.Sphere(radius=5, origin=(0, 0, 0), inside=True),
        (0, 0, 2.0),
        0,
        0,
    ),
    PotentialTestCase(
        hoomd.azplugins.wall.LJ93,
        {"sigma": 1.0, "epsilon": 1.0, "r_cut": 3.0, "r_extrap": 0.0},
        hoomd.wall.Sphere(radius=5, origin=(0, 0, 0), inside=True),
        (0, 0, 5.5),
        0,
        0,
    ),
    # test the calculation of force and potential for cylinder
    PotentialTestCase(
        hoomd.azplugins.wall.LJ93,
        {"sigma": 1.0, "epsilon": 1.0, "r_cut": 3.0, "r_extrap": 0.0},
        hoomd.wall.Cylinder(radius=5, origin=(0, 0, 0), axis=(1, 0, 0), inside=True),
        (0, 0, 3.5),
        -0.2558,
        0.5718,
    ),
    PotentialTestCase(
        hoomd.azplugins.wall.LJ93,
        {"sigma": 1.0, "epsilon": 1.0, "r_cut": 3.0, "r_extrap": 0.0},
        hoomd.wall.Cylinder(radius=5, origin=(0, 0, 0), axis=(1, 0, 0), inside=True),
        (0, 0, 5.5),
        0,
        0,
    ),
    PotentialTestCase(
        hoomd.azplugins.wall.LJ93,
        {"sigma": 1.0, "epsilon": 1.0, "r_cut": 3.0, "r_extrap": 0.0},
        hoomd.wall.Sphere(radius=5, origin=(0, 0, 0), inside=True),
        (0, 0, 2.0),
        0,
        0,
    ),
]


@pytest.mark.parametrize(
    "potential_test", potential_tests, ids=lambda x: x.potential.__name__
)
def test_energy_and_force(
    simulation_factory, one_particle_snapshot_factory, potential_test
):
    """Test energy and force evaluation."""
    # make one particle test configuration
    sim = simulation_factory(
        one_particle_snapshot_factory(position=potential_test.position, L=20)
    )

    # setup dummy NVE integrator
    integrator = hoomd.md.Integrator(dt=0.001)
    nve = hoomd.md.methods.ConstantVolume(hoomd.filter.All())
    integrator.methods = [nve]

    # setup wall potential
    wall = potential_test.wall
    potential = potential_test.potential(walls=[wall])
    potential.params["A"] = potential_test.params
    integrator.forces = [potential]

    # calculate energies and forces
    sim.operations.integrator = integrator
    sim.run(0)

    # test that parameters are still correct after attach runs
    assert potential.params["A"] == potential_test.params

    # test that the energies match reference values, half goes to each particle
    energies = potential.energies
    e = potential_test.energy
    if sim.device.communicator.rank == 0:
        numpy.testing.assert_array_almost_equal(energies, e, decimal=4)

    # test that the forces match reference values, should be directed along z
    forces = potential.forces
    f = potential_test.force
    if sim.device.communicator.rank == 0:
        numpy.testing.assert_array_almost_equal(forces, [[0, 0, f]], decimal=4)
