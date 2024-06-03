# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2024, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""Pair potential unit tests."""

import collections

import hoomd
import numpy
import pytest

PotentialTestCase = collections.namedtuple(
    'PotentialTestCase',
    ['potential', 'params', 'r_cut', 'shift', 'distance', 'energy', 'force'],
)

potential_tests = []
# tests for PerturbedLennardJones potential
potential_tests += [
    PotentialTestCase(
        hoomd.azplugins.pair.PerturbedLennardJones,
        {'epsilon': 2.0, 'sigma': 1.05, 'lam': 0.0},
        3.0,
        False,
        1.05,
        2.0,
        45.7143,
    ),
    PotentialTestCase(
        hoomd.azplugins.pair.PerturbedLennardJones,
        {'epsilon': 2.0, 'sigma': 1.05, 'lam': 0.5},
        3.0,
        False,
        1.05,
        1.0,
        45.7143,
    ),
    PotentialTestCase(
        hoomd.azplugins.pair.PerturbedLennardJones,
        {'epsilon': 2.0, 'sigma': 0.5, 'lam': 0.0},
        3.0,
        False,
        1.05,
        0,
        0,
    ),
    PotentialTestCase(
        hoomd.azplugins.pair.PerturbedLennardJones,
        {'epsilon': 2.0, 'sigma': 0.5, 'lam': 0.5},
        3.0,
        False,
        1.05,
        -0.0460947,
        -0.260291,
    ),
    PotentialTestCase(
        hoomd.azplugins.pair.PerturbedLennardJones,
        {'epsilon': 2.0, 'sigma': 1.05, 'lam': 0.5},
        3.0,
        True,
        1.05,
        1.00734,
        45.7143,
    ),
    PotentialTestCase(
        hoomd.azplugins.pair.PerturbedLennardJones,
        {'epsilon': 2.0, 'sigma': 0.85, 'lam': 0.5},
        3.0,
        True,
        1.05,
        -0.806849,
        -2.81197,
    ),
    PotentialTestCase(
        hoomd.azplugins.pair.PerturbedLennardJones,
        {'epsilon': 1.0, 'sigma': 1.0, 'lam': 0.5},
        1.0,
        False,
        1.05,
        0,
        0,
    ),
    PotentialTestCase(
        hoomd.azplugins.pair.PerturbedLennardJones,
        {'epsilon': 0.0, 'sigma': 1.0, 'lam': 0.5},
        3.0,
        False,
        1.05,
        0,
        0,
    ),
]


@pytest.mark.parametrize(
    'potential_test', potential_tests, ids=lambda x: x.potential.__name__
)
def test_energy_and_force(
    simulation_factory, two_particle_snapshot_factory, potential_test
):
    """Test energy and force evaluation."""
    # make 2 particle test configuration
    sim = simulation_factory(two_particle_snapshot_factory(d=potential_test.distance))

    # setup dummy NVE integrator
    integrator = hoomd.md.Integrator(dt=0.001)
    nve = hoomd.md.methods.ConstantVolume(hoomd.filter.All())
    integrator.methods = [nve]

    # setup pair potential
    potential = potential_test.potential(
        nlist=hoomd.md.nlist.Cell(buffer=0.4),
        default_r_cut=potential_test.r_cut,
        mode='shift' if potential_test.shift else 'none',
    )
    potential.params[('A', 'A')] = potential_test.params
    integrator.forces = [potential]

    # calculate energies and forces
    sim.operations.integrator = integrator
    sim.run(0)

    # test that parameters are still correct after attach runs
    assert potential.params[('A', 'A')] == potential_test.params

    # test that the energies match reference values, half goes to each particle
    energies = potential.energies
    e = potential_test.energy
    if sim.device.communicator.rank == 0:
        numpy.testing.assert_array_almost_equal(energies, [0.5 * e, 0.5 * e], decimal=4)

    # test that the forces match reference values, should be directed along x
    forces = potential.forces
    f = potential_test.force
    if sim.device.communicator.rank == 0:
        numpy.testing.assert_array_almost_equal(
            forces, [[-f, 0, 0], [f, 0, 0]], decimal=4
        )
