# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2024, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""Pair potential unit tests."""

import collections

import hoomd
import hoomd.azplugins
import numpy
import pytest

PotentialTestCase = collections.namedtuple(
    'PotentialTestCase',
    ['potential', 'params', 'r_cut', 'shift', 'distance', 'energy', 'force'],
)

potential_tests = []
# Hertz
potential_tests += [
    # test the calculation of force and potential
    PotentialTestCase(
        hoomd.azplugins.pair.Hertz,
        {'epsilon': 2.0},
        1.5,
        False,
        1.05,
        0.0985,
        0.5477,
    ),
    PotentialTestCase(
        hoomd.azplugins.pair.Hertz,
        {'epsilon': 3.0},
        2.05,
        False,
        1.05,
        0.4985,
        1.2464,
    ),
    # test the cases where the potential should be zero
    # outside cutoff
    PotentialTestCase(
        hoomd.azplugins.pair.Hertz,
        {'epsilon': 1.0},
        1.0,
        False,
        1.05,
        0,
        0,
    ),
    # inside cutoff but epsilon = 0
    PotentialTestCase(
        hoomd.azplugins.pair.Hertz,
        {'epsilon': 0.0},
        3.0,
        False,
        1.05,
        0,
        0,
    ),
]
# PerturbedLennardJones
potential_tests += [
    # test the calculation of force and potential
    # test when it's in the wca part, no potential shifting
    PotentialTestCase(
        hoomd.azplugins.pair.PerturbedLennardJones,
        {'epsilon': 2.0, 'sigma': 1.05, 'attraction_scale_factor': 0.0},
        3.0,
        False,
        1.05,
        2.0,
        45.7143,
    ),
    # change attraction_scale_factor to check for shifting
    # of energy (force stays the same)
    PotentialTestCase(
        hoomd.azplugins.pair.PerturbedLennardJones,
        {'epsilon': 2.0, 'sigma': 1.05, 'attraction_scale_factor': 0.5},
        3.0,
        False,
        1.05,
        1.0,
        45.7143,
    ),
    # change sigma so that now the particle is in the LJ region
    # when attraction_scale_factor = 0, then the potential and force are zero
    PotentialTestCase(
        hoomd.azplugins.pair.PerturbedLennardJones,
        {'epsilon': 2.0, 'sigma': 0.5, 'attraction_scale_factor': 0.0},
        3.0,
        False,
        1.05,
        0,
        0,
    ),
    # partially switch on the LJ with attraction_scale_factor = 0.5
    PotentialTestCase(
        hoomd.azplugins.pair.PerturbedLennardJones,
        {'epsilon': 2.0, 'sigma': 0.5, 'attraction_scale_factor': 0.5},
        3.0,
        False,
        1.05,
        -0.0460947,
        -0.260291,
    ),
    # test that energy shifting works (bump up sigma so that at
    # rcut = 3 the shift is reasonable)
    # check wca is shifted first
    PotentialTestCase(
        hoomd.azplugins.pair.PerturbedLennardJones,
        {'epsilon': 2.0, 'sigma': 1.05, 'attraction_scale_factor': 0.5},
        3.0,
        True,
        1.05,
        1.00734,
        45.7143,
    ),
    # and check lj
    PotentialTestCase(
        hoomd.azplugins.pair.PerturbedLennardJones,
        {'epsilon': 2.0, 'sigma': 0.85, 'attraction_scale_factor': 0.5},
        3.0,
        True,
        1.05,
        -0.806849,
        -2.81197,
    ),
    # test the cases where the potential should be zero
    # outside cutoff
    PotentialTestCase(
        hoomd.azplugins.pair.PerturbedLennardJones,
        {'epsilon': 1.0, 'sigma': 1.0, 'attraction_scale_factor': 0.5},
        1.0,
        False,
        1.05,
        0,
        0,
    ),
    # inside cutoff but epsilon = 0
    PotentialTestCase(
        hoomd.azplugins.pair.PerturbedLennardJones,
        {'epsilon': 0.0, 'sigma': 1.0, 'attraction_scale_factor': 0.5},
        3.0,
        False,
        1.05,
        0,
        0,
    ),
]

# Colloid
potential_tests += [
    # test the calculation of force and potential for Solvent-Solvent
    PotentialTestCase(
        hoomd.azplugins.pair.Colloid,
        {'A': 100.0, 'a1': 0, 'a2': 0, 'sigma': 2.0},
        6.0,
        False,
        3.0,
        -0.2224,
        -0.4020,
    ),
    # test the calculation of force and potential for Colloid-Solvent
    PotentialTestCase(
        hoomd.azplugins.pair.Colloid,
        {'A': 100.0, 'a1': 1.5, 'a2': 0, 'sigma': 1.05},
        6.0,
        False,
        3.0,
        -0.2757,
        -0.7107,
    ),
    # test the calculation of force and potential for Colloid-Colloid
    PotentialTestCase(
        hoomd.azplugins.pair.Colloid,
        {'A': 100.0, 'a1': 1.5, 'a2': 0.75, 'sigma': 1.05},
        6.0,
        False,
        3.0,
        -1.0366,
        -1.8267,
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
