# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2024, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""Bond potential unit tests."""

import collections

import hoomd
import numpy
import pytest

PotentialTestCase = collections.namedtuple(
    'PotentialTestCase',
    ['potential', 'params', 'distance', 'energy', 'force'],
)

potential_tests = []
# bond.DoubleWell
potential_tests += [
    # test potential at first minimum
    PotentialTestCase(
        hoomd.azplugins.bond.DoubleWell,
        dict(r_0=0.5, r_1=2.5, U_1=5.0, U_tilt=0.0),
        0.5,
        0,
        0,
    ),
    # test potential at local maximum
    PotentialTestCase(
        hoomd.azplugins.bond.DoubleWell,
        dict(r_0=0.5, r_1=2.5, U_1=5.0, U_tilt=0.0),
        2.5,
        5.0,
        0,
    ),
    # test potential at second minimum
    PotentialTestCase(
        hoomd.azplugins.bond.DoubleWell,
        dict(r_0=0.5, r_1=2.5, U_1=5.0, U_tilt=0.0),
        4.5,
        0,
        0,
    ),
    # test potential between first minimum and maximum
    PotentialTestCase(
        hoomd.azplugins.bond.DoubleWell,
        dict(r_0=1.0, r_1=2.0, U_1=1.0, U_tilt=0.0),
        1.5,
        0.5625,
        -1.5,
    ),
    # test potential between maximum and second minimum
    PotentialTestCase(
        hoomd.azplugins.bond.DoubleWell,
        dict(r_0=1.0, r_1=2.0, U_1=1.0, U_tilt=0.0),
        2.5,
        0.5625,
        1.5,
    ),
    # test non-zero tilt
    PotentialTestCase(
        hoomd.azplugins.bond.DoubleWell,
        dict(r_0=1.0, r_1=2.0, U_1=1.0, U_tilt=0.5),
        2.5,
        1.03125,
        0.25,
    ),
]


@pytest.mark.parametrize(
    'potential_test', potential_tests, ids=lambda x: x.potential.__name__
)
def test_energy_and_force(
    simulation_factory, bonded_two_particle_snapshot_factory, potential_test
):
    """Test energy and force evaluation."""
    # make 2 particle test configuration
    sim = simulation_factory(
        bonded_two_particle_snapshot_factory(d=potential_test.distance)
    )

    # setup dummy NVE integrator
    integrator = hoomd.md.Integrator(dt=0.001)
    nve = hoomd.md.methods.ConstantVolume(hoomd.filter.All())
    integrator.methods = [nve]

    # setup pair potential
    potential = potential_test.potential()
    potential.params['A-A'] = potential_test.params
    integrator.forces = [potential]

    # calculate energies and forces
    sim.operations.integrator = integrator
    sim.run(0)

    # test that parameters are still correct after attach runs
    assert potential.params['A-A'] == potential_test.params

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
