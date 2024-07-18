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

# bond.Quartic
potential_tests += [
    # test potential with sigma = epsilon = 0
    PotentialTestCase(
        hoomd.azplugins.bond.Quartic,
        dict(k=1434.3,r_0=1.5,b_1=-0.7589,b_2=0,U_0=67.2234,sigma=0.0,epsilon=0.0,delta=0.0),
        1,
        20.80586625,
        -99.2177025,
    ),
    # test potential with k == 0
    PotentialTestCase(
        hoomd.azplugins.bond.Quartic,
        dict(epsilon=1.0,sigma=1.0,k=0.0,r_0=1.5,b_1=-0.7589,b_2=0,U_0=67.2234,delta=0),
        1,
        68.2234,
        24,
    ),
    # test potential with delta passed
    PotentialTestCase(
        hoomd.azplugins.bond.Quartic,
        dict(epsilon=1.0, sigma=1.0, k=1434.3,r_0=1.5,b_1=-0.7589, b_2=0,U_0=67.2234,delta=0.0),
        1,
        21.80586625,
        -75.2177025,
    ),
    # test potential with nonzero delta passed
    PotentialTestCase(
        hoomd.azplugins.bond.Quartic,
        dict(epsilon=1.0, sigma=1.0, k=1434.3,r_0=1.5,b_1=-0.7589, b_2=0,U_0=67.2234,delta=0.5),
        1.5,
        21.80586625,
        -75.2177025,
    ),
    # test potential at breaking point
    PotentialTestCase(
        hoomd.azplugins.bond.Quartic,
        dict(epsilon=1.0,sigma=1.0,k=1434.3,r_0=1.5,b_1=-0.7589,b_2=0,U_0=67.2234,delta=0.0),
        1.5,
        67.2234,
        0,
    ),
    # test potential beyond breaking point
    PotentialTestCase(
        hoomd.azplugins.bond.Quartic,
        dict(epsilon=1.0,sigma=1.0,k=1434.3,r_0=1.5,b_1=-0.7589,b_2=0,U_0=67.2234,delta=0.0),
        1.5,
        67.2234,
        0,
    ),
    # test potential b_1 = b_2 = 0
    PotentialTestCase(
        hoomd.azplugins.bond.Quartic,
        dict(epsilon=1.0,sigma=1.0,k=1434.3,r_0=1.5,b_1=0,b_2=0,U_0=67.2234,delta=0.0),
        1.25,
        72.82613438,
        89.64375,
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
