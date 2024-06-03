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
    PotentialTestCase(
        hoomd.azplugins.bond.DoubleWell, dict(a=3.0, b=0.5, V_max=1.0, c=0.0), 1.0, 0, 0
    ),
    PotentialTestCase(
        hoomd.azplugins.bond.DoubleWell,
        dict(a=2.0, b=2.0, V_max=5.0, c=0.0),
        1.0,
        5.0,
        0,
    ),
    PotentialTestCase(
        hoomd.azplugins.bond.DoubleWell,
        dict(a=1.0, b=1.0, V_max=1.0, c=0.0),
        1.0,
        0.5625,
        1.5,
    ),
    PotentialTestCase(
        hoomd.azplugins.bond.DoubleWell,
        dict(a=1.0, b=1.0, V_max=1.0, c=1.0),
        1.0,
        1.03125,
        0.25,
    ),
]


@pytest.fixture(scope='session')
def bonded_two_particle_snapshot_factory(two_particle_snapshot_factory):
    """Fixture for a single bond."""

    def make_snapshot(bond_types=None, **kwargs):
        if bond_types is None:
            bond_types = ['A-A']
        snap = two_particle_snapshot_factory(**kwargs)
        if snap.communicator.rank == 0:
            snap.bonds.types = bond_types
            snap.bonds.N = 1
            snap.bonds.group[0] = [0, 1]
        return snap

    return make_snapshot


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
