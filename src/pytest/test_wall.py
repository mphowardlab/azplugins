# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2024, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""Wall potential unit tests."""


import collections

import hoomd
import hoomd.azplugins
import numpy
import pytest

PotentialTestCase = collections.namedtuple(
    'PotentialTestCase',
    ['potential', 'params', 'pos', 'energy', 'force'],
)  ## Assume wall is always at the same place


potential_tests = []
# Hertz
potential_tests += [
    # test the calculation of force and potential
    PotentialTestCase(
        hoomd.azplugins.wall.Colloid,
        {'epsilon': 100.0, 'sigma': 1.05, 'r_cut': 6.0, 'a': 1.5, 'r_extrap': 0.0},
        numpy.array([[1.0, 1.0, -2.0]]),
        -0.374977848076 - (-0.0442302367107),
        -0.394551653468,
    ),
    PotentialTestCase(
        hoomd.azplugins.wall.LJ93,
        {'epsilon': 2.0, 'sigma': 1.05, 'r_cut': 3.0, 'r_extrap': 0.0},
        numpy.array([[1, 1, -3.95]]),
        -1.7333333333333334 - (-0.08572898249635419),
        -3.4285714285714284,
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
    r_buff = 0.4
    # L_domain_min = 20.
    snap = two_particle_snapshot_factory()
    snap.particles.N = 1
    snap.particles.position[:] = potential_test.pos
    sim = simulation_factory(snap)

    top = hoomd.wall.Plane(origin=(0, 0, -5), normal=(0, 0, 1))

    # setup dummy NVE integrator
    integrator = hoomd.md.Integrator(dt=0.001)
    nve = hoomd.md.methods.ConstantVolume(hoomd.filter.All())
    integrator.methods = [nve]

    wall_potential = potential_test.potential(walls=[top])
    wall_potential.params[('A')] = potential_test.params
    integrator.forces = [wall_potential]

    # calculate energies and forces
    sim.operations.integrator = integrator
    sim.run(0)

    # print(wall_potential.params[('A')])
    # test that parameters are still correct after attach runs
    assert wall_potential.params[('A')] == potential_test.params

    # test that the energies match reference values, half goes to each particle
    energies = wall_potential.energies
    e = potential_test.energy
    # e = -0.374977848076 - (-0.0442302367107)
    if sim.device.communicator.rank == 0:
        numpy.testing.assert_array_almost_equal(energies, e, decimal=4)

    # test that the forces match reference values, should be directed along x
    forces = wall_potential.forces
    f = potential_test.force
    # f = -0.394551653468
    if sim.device.communicator.rank == 0:
        numpy.testing.assert_array_almost_equal(forces, [[0, 0, f]], decimal=4)

    # check epsilon = 0 is zero
    wall_potential.params[('A')]['epsilon'] = 0.0
    sim.run(1)
    if sim.device.communicator.rank == 0:
        numpy.testing.assert_array_almost_equal(wall_potential.energies, 0.0, decimal=4)
        numpy.testing.assert_array_almost_equal(
            wall_potential.forces, [[0.0, 0.0, 0.0]], decimal=4
        )

    # check outside the cutoff is zero
    wall_potential.params[('A')]['r_cut'] = 1.0
    wall_potential.params[('A')]['epsilon'] = 1.0
    sim.run(1)
    if sim.device.communicator.rank == 0:
        numpy.testing.assert_array_almost_equal(wall_potential.energies, 0.0, decimal=4)
        numpy.testing.assert_array_almost_equal(
            wall_potential.forces, [[0.0, 0.0, 0.0]], decimal=4
        )
