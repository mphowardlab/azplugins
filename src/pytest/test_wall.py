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
    ['potential', 'params', 'r_cut', 'shift', 'pos', 'energy', 'force'],
)## Assume wall is always at the same place


potential_tests = []
# Hertz
potential_tests += [
    # test the calculation of force and potential
    PotentialTestCase(
        hoomd.azplugins.wall.Colloid,
        {'epsilon': 100.0, 'sigma': 1.05},
        6.0,
        numpy.array([1.,1.,-2.]),
        1.05,
        -0.374977848076 - (-0.0442302367107),
        -0.394551653468,
    ),
    PotentialTestCase(
        hoomd.azplugins.wall.Colloid,
        {'epsilon': 100.0, 'sigma': 1.05},
        6.0,
        False,
        1.05,
        -0.374977848076 - (-0.0442302367107),
        -0.394551653468,
    ),
]


@pytest.mark.parametrize(
    'potential_test', potential_tests, ids=lambda x: x.potential.__name__
)
def test_energy_and_force(
    simulation_factory, snapshot_factory, potential_test
):
    """Test energy and force evaluation."""
    # make 2 particle test configuration
    r_cut = potential_test.r_cut
    r_buff = 0.4
    L_domain_min = 2 * (r_cut + r_buff)
    snap = snapshot_factory()
    snap.particles.position[:] = [[1., 1., -2.]]
    snap.particles.diameter[:] = [1.5]
    sim = simulation_factory(snap)

    walls = hoomd.wall.Plane(origin=(0, 0, -5), normal=(0, 0, 1))

    # setup dummy NVE integrator
    integrator = hoomd.md.Integrator(dt=0.001)
    nve = hoomd.md.methods.ConstantVolume(hoomd.filter.All())
    integrator.methods = [nve]

    wall_potential = potential_test.potential(
        nlist=hoomd.md.nlist.Cell(buffer=r_buff),
        # default_r_cut=potential_test.r_cut,
        default_r_cut=6.0,
        **extra_args,
    )
    wall_potential.params[('A')] = potential_test.params


    integrator.forces = [wall_potential]

    # calculate energies and forces
    sim.operations.integrator = integrator
    sim.run(0)

    # test that parameters are still correct after attach runs
    assert wall_potential.params[('A')] == potential_test.params

    # test that the energies match reference values, half goes to each particle
    energies = wall_potential.energies
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
