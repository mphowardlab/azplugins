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
    ['potential', 'params', 'r_cut', 'shift', 'energy', 'force', 'torque'],
)

potential_tests = []
# Two Patch Morse
potential_tests += [
    # test the calculation of energy, force and torque
    # test without potential shifting
    PotentialTestCase(
        hoomd.azplugins.pair.TwoPatchMorse,
        {
            'M_d': 1.8341,
            'M_r': 0.0302,
            'r_eq': 1.0043,
            'omega': 5.0,
            'alpha': 0.40,
            'repulsion': False,
        },
        1.6,
        False,
        -0.20567 * 2,
        (-11.75766, -2.46991, -3.70487),
        (-0.000000, -0.08879, 0.05919),
    ),
    # test that energy shifting works
    PotentialTestCase(
        hoomd.azplugins.pair.TwoPatchMorse,
        {
            'M_d': 1.8341,
            'M_r': 0.0302,
            'r_eq': 1.0043,
            'omega': 5.0,
            'alpha': 0.40,
            'repulsion': False,
        },
        1.10,
        True,
        -0.14195 * 2,
        None,
        None,
    ),
    # test the cases where the potential should be zero
    # test without potential shifting, particles are outside cutoff
    PotentialTestCase(
        hoomd.azplugins.pair.TwoPatchMorse,
        {
            'M_d': 1.8341,
            'M_r': 0.0302,
            'r_eq': 1.0043,
            'omega': 5.0,
            'alpha': 0.40,
            'repulsion': False,
        },
        1.0,
        True,
        0,
        None,
        None,
    ),
    # inside cutoff but Md = 0
    PotentialTestCase(
        hoomd.azplugins.pair.TwoPatchMorse,
        {
            'M_d': 0.0,
            'M_r': 0.0302,
            'r_eq': 1.0043,
            'omega': 5.0,
            'alpha': 0.40,
            'repulsion': False,
        },
        1.6,
        True,
        0,
        None,
        None,
    ),
    # test no force
    PotentialTestCase(
        hoomd.azplugins.pair.TwoPatchMorse,
        {
            'M_d': 1.8341,
            'M_r': 0.0302,
            'r_eq': 1.1,
            'omega': 100.0,
            'alpha': 0.40,
            'repulsion': False,
        },
        1.6,
        False,
        -1.8341,
        (0, 0, 0),
        None,
    ),
]


@pytest.mark.parametrize(
    'potential_test', potential_tests, ids=lambda x: x.potential.__name__
)
def test_energy_force_and_torque(
    simulation_factory, two_particle_snapshot_factory, potential_test
):
    """Test energy and force evaluation."""
    # make 2 particle test configuration
    snap = two_particle_snapshot_factory()
    if snap.communicator.rank == 0:
        snap.particles.position[:] = [[-0.5, -0.10, -0.15], [0.5, 0.10, 0.15]]
        snap.particles.orientation[:] = [[1, 0, 0, 0], [1, 0, 0, 0]]
        snap.particles.moment_inertia[:] = [0.1, 0.1, 0.1]
    sim = simulation_factory(snap)

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
    ref_values = (list(potential_test.params.values()),)
    test_values = [potential.params[('A', 'A')][k] for k in potential_test.params]
    assert numpy.allclose(test_values, ref_values)

    # test that the energies match reference values, half goes to each particle
    energies = potential.energies
    e = potential_test.energy
    if sim.device.communicator.rank == 0:
        numpy.testing.assert_array_almost_equal(energies, [0.5 * e, 0.5 * e], decimal=4)

    # test that the forces match reference values, should be directed along x
    forces = potential.forces
    f = potential_test.force
    if f is not None and sim.device.communicator.rank == 0:
        f = numpy.array(f)
        numpy.testing.assert_array_almost_equal(forces, [-f, f], decimal=4)

    # test that the torques match reference value
    torques = potential.torques
    T = potential_test.torque
    if T is not None and sim.device.communicator.rank == 0:
        numpy.testing.assert_array_almost_equal(torques, [T, T], decimal=4)
