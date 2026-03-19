# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2025, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

import numpy
import hoomd
import hoomd.azplugins


def test_chebyshev_construct_attach_zero(
    simulation_factory, two_particle_snapshot_factory
):
    """Construct, attach, and check force/torque output."""

    snap = two_particle_snapshot_factory()
    if snap.communicator.rank == 0:
        snap.particles.position[:] = [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]
        snap.particles.orientation[:] = [[1, 0, 0, 0], [1, 0, 0, 0]]
        snap.particles.moment_inertia[:] = [0.1, 0.1, 0.1]

    sim = simulation_factory(snap)

    integrator = hoomd.md.Integrator(dt=0.001)
    nve = hoomd.md.methods.ConstantVolume(hoomd.filter.All())
    integrator.methods = [nve]

    nlist = hoomd.md.nlist.Cell(buffer=0.4)

    domain = numpy.asarray(
        [
            [0.0, 2.0 * numpy.pi],  # theta
            [0.0, numpy.pi],  # phi
            [0.0, 2.0 * numpy.pi],  # alpha
            [0.0, numpy.pi],  # beta
            [0.0, 2.0 * numpy.pi],  # gamma
        ],
        dtype=numpy.float64,
    )

    terms = numpy.asarray(
        [
            [0, 0, 0, 0, 0, 0],
            [1, 0, 2, 0, 1, 3],
        ],
        dtype=numpy.uint32,
    )

    coeffs = numpy.asarray([1.0, -0.25], dtype=numpy.float64)

    # r0 must be 5D (and each dimension >= 2)
    r0 = (numpy.arange(32, dtype=numpy.float64).reshape((2, 2, 2, 2, 2))) * 0.01

    r_cut = 3.0

    pot = hoomd.azplugins.pair.ChebyshevAnisotropicPairPotential(
        nlist=nlist, domain=domain, terms=terms, coeffs=coeffs, r0=r0, r_cut=r_cut
    )

    assert numpy.isclose(pot.r_cut, r_cut)
    assert isinstance(pot.r0, numpy.ndarray)
    assert pot.r0.ndim == 5
    assert pot.r0.shape == (2, 2, 2, 2, 2)

    integrator.forces = [pot]
    sim.operations.integrator = integrator

    # attach
    sim.run(0)

    # check if attach happened
    assert hasattr(pot, "_cpp_obj")
    assert pot._cpp_obj is not None

    # recheck key properties after attach
    assert numpy.isclose(pot.r_cut, r_cut)
    assert pot.r0.shape == (2, 2, 2, 2, 2)

    if sim.device.communicator.rank == 0:
        numpy.testing.assert_array_equal(pot.forces, numpy.zeros((2, 3)))
        numpy.testing.assert_array_equal(pot.torques, numpy.zeros((2, 3)))
        numpy.testing.assert_array_equal(pot.energies, numpy.zeros((2,)))
