# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2025, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

import numpy
import hoomd
import hoomd.azplugins


def test_attach_and_zero_force(simulation_factory, two_particle_snapshot_factory):
    """Construct, attach, and check force/torque output."""

    # Construct the Python object
    nlist = hoomd.md.nlist.Cell(buffer=0.4)

    domain = numpy.zeros((5, 2), dtype=numpy.float64)
    terms = numpy.zeros((2, 6), dtype=numpy.uint32)
    coeffs = numpy.zeros((2,), dtype=numpy.float64)
    r0_data = numpy.zeros((2, 2, 2, 2, 2), dtype=numpy.float64)
    r_cut = 3.0

    pot = hoomd.azplugins.pair.ChebyshevAnisotropicPairPotential(
        nlist=nlist,
        domain=domain,
        r_cut=r_cut,
        terms=terms,
        coeffs=coeffs,
        r0_data=r0_data,
    )

    # Pre-attach checks
    assert numpy.isclose(pot.r_cut, r_cut)
    assert pot.n_terms == 2
    assert pot.r0_shape == (2, 2, 2, 2, 2)

    # Attach via a 0-step simulation
    snap = two_particle_snapshot_factory()
    if snap.communicator.rank == 0:
        snap.particles.position[:] = [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]
        snap.particles.orientation[:] = [[1, 0, 0, 0], [1, 0, 0, 0]]
        snap.particles.moment_inertia[:] = [0.1, 0.1, 0.1]

    sim = simulation_factory(snap)

    integrator = hoomd.md.Integrator(dt=0.001)
    nve = hoomd.md.methods.ConstantVolume(hoomd.filter.All())
    integrator.methods = [nve]

    integrator.forces = [pot]
    sim.operations.integrator = integrator

    # Attach all objects
    sim.run(0)

    # After attach
    assert hasattr(pot, "_cpp_obj")
    assert pot._cpp_obj is not None

    # Post-attach checks
    assert numpy.isclose(pot.r_cut, r_cut)
    assert pot.n_terms == 2
    assert pot.r0_shape == (2, 2, 2, 2, 2)

    assert pot._cpp_obj.n_terms == 2
    assert numpy.isclose(pot._cpp_obj.r_cut, r_cut)

    # Check Force/torque/energy outputs
    if sim.device.communicator.rank == 0:
        numpy.testing.assert_array_equal(pot.forces, numpy.zeros((2, 3)))
        numpy.testing.assert_array_equal(pot.torques, numpy.zeros((2, 3)))
        numpy.testing.assert_array_equal(pot.energies, numpy.zeros((2,)))
