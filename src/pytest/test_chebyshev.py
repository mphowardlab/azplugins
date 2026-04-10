# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2025, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

import numpy
import hoomd
import hoomd.azplugins
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation


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
            [0.0, 2.0 * numpy.pi],
            [0.0, numpy.pi],
            [0.0, 2.0 * numpy.pi],
            [0.0, numpy.pi],
            [0.0, 2.0 * numpy.pi],
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

    coeffs = numpy.asarray([0.0, 0.0], dtype=numpy.float64)

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


def test_chebyshev_force_torque_energy_no_symmetry(
    simulation_factory, two_particle_snapshot_factory
):
    """Test energy, force, and torque, without considering symmetry."""
    rc = 3.0
    phi_min = 1e-5
    beta_min = 1e-5

    domain = numpy.array(
        [
            [0.0, 2.0 * numpy.pi],
            [phi_min, numpy.pi - phi_min],
            [0.0, 2.0 * numpy.pi],
            [beta_min, numpy.pi - phi_min],
            [0.0, 2.0 * numpy.pi],
        ],
        dtype=numpy.float64,
    )

    terms = numpy.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0],
        ],
        dtype=numpy.uint32,
    )
    coeffs = numpy.array([1.0, 0.25, 1.5, -1.0], dtype=numpy.float64)

    # r0 data: shape (3, 2, 3, 2, 3) = 108 values.
    r0_data = numpy.array([1, 2.1, 3.2] * 36, dtype=numpy.float64).reshape(
        3, 2, 3, 2, 3
    )

    theta_grid = numpy.linspace(0, 2 * numpy.pi, 3)
    phi_grid = numpy.linspace(phi_min, numpy.pi - phi_min, 2)
    alpha_grid = numpy.linspace(0, 2 * numpy.pi, 3)
    beta_grid = numpy.linspace(beta_min, numpy.pi - phi_min, 2)
    gamma_grid = numpy.linspace(0, 2 * numpy.pi, 3)

    r0_interp = RegularGridInterpolator(
        (theta_grid, phi_grid, alpha_grid, beta_grid, gamma_grid),
        r0_data,
        method="linear",
        bounds_error=False,
        fill_value=numpy.nan,
    )

    def rho_to_r(rho, r0, rc):
        """Invert  rho = (1/r - 1/r0) / (1/(r0+rc) - 1/r0)  to obtain r."""
        inv_r0 = 1.0 / r0
        inv_r0_rc = 1.0 / (r0 + rc)
        inv_r = rho * (inv_r0_rc - inv_r0) + inv_r0
        return 1.0 / inv_r

    def run_pair(rho, theta, phi, alpha, beta, gamma):
        """Build a two-particle simulation, run for one step, and return
        the potential object."""
        snap = two_particle_snapshot_factory()
        if snap.communicator.rank == 0:
            r0 = float(r0_interp(numpy.array([theta, phi, alpha, beta, gamma]))[0])
            r = rho_to_r(rho, r0, rc)

            dx = r * numpy.sin(phi) * numpy.cos(theta)
            dy = r * numpy.sin(phi) * numpy.sin(theta)
            dz = r * numpy.cos(phi)

            rot = Rotation.from_euler("ZXZ", [alpha, beta, gamma])
            q_j = rot.as_quat(scalar_first=True)

            snap.particles.position[0] = [0.0, 0.0, 0.0]
            snap.particles.position[1] = [-dx, -dy, -dz]
            snap.particles.orientation[0] = [1, 0, 0, 0]
            snap.particles.orientation[1] = q_j
            snap.particles.moment_inertia[:] = [0.1, 0.1, 0.1]

        sim = simulation_factory(snap)

        integrator = hoomd.md.Integrator(dt=0.001)
        nve = hoomd.md.methods.ConstantVolume(hoomd.filter.All())
        integrator.methods = [nve]

        nlist = hoomd.md.nlist.Cell(buffer=1)
        pot = hoomd.azplugins.pair.ChebyshevAnisotropicPairPotential(
            nlist=nlist,
            domain=domain,
            terms=terms,
            coeffs=coeffs,
            r0=r0_data,
            r_cut=rc,
        )

        integrator.forces = [pot]
        sim.operations.integrator = integrator
        sim.run(0)
        return sim, pot

    def check(sim, pot, expected_energy, expected_force, expected_torque):
        """Compare the output on particle 0 to the Python reference (smolyay)."""
        if sim.device.communicator.rank == 0:
            numpy.testing.assert_allclose(
                pot.energies[0], expected_energy, atol=1e-3, rtol=1e-3
            )
            numpy.testing.assert_allclose(
                pot.forces[0], expected_force, atol=1e-3, rtol=1e-3
            )
            numpy.testing.assert_allclose(
                pot.torques[0], expected_torque, atol=1e-3, rtol=1e-3
            )

    # point 1: interior
    sim, pot = run_pair(
        rho=0.2,
        theta=numpy.pi / 4,
        phi=numpy.pi / 4,
        alpha=2 * numpy.pi / 5,
        beta=numpy.pi / 2,
        gamma=numpy.pi,
    )
    check(
        sim,
        pot,
        expected_energy=-0.41,
        expected_force=numpy.array([-1.324, -1.324, -1.872]),
        expected_torque=numpy.array([0.944, -0.307, -0.271]),
    )

    # point 2: rho < 0
    sim, pot = run_pair(
        rho=-0.1,
        theta=numpy.pi / 4,
        phi=numpy.pi / 4,
        alpha=2 * numpy.pi / 5,
        beta=numpy.pi / 2,
        gamma=numpy.pi,
    )
    check(
        sim,
        pot,
        expected_energy=-1.25,
        expected_force=numpy.array([-1.906, -1.906, -2.695]),
        expected_torque=numpy.array([1.226, -0.398, -0.398]),
    )

    # point 3: rho < 0 and phi at upper boundary
    sim, pot = run_pair(
        rho=-0.1,
        theta=numpy.pi / 4,
        phi=numpy.pi - phi_min,
        alpha=2 * numpy.pi / 15,
        beta=numpy.pi / 2,
        gamma=numpy.pi,
    )
    check(
        sim,
        pot,
        expected_energy=-1.583,
        expected_force=numpy.array([0.0, 0.0, 4.296]),
        expected_torque=numpy.array([0.591, -1.327, -0.398]),
    )

    # point 4: beta at lower boundary
    sim, pot = run_pair(
        rho=0.2,
        theta=numpy.pi / 4,
        phi=numpy.pi / 4,
        alpha=2 * numpy.pi / 5,
        beta=beta_min,
        gamma=numpy.pi,
    )
    check(
        sim,
        pot,
        expected_energy=-0.41,
        expected_force=numpy.array([-1.324, -1.324, -1.872]),
        expected_torque=numpy.array([120148.0, -39038.6, -0.271]),
    )

    # point 5: interior with rho near 1
    sim, pot = run_pair(
        rho=0.95,
        theta=numpy.pi / 4,
        phi=numpy.pi / 6,
        alpha=2 * numpy.pi / 5,
        beta=numpy.pi / 2,
        gamma=numpy.pi / 8,
    )
    check(
        sim,
        pot,
        expected_energy=2.74,
        expected_force=numpy.array([-0.174, -0.174, -0.427]),
        expected_torque=numpy.array([0.207, -0.067, 0.207]),
    )

    # point 6: rho > 1, pair is beyond the surface cutoff
    sim, pot = run_pair(
        rho=1.05,
        theta=numpy.pi / 4,
        phi=numpy.pi / 6,
        alpha=2 * numpy.pi / 5,
        beta=numpy.pi / 2,
        gamma=numpy.pi / 8,
    )
    if sim.device.communicator.rank == 0:
        numpy.testing.assert_allclose(pot.energies[0], 0.0, atol=1e-10)
        numpy.testing.assert_allclose(pot.forces[0], [0.0, 0.0, 0.0], atol=1e-10)
        numpy.testing.assert_allclose(pot.torques[0], [0.0, 0.0, 0.0], atol=1e-10)
