# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2025, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

import numpy
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation

import hoomd
import hoomd.azplugins

import pytest

# Parameters that are identical across every energy/force/torque test.
rc = 3.0
phi_min = 1e-5
beta_min = 1e-5

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
r0_data = numpy.array([1, 2.1, 3.2] * 36, dtype=numpy.float64).reshape(3, 2, 3, 2, 3)


def rho_to_r(rho, r0, rc):
    """Invert rho = (1/r - 1/r0) / (1/(r0+rc) - 1/r0) to recover r."""
    inv_r0 = 1.0 / r0
    inv_r0_rc = 1.0 / (r0 + rc)
    inv_r = rho * (inv_r0_rc - inv_r0) + inv_r0
    return 1.0 / inv_r


def build_simulation(
    simulation_factory,
    two_particle_snapshot_factory,
    pot_cls,
    r0,
    rho,
    theta,
    phi,
    alpha,
    beta,
    gamma,
):
    """Place two particles at the prescribed coordinates and
    return the attached potential.

    Particle 0 sits at the origin with identity orientation.  Particle 1
    is placed so that the C++ code sees (rho, theta, phi, alpha, beta,
    gamma) as the pair's generalised coordinates.  For Null symmetry the
    caller supplies ``r0`` from the test's own interpolator; for Cube /
    Tetrahedron tests the caller supplies the reduced-frame ``r0``
    directly because the input angles need not to be in the reduced
    coordinates which interpolator expects.
    """
    snap = two_particle_snapshot_factory()
    if snap.communicator.rank == 0:
        r = rho_to_r(rho, r0, rc)

        dx = r * numpy.sin(phi) * numpy.cos(theta)
        dy = r * numpy.sin(phi) * numpy.sin(theta)
        dz = r * numpy.cos(phi)

        q_j = Rotation.from_euler("ZXZ", [alpha, beta, gamma]).as_quat(
            scalar_first=True
        )

        snap.particles.position[0] = [0.0, 0.0, 0.0]
        snap.particles.position[1] = [-dx, -dy, -dz]
        snap.particles.orientation[0] = [1, 0, 0, 0]
        snap.particles.orientation[1] = q_j
        snap.particles.moment_inertia[:] = [0.1, 0.1, 0.1]

    sim = simulation_factory(snap)

    integrator = hoomd.md.Integrator(dt=0.001)
    integrator.methods = [hoomd.md.methods.ConstantVolume(hoomd.filter.All())]

    nlist = hoomd.md.nlist.Cell(buffer=1)
    pot = pot_cls(
        nlist=nlist,
        terms=terms,
        coeffs=coeffs,
        r0=r0_data,
        r_cut=rc,
    )

    integrator.forces = [pot]
    sim.operations.integrator = integrator
    sim.run(0)
    return sim, pot


def check_pair(sim, pot, expected_energy, expected_force, expected_torque):
    """Compare the C++ output on both particles."""
    if sim.device.communicator.rank == 0:
        half_e = 0.5 * expected_energy

        numpy.testing.assert_allclose(pot.energies[0], half_e, atol=1e-3, rtol=1e-3)
        numpy.testing.assert_allclose(
            pot.forces[0], expected_force, atol=1e-3, rtol=1e-3
        )
        numpy.testing.assert_allclose(
            pot.torques[0], expected_torque, atol=1e-3, rtol=1e-3
        )

        numpy.testing.assert_allclose(pot.energies[1], half_e, atol=1e-3, rtol=1e-3)
        numpy.testing.assert_allclose(
            pot.forces[1], -expected_force, atol=1e-3, rtol=1e-3
        )
        numpy.testing.assert_allclose(
            pot.torques[1], -expected_torque, atol=1e-3, rtol=1e-3
        )


def check_zero_pair(sim, pot):
    """Assert that both particles have zero force, torque, and energy."""
    if sim.device.communicator.rank == 0:
        numpy.testing.assert_allclose(pot.energies[0], 0.0, atol=1e-10)
        numpy.testing.assert_allclose(pot.forces[0], [0.0, 0.0, 0.0], atol=1e-10)
        numpy.testing.assert_allclose(pot.torques[0], [0.0, 0.0, 0.0], atol=1e-10)
        numpy.testing.assert_allclose(pot.energies[1], 0.0, atol=1e-10)
        numpy.testing.assert_allclose(pot.forces[1], [0.0, 0.0, 0.0], atol=1e-10)
        numpy.testing.assert_allclose(pot.torques[1], [0.0, 0.0, 0.0], atol=1e-10)


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
    integrator.methods = [hoomd.md.methods.ConstantVolume(hoomd.filter.All())]

    nlist = hoomd.md.nlist.Cell(buffer=0.4)

    terms = numpy.asarray(
        [
            [0, 0, 0, 0, 0, 0],
            [1, 0, 2, 0, 1, 3],
        ],
        dtype=numpy.uint32,
    )
    coeffs = numpy.asarray([0.0, 0.0], dtype=numpy.float64)

    # r0 must be 5D (each dimension >= 2)
    r0 = (numpy.arange(32, dtype=numpy.float64).reshape((2, 2, 2, 2, 2))) * 0.01
    r_cut = 3.0

    pot = hoomd.azplugins.pair.ChebyshevAnisotropicPairPotential(
        nlist=nlist, terms=terms, coeffs=coeffs, r0=r0, r_cut=r_cut
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


def good_kwargs():
    """A set of constructor kwargs known to be valid."""
    return dict(
        nlist=hoomd.md.nlist.Cell(buffer=0.4),
        terms=numpy.zeros((1, 6), dtype=numpy.uint32),
        coeffs=numpy.zeros((1,), dtype=numpy.float64),
        r0=numpy.zeros((2, 2, 2, 2, 2), dtype=numpy.float64),
        r_cut=3.0,
    )


def test_chebyshev_invalid_terms_shape():
    """Raise ValueError when ``terms`` is not (Nterms, 6)."""
    kwargs = good_kwargs()
    kwargs["terms"] = numpy.zeros((1, 5), dtype=numpy.uint32)
    kwargs["coeffs"] = numpy.zeros((1,), dtype=numpy.float64)
    with pytest.raises(ValueError, match=r"terms must have shape \(Nterms, 6\)\."):
        hoomd.azplugins.pair.ChebyshevAnisotropicPairPotential(**kwargs)


def test_chebyshev_invalid_coeffs_shape():
    """Raise ValueError when ``coeffs`` length does not match Nterms."""
    kwargs = good_kwargs()
    kwargs["terms"] = numpy.zeros((2, 6), dtype=numpy.uint32)
    kwargs["coeffs"] = numpy.zeros((1,), dtype=numpy.float64)
    with pytest.raises(ValueError, match=r"coeffs must have shape \(Nterms,\)"):
        hoomd.azplugins.pair.ChebyshevAnisotropicPairPotential(**kwargs)


def test_chebyshev_invalid_r0_ndim():
    """Raise ValueError when ``r0`` is not a 5D array."""
    kwargs = good_kwargs()
    kwargs["r0"] = numpy.zeros((2, 2, 2, 2), dtype=numpy.float64)
    with pytest.raises(ValueError, match=r"r0 must be a 5D array\."):
        hoomd.azplugins.pair.ChebyshevAnisotropicPairPotential(**kwargs)


def test_chebyshev_invalid_r0_shape():
    """Raise ValueError when ``r0`` has a dimension with less than 2 points."""
    kwargs = good_kwargs()
    kwargs["r0"] = numpy.zeros((2, 2, 1, 2, 2), dtype=numpy.float64)
    with pytest.raises(
        ValueError,
        match=r"r0 must have at least 2 grid points along each of its 5 dimensions\.",
    ):
        hoomd.azplugins.pair.ChebyshevAnisotropicPairPotential(**kwargs)


# Null symmetry
def test_chebyshev_force_torque_energy_null_symmetry(
    simulation_factory, two_particle_snapshot_factory
):
    """Force, torque, and energy with no symmetry reduction."""
    # r0 interpolator aligned with r0_data's shape (3, 2, 3, 2, 3).
    theta_grid = numpy.linspace(0, 2 * numpy.pi, 3)
    phi_grid = numpy.linspace(phi_min, numpy.pi - phi_min, 2)
    alpha_grid = numpy.linspace(0, 2 * numpy.pi, 3)
    beta_grid = numpy.linspace(beta_min, numpy.pi - beta_min, 2)
    gamma_grid = numpy.linspace(0, 2 * numpy.pi, 3)

    r0_interp = RegularGridInterpolator(
        (theta_grid, phi_grid, alpha_grid, beta_grid, gamma_grid),
        r0_data,
        method="linear",
        bounds_error=False,
        fill_value=numpy.nan,
    )

    def run(rho, theta, phi, alpha, beta, gamma):
        r0 = float(r0_interp(numpy.array([theta, phi, alpha, beta, gamma]))[0])
        return build_simulation(
            simulation_factory,
            two_particle_snapshot_factory,
            hoomd.azplugins.pair.ChebyshevAnisotropicPairPotential,
            r0,
            rho,
            theta,
            phi,
            alpha,
            beta,
            gamma,
        )

    # point 1: interior
    sim, pot = run(
        rho=0.2,
        theta=numpy.pi / 4,
        phi=numpy.pi / 4,
        alpha=2 * numpy.pi / 5,
        beta=numpy.pi / 2,
        gamma=numpy.pi,
    )
    check_pair(
        sim,
        pot,
        expected_energy=-0.41,
        expected_force=numpy.array([-1.324, -1.324, -1.872]),
        expected_torque=numpy.array([0.944, -0.307, -0.271]),
    )

    # point 2: rho < 0 (clamped for derivatives, extrapolated for energy)
    sim, pot = run(
        rho=-0.1,
        theta=numpy.pi / 4,
        phi=numpy.pi / 4,
        alpha=2 * numpy.pi / 5,
        beta=numpy.pi / 2,
        gamma=numpy.pi,
    )
    check_pair(
        sim,
        pot,
        expected_energy=-1.67,
        expected_force=numpy.array([-1.906, -1.906, -2.695]),
        expected_torque=numpy.array([1.226, -0.398, -0.398]),
    )

    # point 3: phi at upper boundary
    sim, pot = run(
        rho=0.0,
        theta=numpy.pi / 4,
        phi=numpy.pi - phi_min,
        alpha=2 * numpy.pi / 15,
        beta=numpy.pi / 2,
        gamma=numpy.pi,
    )
    check_pair(
        sim,
        pot,
        expected_energy=-1.583,
        expected_force=numpy.array([0.0, 0.0, 3.832]),
        expected_torque=numpy.array([0.546, -1.226, -0.398]),
    )

    # point 4: beta at lower boundary
    sim, pot = run(
        rho=0.2,
        theta=numpy.pi / 4,
        phi=numpy.pi / 4,
        alpha=2 * numpy.pi / 5,
        beta=beta_min,
        gamma=numpy.pi,
    )
    check_pair(
        sim,
        pot,
        expected_energy=-0.41,
        expected_force=numpy.array([-1.324, -1.324, -1.872]),
        expected_torque=numpy.array([120148.0, -39038.6, -0.271]),
    )

    # point 5: interior with rho near 1
    sim, pot = run(
        rho=0.95,
        theta=numpy.pi / 4,
        phi=numpy.pi / 6,
        alpha=2 * numpy.pi / 5,
        beta=numpy.pi / 2,
        gamma=numpy.pi / 8,
    )
    check_pair(
        sim,
        pot,
        expected_energy=2.74,
        expected_force=numpy.array([-0.174, -0.174, -0.427]),
        expected_torque=numpy.array([0.207, -0.067, 0.207]),
    )

    # point 6: rho > 1, beyond surface cutoff - all zeros
    sim, pot = run(
        rho=1.05,
        theta=numpy.pi / 4,
        phi=numpy.pi / 6,
        alpha=2 * numpy.pi / 5,
        beta=numpy.pi / 2,
        gamma=numpy.pi / 8,
    )
    check_zero_pair(sim, pot)


# Cube symmetry
def test_chebyshev_force_torque_energy_cube_symmetry(
    simulation_factory, two_particle_snapshot_factory
):
    """Force, torque, and energy with cube symmetry reduction.

    Reduced domain: theta in [0, pi/4], phi in [1e-5, pi/2],
    alpha in [0, 2 pi], beta in [1e-5, arccos(1/sqrt(3))],
    gamma in [0, pi/2]."""

    def run(r0, rho, theta, phi, alpha, beta, gamma):
        return build_simulation(
            simulation_factory,
            two_particle_snapshot_factory,
            hoomd.azplugins.pair.ChebyshevAnisotropicPairPotentialCube,
            r0,
            rho,
            theta,
            phi,
            alpha,
            beta,
            gamma,
        )

    # point 1: interior
    sim, pot = run(
        r0=2.46666667,
        rho=0.2,
        theta=numpy.pi / 8,
        phi=numpy.pi / 5,
        alpha=2 * numpy.pi / 5,
        beta=numpy.pi / 6,
        gamma=numpy.pi / 3,
    )
    check_pair(
        sim,
        pot,
        expected_energy=-0.41,
        expected_force=numpy.array([-1.335, -0.553, -1.989]),
        expected_torque=numpy.array([7.395, -2.403, -0.271]),
    )

    # point 2: rho < 0 (clamped for derivatives, extrapolated for energy)
    sim, pot = run(
        r0=2.46666667,
        rho=-0.1,
        theta=numpy.pi / 8,
        phi=numpy.pi / 5,
        alpha=2 * numpy.pi / 5,
        beta=numpy.pi / 6,
        gamma=numpy.pi / 3,
    )
    check_pair(
        sim,
        pot,
        expected_energy=-1.67,
        expected_force=numpy.array([-1.875, -0.777, -2.793]),
        expected_torque=numpy.array([9.579, -3.113, -0.398]),
    )

    # point 3: interior with rho=0
    sim, pot = run(
        r0=2.62072583,
        rho=0.0,
        theta=2 * numpy.pi / 7,
        phi=numpy.pi / 9,
        alpha=2 * numpy.pi / 15,
        beta=numpy.pi / 4,
        gamma=numpy.pi / 5,
    )
    check_pair(
        sim,
        pot,
        expected_energy=-0.751,
        expected_force=numpy.array([-0.518, -0.65, -2.285]),
        expected_torque=numpy.array([4.223, -0.398, 3.08]),
    )

    # point 4: theta, phi, and beta out of bound
    sim, pot = run(
        r0=2.11254315,
        rho=0.0,
        theta=2 * numpy.pi / 7,
        phi=2 * numpy.pi / 3,
        alpha=2 * numpy.pi / 15,
        beta=numpy.pi / 3,
        gamma=numpy.pi / 5,
    )
    check_pair(
        sim,
        pot,
        expected_energy=-0.427,
        expected_force=numpy.array([-1.256, -1.575, 1.163]),
        expected_torque=numpy.array([4.872, -0.906, 0.398]),
    )

    # point 5: beta at lower boundary, gamma outside the domain
    sim, pot = run(
        r0=1.0,
        rho=0.2,
        theta=numpy.pi / 4,
        phi=numpy.pi / 4,
        alpha=2 * numpy.pi / 5,
        beta=beta_min,
        gamma=numpy.pi,
    )
    check_pair(
        sim,
        pot,
        expected_energy=-0.41,
        expected_force=numpy.array([-2.023, -2.023, -2.861]),
        expected_torque=numpy.array([6.31798953e05, -2.05283924e05, -0.271]),
    )

    # point 6: all angles outside the domain (except alpha)
    sim, pot = run(
        r0=1.0,
        rho=0.95,
        theta=numpy.pi,
        phi=2 * numpy.pi / 3,
        alpha=2 * numpy.pi / 5,
        beta=2 * numpy.pi / 3,
        gamma=2 * numpy.pi,
    )
    check_pair(
        sim,
        pot,
        expected_energy=2.61,
        expected_force=numpy.array([0.363, 0.0, 0.209]),
        expected_torque=numpy.array([-1.135, 0.369, -0.207]),
    )

    # point 7: equivalent to point 6 but already in the reduced domain
    sim, pot = run(
        r0=1.0,
        rho=0.95,
        theta=0.0,
        phi=1.0471975511965979,
        alpha=1.8849555921538759,
        beta=0.5235987755982987,
        gamma=0.0,
    )
    check_pair(
        sim,
        pot,
        expected_energy=2.61,
        expected_force=numpy.array([-0.363, -0.0, -0.209]),
        expected_torque=numpy.array([1.135, 0.369, 0.207]),
    )

    # point 8: rho > 1, beyond surface cutoff - all zeros
    sim, pot = run(
        r0=1.0,
        rho=1.05,
        theta=numpy.pi / 4,
        phi=numpy.pi / 6,
        alpha=2 * numpy.pi / 5,
        beta=numpy.pi / 2,
        gamma=numpy.pi / 8,
    )
    check_zero_pair(sim, pot)


# Tetrahedron symmetry
def test_chebyshev_force_torque_energy_tetrahedron_symmetry(
    simulation_factory, two_particle_snapshot_factory
):
    """Force, torque, and energy with tetrahedron symmetry reduction.

    Reduced domain: theta in [0, 2 pi/3], phi in [1e-5, pi],
    alpha in [0, 2 pi], beta in [1e-5, pi], gamma in [0, 2 pi/3]."""

    def run(r0, rho, theta, phi, alpha, beta, gamma):
        return build_simulation(
            simulation_factory,
            two_particle_snapshot_factory,
            hoomd.azplugins.pair.ChebyshevAnisotropicPairPotentialTetrahedron,
            r0,
            rho,
            theta,
            phi,
            alpha,
            beta,
            gamma,
        )

    # point 1: interior
    sim, pot = run(
        r0=2.1,
        rho=0.2,
        theta=numpy.pi / 8,
        phi=numpy.pi / 5,
        alpha=2 * numpy.pi / 5,
        beta=numpy.pi / 6,
        gamma=numpy.pi / 3,
    )
    check_pair(
        sim,
        pot,
        expected_energy=-0.41,
        expected_force=numpy.array([-1.437, -0.595, -2.142]),
        expected_torque=numpy.array([6.111, -1.985, -0.271]),
    )

    # point 2: rho < 0 (clamped for derivatives, extrapolated for energy)
    sim, pot = run(
        r0=2.1,
        rho=-0.1,
        theta=numpy.pi / 8,
        phi=numpy.pi / 5,
        alpha=2 * numpy.pi / 5,
        beta=numpy.pi / 6,
        gamma=numpy.pi / 3,
    )
    check_pair(
        sim,
        pot,
        expected_energy=-1.67,
        expected_force=numpy.array([-2.07, -0.857, -3.084]),
        expected_torque=numpy.array([8.013, -2.604, -0.398]),
    )

    # point 3: phi at the boundary and theta is outside the domain
    sim, pot = run(
        r0=1.66,
        rho=0.0,
        theta=3 * numpy.pi / 2,
        phi=phi_min,
        alpha=2 * numpy.pi / 15,
        beta=numpy.pi / 4,
        gamma=numpy.pi / 5,
    )
    check_pair(
        sim,
        pot,
        expected_energy=-0.75,
        expected_force=numpy.array([0.0, 0.0, -3.182]),
        expected_torque=numpy.array([2.084, -4.680, -0.398]),
    )

    # point 4: beta at the bound and gamma out of bound
    sim, pot = run(
        r0=2.1,
        rho=0.65,
        theta=numpy.pi / 4,
        phi=numpy.pi / 4,
        alpha=2 * numpy.pi / 5,
        beta=beta_min,
        gamma=numpy.pi,
    )
    check_pair(
        sim,
        pot,
        expected_energy=1.48,
        expected_force=numpy.array([-0.649, -0.649, -0.917]),
        expected_torque=numpy.array([1.54802229e05, -5.02982930e04, 0.016]),
    )

    # point 5: phi and beta at boundary, gamma outside the domain
    sim, pot = run(
        r0=2.1,
        rho=0.2,
        theta=numpy.pi / 4,
        phi=numpy.pi - phi_min,
        alpha=2 * numpy.pi / 5,
        beta=numpy.pi - beta_min,
        gamma=numpy.pi,
    )
    check_pair(
        sim,
        pot,
        expected_energy=-0.41,
        expected_force=numpy.array([0.0, 0.0, 2.647]),
        expected_torque=numpy.array([2.57516972e05, -8.36723360e04, -0.271]),
    )

    # point 6: all angles outside the domain (except alpha)
    sim, pot = run(
        r0=1.0,
        rho=0.95,
        theta=numpy.pi,
        phi=2 * numpy.pi / 3,
        alpha=2 * numpy.pi / 5,
        beta=2 * numpy.pi / 3,
        gamma=2 * numpy.pi,
    )
    check_pair(
        sim,
        pot,
        expected_energy=1.873,
        expected_force=numpy.array([0.146, 0.0, 0.084]),
        expected_torque=numpy.array([0.371, -0.120, 0.207]),
    )

    # point 7: equivalent to point 6 but already in the reduced domain
    sim, pot = run(
        r0=1.0,
        rho=0.95,
        theta=1.0471975511965979,
        phi=2.0943951023931953,
        alpha=5.445427266222309,
        beta=2.0943951023931953,
        gamma=0.0,
    )
    check_pair(
        sim,
        pot,
        expected_energy=1.873,
        expected_force=numpy.array([-0.073, -0.127, 0.084]),
        expected_torque=numpy.array([-0.29, -0.261, 0.207]),
    )

    # point 8: rho > 1, beyond surface cutoff - all zeros
    sim, pot = run(
        r0=1.0,
        rho=1.05,
        theta=numpy.pi / 4,
        phi=numpy.pi / 6,
        alpha=2 * numpy.pi / 5,
        beta=numpy.pi / 2,
        gamma=numpy.pi / 8,
    )
    check_zero_pair(sim, pot)
