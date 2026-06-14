# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2025, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""Chebyshev anisotropic pair potential unit tests."""

import collections

import numpy
from scipy.spatial.transform import Rotation

import hoomd
import hoomd.azplugins

import pytest


_DEVICE_PARAMS = ["cpu"]

if hoomd.version.gpu_enabled:
    try:
        if len(hoomd.device.GPU.get_available_devices()) > 0:
            _DEVICE_PARAMS.append("gpu")
    except Exception:
        pass


@pytest.fixture(params=_DEVICE_PARAMS)
def simulation_factory(request):
    """Create a Simulation on CPU, and on GPU when available."""

    def make_simulation(snapshot):
        if request.param == "cpu":
            device = hoomd.device.CPU()
        else:
            device = hoomd.device.GPU()

        sim = hoomd.Simulation(device=device, seed=1)
        sim.create_state_from_snapshot(snapshot)
        return sim

    return make_simulation


@pytest.fixture
def two_particle_snapshot_factory():
    """Create a basic 2-particle snapshot for pair-potential tests."""

    def make_snapshot():
        snap = hoomd.Snapshot()

        if snap.communicator.rank == 0:
            snap.configuration.box = [20, 20, 20, 0, 0, 0]
            snap.particles.N = 2
            snap.particles.types = ["A"]
            snap.particles.typeid[:] = [0, 0]
            snap.particles.position[:] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            snap.particles.orientation[:] = [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
            snap.particles.moment_inertia[:] = [0.0, 0.0, 0.0]

        return snap

    return make_snapshot


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


# Parameters that are identical across every test.
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

PotentialTestCase = collections.namedtuple(
    "PotentialTestCase",
    [
        "name",
        "potential",
        "r0",
        "rho",
        "theta",
        "phi",
        "alpha",
        "beta",
        "gamma",
        "energy",
        "force",
        "torque",
        "zero_output",
    ],
)

potential_tests = []

# Null symmetry
potential_tests += [
    PotentialTestCase(
        "null_point_1",
        hoomd.azplugins.pair.ChebyshevAnisotropicPairPotential,
        2.1,
        0.2,
        numpy.pi / 4,
        numpy.pi / 4,
        2 * numpy.pi / 5,
        numpy.pi / 2,
        numpy.pi,
        -0.41,
        (-1.324, -1.324, -1.872),
        (0.944, -0.307, -0.271),
        False,
    ),
    PotentialTestCase(
        "null_point_2",
        hoomd.azplugins.pair.ChebyshevAnisotropicPairPotential,
        2.1,
        -0.1,
        numpy.pi / 4,
        numpy.pi / 4,
        2 * numpy.pi / 5,
        numpy.pi / 2,
        numpy.pi,
        -1.67,
        (-1.906, -1.906, -2.695),
        (1.226, -0.398, -0.398),
        False,
    ),
    PotentialTestCase(
        "null_point_3",
        hoomd.azplugins.pair.ChebyshevAnisotropicPairPotential,
        2.1,
        0.0,
        numpy.pi / 4,
        numpy.pi - phi_min,
        2 * numpy.pi / 15,
        numpy.pi / 2,
        numpy.pi,
        -1.583,
        (0.0, 0.0, 3.832),
        (0.546, -1.226, -0.398),
        False,
    ),
    PotentialTestCase(
        "null_point_4",
        hoomd.azplugins.pair.ChebyshevAnisotropicPairPotential,
        2.1,
        0.2,
        numpy.pi / 4,
        numpy.pi / 4,
        2 * numpy.pi / 5,
        beta_min,
        numpy.pi,
        -0.41,
        (-1.324, -1.324, -1.872),
        (120148.0, -39038.6, -0.271),
        False,
    ),
    PotentialTestCase(
        "null_point_5",
        hoomd.azplugins.pair.ChebyshevAnisotropicPairPotential,
        1.1375,
        0.95,
        numpy.pi / 4,
        numpy.pi / 6,
        2 * numpy.pi / 5,
        numpy.pi / 2,
        numpy.pi / 8,
        2.74,
        (-0.174, -0.174, -0.427),
        (0.207, -0.067, 0.207),
        False,
    ),
    PotentialTestCase(
        "null_point_6",
        hoomd.azplugins.pair.ChebyshevAnisotropicPairPotential,
        1.1375,
        1.05,
        numpy.pi / 4,
        numpy.pi / 6,
        2 * numpy.pi / 5,
        numpy.pi / 2,
        numpy.pi / 8,
        0.0,
        None,
        None,
        True,
    ),
]

# Cube symmetry
potential_tests += [
    PotentialTestCase(
        "cube_point_1",
        hoomd.azplugins.pair.ChebyshevAnisotropicPairPotentialCube,
        2.46666667,
        0.2,
        numpy.pi / 8,
        numpy.pi / 5,
        2 * numpy.pi / 5,
        numpy.pi / 6,
        numpy.pi / 3,
        -0.41,
        (-1.335, -0.553, -1.989),
        (7.395, -2.403, -0.271),
        False,
    ),
    PotentialTestCase(
        "cube_point_2",
        hoomd.azplugins.pair.ChebyshevAnisotropicPairPotentialCube,
        2.46666667,
        -0.1,
        numpy.pi / 8,
        numpy.pi / 5,
        2 * numpy.pi / 5,
        numpy.pi / 6,
        numpy.pi / 3,
        -1.67,
        (-1.875, -0.777, -2.793),
        (9.579, -3.113, -0.398),
        False,
    ),
    PotentialTestCase(
        "cube_point_3",
        hoomd.azplugins.pair.ChebyshevAnisotropicPairPotentialCube,
        2.62072583,
        0.0,
        2 * numpy.pi / 7,
        numpy.pi / 9,
        2 * numpy.pi / 15,
        numpy.pi / 4,
        numpy.pi / 5,
        -0.751,
        (-0.518, -0.65, -2.285),
        (4.223, -0.398, 3.08),
        False,
    ),
    PotentialTestCase(
        "cube_point_4",
        hoomd.azplugins.pair.ChebyshevAnisotropicPairPotentialCube,
        2.11254315,
        0.0,
        2 * numpy.pi / 7,
        2 * numpy.pi / 3,
        2 * numpy.pi / 15,
        numpy.pi / 3,
        numpy.pi / 5,
        -0.427,
        (-1.256, -1.575, 1.163),
        (4.872, -0.906, 0.398),
        False,
    ),
    PotentialTestCase(
        "cube_point_5",
        hoomd.azplugins.pair.ChebyshevAnisotropicPairPotentialCube,
        1.0,
        0.2,
        numpy.pi / 4,
        numpy.pi / 4,
        2 * numpy.pi / 5,
        beta_min,
        numpy.pi,
        -0.41,
        (-2.023, -2.023, -2.861),
        (6.31798953e05, -2.05283924e05, -0.271),
        False,
    ),
    PotentialTestCase(
        "cube_point_6",
        hoomd.azplugins.pair.ChebyshevAnisotropicPairPotentialCube,
        1.0,
        0.95,
        numpy.pi,
        2 * numpy.pi / 3,
        2 * numpy.pi / 5,
        2 * numpy.pi / 3,
        2 * numpy.pi,
        2.61,
        (0.363, 0.0, 0.209),
        (-1.135, 0.369, -0.207),
        False,
    ),
    PotentialTestCase(
        "cube_point_7",
        hoomd.azplugins.pair.ChebyshevAnisotropicPairPotentialCube,
        1.0,
        0.95,
        0.0,
        1.0471975511965979,
        1.8849555921538759,
        0.5235987755982987,
        0.0,
        2.61,
        (-0.363, -0.0, -0.209),
        (1.135, 0.369, 0.207),
        False,
    ),
    PotentialTestCase(
        "cube_point_8",
        hoomd.azplugins.pair.ChebyshevAnisotropicPairPotentialCube,
        1.0,
        1.05,
        numpy.pi / 4,
        numpy.pi / 6,
        2 * numpy.pi / 5,
        numpy.pi / 2,
        numpy.pi / 8,
        0.0,
        None,
        None,
        True,
    ),
]

# Tetrahedron symmetry
potential_tests += [
    PotentialTestCase(
        "tetrahedron_point_1",
        hoomd.azplugins.pair.ChebyshevAnisotropicPairPotentialTetrahedron,
        2.1,
        0.2,
        numpy.pi / 8,
        numpy.pi / 5,
        2 * numpy.pi / 5,
        numpy.pi / 6,
        numpy.pi / 3,
        -0.41,
        (-1.437, -0.595, -2.142),
        (6.111, -1.985, -0.271),
        False,
    ),
    PotentialTestCase(
        "tetrahedron_point_2",
        hoomd.azplugins.pair.ChebyshevAnisotropicPairPotentialTetrahedron,
        2.1,
        -0.1,
        numpy.pi / 8,
        numpy.pi / 5,
        2 * numpy.pi / 5,
        numpy.pi / 6,
        numpy.pi / 3,
        -1.67,
        (-2.07, -0.857, -3.084),
        (8.013, -2.604, -0.398),
        False,
    ),
    PotentialTestCase(
        "tetrahedron_point_3",
        hoomd.azplugins.pair.ChebyshevAnisotropicPairPotentialTetrahedron,
        1.66,
        0.0,
        3 * numpy.pi / 2,
        phi_min,
        2 * numpy.pi / 15,
        numpy.pi / 4,
        numpy.pi / 5,
        -0.75,
        (0.0, 0.0, -3.182),
        (2.084, -4.680, -0.398),
        False,
    ),
    PotentialTestCase(
        "tetrahedron_point_4",
        hoomd.azplugins.pair.ChebyshevAnisotropicPairPotentialTetrahedron,
        2.1,
        0.65,
        numpy.pi / 4,
        numpy.pi / 4,
        2 * numpy.pi / 5,
        beta_min,
        numpy.pi,
        1.48,
        (-0.649, -0.649, -0.917),
        (1.54802229e05, -5.02982930e04, 0.016),
        False,
    ),
    PotentialTestCase(
        "tetrahedron_point_5",
        hoomd.azplugins.pair.ChebyshevAnisotropicPairPotentialTetrahedron,
        2.1,
        0.2,
        numpy.pi / 4,
        numpy.pi - phi_min,
        2 * numpy.pi / 5,
        numpy.pi - beta_min,
        numpy.pi,
        -0.41,
        (0.0, 0.0, 2.647),
        (2.57516972e05, -8.36723360e04, -0.271),
        False,
    ),
    PotentialTestCase(
        "tetrahedron_point_6",
        hoomd.azplugins.pair.ChebyshevAnisotropicPairPotentialTetrahedron,
        1.0,
        0.95,
        numpy.pi,
        2 * numpy.pi / 3,
        2 * numpy.pi / 5,
        2 * numpy.pi / 3,
        2 * numpy.pi,
        1.873,
        (0.146, 0.0, 0.084),
        (0.371, -0.120, 0.207),
        False,
    ),
    PotentialTestCase(
        "tetrahedron_point_7",
        hoomd.azplugins.pair.ChebyshevAnisotropicPairPotentialTetrahedron,
        1.0,
        0.95,
        1.0471975511965979,
        2.0943951023931953,
        5.445427266222309,
        2.0943951023931953,
        0.0,
        1.873,
        (-0.073, -0.127, 0.084),
        (-0.29, -0.261, 0.207),
        False,
    ),
    PotentialTestCase(
        "tetrahedron_point_8",
        hoomd.azplugins.pair.ChebyshevAnisotropicPairPotentialTetrahedron,
        1.0,
        1.05,
        numpy.pi / 4,
        numpy.pi / 6,
        2 * numpy.pi / 5,
        numpy.pi / 2,
        numpy.pi / 8,
        0.0,
        None,
        None,
        True,
    ),
]


def rho_to_r(rho, r0, rc):
    """Invert rho = (1/r - 1/r0) / (1/(r0+rc) - 1/r0)."""
    inv_r0 = 1.0 / r0
    inv_r0_rc = 1.0 / (r0 + rc)
    inv_r = rho * (inv_r0_rc - inv_r0) + inv_r0
    return 1.0 / inv_r


@pytest.mark.parametrize("potential_test", potential_tests, ids=lambda x: x.name)
def test_energy_force_and_torque(
    simulation_factory, two_particle_snapshot_factory, potential_test
):
    """Test energy, force, and torque evaluation."""
    snap = two_particle_snapshot_factory()
    if snap.communicator.rank == 0:
        r = rho_to_r(potential_test.rho, potential_test.r0, rc)

        dx = r * numpy.sin(potential_test.phi) * numpy.cos(potential_test.theta)
        dy = r * numpy.sin(potential_test.phi) * numpy.sin(potential_test.theta)
        dz = r * numpy.cos(potential_test.phi)

        q_j = Rotation.from_euler(
            "ZXZ",
            [potential_test.alpha, potential_test.beta, potential_test.gamma],
        ).as_quat(scalar_first=True)

        snap.particles.position[:] = [[0.0, 0.0, 0.0], [-dx, -dy, -dz]]
        snap.particles.orientation[:] = [[1, 0, 0, 0], q_j]
        snap.particles.moment_inertia[:] = [0.1, 0.1, 0.1]

    sim = simulation_factory(snap)

    integrator = hoomd.md.Integrator(dt=0.001)
    nve = hoomd.md.methods.ConstantVolume(hoomd.filter.All())
    integrator.methods = [nve]

    potential = potential_test.potential(
        nlist=hoomd.md.nlist.Cell(buffer=1),
        terms=terms,
        coeffs=coeffs,
        r0=r0_data,
        r_cut=rc,
    )
    integrator.forces = [potential]

    sim.operations.integrator = integrator
    sim.run(0)
    if sim.device.communicator.rank == 0:
        if potential_test.zero_output:
            numpy.testing.assert_allclose(potential.energies, [0.0, 0.0], atol=1e-10)
            numpy.testing.assert_allclose(
                potential.forces,
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                atol=1e-10,
            )
            numpy.testing.assert_allclose(
                potential.torques,
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                atol=1e-10,
            )
        else:
            e = potential_test.energy
            f = numpy.array(potential_test.force)
            T = numpy.array(potential_test.torque)

            numpy.testing.assert_allclose(
                potential.energies,
                [0.5 * e, 0.5 * e],
                atol=1e-3,
                rtol=1e-3,
            )
            numpy.testing.assert_allclose(
                potential.forces,
                [f, -f],
                atol=1e-3,
                rtol=1e-3,
            )
            numpy.testing.assert_allclose(
                potential.torques,
                [T, -T],
                atol=1e-3,
                rtol=1e-3,
            )
