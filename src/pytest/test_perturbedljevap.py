# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2025, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""Test Perturbed Lennard-Jones for Evaporation pair potential."""

import hoomd
import hoomd.azplugins
import numpy

import pytest


@pytest.fixture
def two_particle_snapshot_factory():
    snap = hoomd.Snapshot()
    if snap.communicator.rank == 0:
        snap.configuration.box = [20, 20, 20, 0, 0, 0]
        snap.particles.N = 2
        snap.particles.types = ["A"]
        snap.particles.position[:] = [
            [-10, -10, -10],
            [-10, -8.8, -10],
        ]
    return snap


@pytest.fixture
def valid_args_const():
    return {
        "nlist": hoomd.md.nlist.Cell(buffer=0.4),
        "rcut": 3.0,
        "epsilon": 1.0,
        "sigma": 1.0,
        "time_scale_factor": 1.0,
        "energy_shift": False,
        "attraction_scale_factor_data": numpy.array([[0.6, 0.6], [0.6, 0.6]]),
        "domain": [0.0, 100.0],
        "variant": hoomd.azplugins.variant.VariantInterpolated(
            [5.0, 4.0, 2.0, 1.0], 0, 300
        ),
    }


def test_constructor(valid_args_const):
    evap = hoomd.azplugins.pair.PerturbedLennardJonesEvap(**valid_args_const)

    assert evap.epsilon == 1.0
    assert evap.sigma == 1.0

    assert evap._nlist is valid_args_const["nlist"]
    assert evap._variant is valid_args_const["variant"]

    # Check numpy array casting
    numpy.testing.assert_allclose(evap._domain, valid_args_const["domain"])
    numpy.testing.assert_allclose(
        evap._attraction_scale_factor_data,
        valid_args_const["attraction_scale_factor_data"],
    )

    assert evap.rcut == 3.0
    assert evap.time_scale_factor == 1.0


def test_domain_mismatch(
    valid_args_const, two_particle_snapshot_factory, simulation_factory
):
    snap = two_particle_snapshot_factory
    sim = simulation_factory(snap)

    bad_args = valid_args_const.copy()
    bad_args["domain"] = [10.0]

    integrator = hoomd.md.Integrator(dt=0.005)
    sim.operations.integrator = integrator

    evap = hoomd.azplugins.pair.PerturbedLennardJonesEvap(**bad_args)
    sim.operations.integrator.forces = [evap]

    with pytest.raises(RuntimeError):
        sim.run(0)


def test_data_size_mismatch(
    valid_args_const, two_particle_snapshot_factory, simulation_factory
):
    snap = two_particle_snapshot_factory
    bad_args = valid_args_const.copy()
    bad_args["attraction_scale_factor_data"] = [1.0, 1.0]

    sim = simulation_factory(snap)

    integrator = hoomd.md.Integrator(dt=0.005)
    sim.operations.integrator = integrator

    evap = hoomd.azplugins.pair.PerturbedLennardJonesEvap(**bad_args)
    sim.operations.integrator.forces = [evap]

    with pytest.raises(RuntimeError):
        sim.run(0)


def test_variant_mismatch(
    valid_args_const, two_particle_snapshot_factory, simulation_factory
):
    snap = two_particle_snapshot_factory
    bad_args = valid_args_const.copy()
    bad_args["variant"] = hoomd.variant.Constant(
        1.0
    )  # Invalid: Should be VariantInterpolated

    sim = simulation_factory(snap)

    integrator = hoomd.md.Integrator(dt=0.005)
    sim.operations.integrator = integrator

    evap = hoomd.azplugins.pair.PerturbedLennardJonesEvap(**bad_args)
    sim.operations.integrator.forces = [evap]

    with pytest.raises(TypeError):
        sim.run(0)


"""Test energy and force calculation for single particle type
with constant attraction scalefactor.
"""


def test_energy_and_force_calculation_const(
    valid_args_const, two_particle_snapshot_factory, simulation_factory
):
    snap = two_particle_snapshot_factory
    evap = hoomd.azplugins.pair.PerturbedLennardJonesEvap(**valid_args_const)

    sim = simulation_factory(snap)

    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces = [evap]

    sim.operations.integrator = integrator

    sim.run(0)

    expected_forces = [[0.0, 1.32701601, 0.0], [0.0, -1.32701601, 0.0]]
    expected_energies = [-0.267289586, -0.267289586]

    forces = evap.forces
    energies = evap.energies

    if sim.device.communicator.rank == 0:
        numpy.testing.assert_allclose(forces, expected_forces)
        numpy.testing.assert_allclose(energies, expected_energies)


@pytest.fixture(
    params=[
        # (time_scale_factor, domain)
        (1.0, [0.0, 100.0]),
        (2.0, [0.0, 50.0]),
    ],
    ids=["unscaled_time", "scaled_time"],
)
def valid_args_vary(request):

    time_scale_factor, domain = request.param
    attraction_factor_table = numpy.array(
        [
            [0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            [0.4, 0.3, 0.2, 0.1, 0.0, 0.0],
        ]
    )

    return {
        "nlist": hoomd.md.nlist.Cell(buffer=0.4),
        "rcut": 3.0,
        "epsilon": 1.0,
        "sigma": 1.0,
        "time_scale_factor": time_scale_factor,
        "energy_shift": True,
        "attraction_scale_factor_data": attraction_factor_table,
        "domain": domain,
        "variant": hoomd.azplugins.variant.VariantInterpolated(
            [2, 0, -2, -4, -6, -8],
            0.0,
            100.0,  # Run for 100 timesteps, which corresponds to t = 0.5 for dt = 0.005
        ),
    }


"""Test energy and force calculation for single particle type
with varying attraction scale factor.
"""


def test_energy_and_force_calculation_vary(
    valid_args_vary, two_particle_snapshot_factory, simulation_factory
):
    """Energies/forces at t=0 and t=0.5, with and without time scaling."""
    snap = two_particle_snapshot_factory
    evap = hoomd.azplugins.pair.PerturbedLennardJonesEvap(**valid_args_vary)
    sim = simulation_factory(snap)

    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces = [evap]
    sim.operations.integrator = integrator
    sim.run(0)

    expected_forces = [
        [0, 1.5150099394228083, 0.0],
        [0, -1.5150099394228083, 0.0],
    ]
    expected_energies = [-0.3032789, -0.3032789]

    forces = evap.forces
    energies = evap.energies

    if sim.device.communicator.rank == 0:
        numpy.testing.assert_allclose(forces, expected_forces)
        numpy.testing.assert_allclose(energies, expected_energies)

    """Test if the potential energy and forces change as expected after a certain time.
    """

    sim.run(100)

    expected_forces = [
        [0.0, 0.24328626, 0.0],
        [0.0, -0.24328626, 0.0],
    ]
    expected_energies = [-0.048701721, -0.048701721]

    forces = evap.forces
    energies = evap.energies

    if sim.device.communicator.rank == 0:
        numpy.testing.assert_allclose(forces, expected_forces)
        numpy.testing.assert_allclose(energies, expected_energies)
