# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2025, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""Test Perturbed Lennard-Jones for Evaporation pair potential."""

import hoomd
import hoomd.azplugins
import numpy

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
    snap = hoomd.Snapshot()
    if snap.communicator.rank == 0:
        snap.configuration.box = [20, 20, 20, 0, 0, 0]
        snap.particles.N = 2
        snap.particles.types = ["A"]
        snap.particles.position[:] = [[0, 0, 0], [0, 1.2, 0]]

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
        "attraction_scale_factor_shape": [2, 2],
        "domain": [0.0, 1.0, 0.0, 100.0],
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
    numpy.testing.assert_array_equal(
        evap._attraction_scale_factor_shape,
        valid_args_const["attraction_scale_factor_shape"],
    )

    assert evap.rcut == 3.0
    assert evap.time_scale_factor == 1.0


def test_domain_mismatch(
    valid_args_const, two_particle_snapshot_factory, simulation_factory
):
    snap = two_particle_snapshot_factory
    sim = simulation_factory(snap)

    bad_args = valid_args_const.copy()
    bad_args["domain"] = [0.0, 10.0]

    integrator = hoomd.md.Integrator(dt=0.005)
    sim.operations.integrator = integrator

    evap = hoomd.azplugins.pair.PerturbedLennardJonesEvap(**bad_args)
    sim.operations.integrator.forces = [evap]

    with pytest.raises(RuntimeError):
        sim.run(0)


def test_shape_mismatch(
    valid_args_const, two_particle_snapshot_factory, simulation_factory
):
    snap = two_particle_snapshot_factory
    bad_args = valid_args_const.copy()
    bad_args["attraction_scale_factor_shape"] = [2, 2, 2]

    sim = simulation_factory(snap)

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
    expected_energies = [-0.26728958627492283, -0.26728958627492283]

    forces = evap.forces
    energies = evap.energies

    if sim.device.communicator.rank == 0:
        numpy.testing.assert_allclose(forces, expected_forces)
        numpy.testing.assert_allclose(energies, expected_energies)


@pytest.fixture(
    params=[
        # (time_scale_factor, domain)
        (1.0, [0.0, 1.0, 0.0, 100.0]),
        (2.0, [0.0, 1.0, 0.0, 50.0]),
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
        "energy_shift": False,
        "attraction_scale_factor_data": attraction_factor_table,
        "attraction_scale_factor_shape": attraction_factor_table.shape,
        "domain": domain,
        "variant": hoomd.azplugins.variant.VariantInterpolated(
            [12, 10, 8, 6, 4, 2],
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

    expected_forces = [[0, 1.51500993942280826, 0], [0, -1.51500993942280826, 0]]

    expected_energies = [-0.305155610997203564, -0.305155610997203564]

    forces = evap.forces
    energies = evap.energies
    if sim.device.communicator.rank == 0:
        numpy.testing.assert_allclose(forces, expected_forces)
        numpy.testing.assert_allclose(energies, expected_energies)

    """Test if the potential energy and forces change as expected after a certain time.
    """

    sim.run(100)

    snapshot = sim.state.get_snapshot()
    positions = snapshot.particles.position

    print(f"Particle 1 position: {positions[1]}")

    expected_energies = [-0.04900309081706919, -0.04900309081706919]
    expected_forces = [
        [0.0, 0.24328626764453856, 0.0],
        [0.0, -0.24328626764453856, 0.0],
    ]

    forces = evap.forces
    energies = evap.energies

    if sim.device.communicator.rank == 0:
        numpy.testing.assert_allclose(forces, expected_forces)
        numpy.testing.assert_allclose(energies, expected_energies)
