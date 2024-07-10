# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2024, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""Test flow fields."""

import hoomd
import numpy
from hoomd.conftest import pickling_check


def test_dpd_general_weight_temperature(simulation_factory,
                                        lattice_snapshot_factory):
    """Test dpd general weight thermostat."""

    # use lattice snapshot to generate a simulation with N=1000 in a box 6 x 6 x 6
    n = 10
    snap = lattice_snapshot_factory(dimensions=3, n=n, a=0.6)
    # randomize  positions
    snap.particles.position[:] = numpy.random.uniform(low=-1.5, high=1.5, size=(snap.particles.N, 3))

    sim = simulation_factory(snap)

    # set up NVE integration to test thermostatting part of DPD General Weight
    integrator = hoomd.md.Integrator(dt=0.01)
    cell = hoomd.md.nlist.Cell(buffer=0.4)
    dpd = hoomd.azplugins.pair.DPDGeneralWeight(nlist=cell, kT=1.5, default_r_cut=1.0)
    # No pair interactions, A = 0
    dpd.params[('A', 'A')] = dict(A=0.0, gamma=4.5,s=0.5)

    sim.operations.integrator = integrator
    integrator.forces = [dpd]

    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.5)

    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    sim.operations.integrator.methods.append(nve)

    # This custom action calculates the temperature from particle velocities
    # Ideally, we should be able to use hoomd.md.compute.ThermodynamicQuantities
    # with a logger function, but I could not figure out how to save the output into
    # an numy array instead of stdout (with Table logger) or file.
    class calc_temperature(hoomd.custom.Action):
        def __init__(self):
            self.kT=[]

        def act(self, timestep):
            snap = self._state.get_snapshot()
            if snap.communicator.rank == 0:
                vel = snap.particles.velocity[:]
                mass = snap.particles.mass[:]
                kin_energy = 1/2.0*mass*numpy.sum(vel**2,axis=1)
            kT = 2.0/(snap.particles.N-1)*numpy.sum(kin_energy)/3.
            self.kT.append(kT)

        def calc_average(self):
            return(numpy.average(self.kT))

    custom_action = calc_temperature()

    custom_op = hoomd.write.CustomWriter(
        action=custom_action, trigger=hoomd.trigger.Periodic(1)
        )

    sim.operations.writers.append(custom_op)
    sim.run(100)

    # average temperature should be close (within a decimal place) to the set value
    av_kT = custom_action.calc_average()
    numpy.testing.assert_almost_equal(av_kT,1.5,1)
