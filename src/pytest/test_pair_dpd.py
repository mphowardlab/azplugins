# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2024, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""Test DPD pair potentials."""

import hoomd
import numpy
import pytest


def test_dpd_temperature(simulation_factory, lattice_snapshot_factory):
    """Test dpd general weight thermostat."""
    # use lattice snapshot to generate a simulation with N=1000 in a box 6 x 6 x 6
    snap = lattice_snapshot_factory(dimensions=3, n=10, a=0.6)
    sim = simulation_factory(snap)
    all_ = hoomd.filter.All()
    sim.state.thermalize_particle_momenta(filter=all_, kT=1.5)

    integrator = hoomd.md.Integrator(dt=0.01)
    sim.operations.integrator = integrator

    # create DPD with no repulsion to test random part only
    cell = hoomd.md.nlist.Cell(buffer=0.4)
    dpd = hoomd.azplugins.pair.DPDGeneralWeight(nlist=cell, kT=1.5, default_r_cut=1.0)
    dpd.params[('A', 'A')] = dict(A=0.0, gamma=4.5, s=0.5)
    integrator.forces.append(dpd)

    # set up NVE integration to test thermostatting part of DPD General Weight
    nve = hoomd.md.methods.ConstantVolume(filter=all_)
    integrator.methods.append(nve)

    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=all_)
    sim.operations.computes.append(thermo)

    sim.run(10)

    num_samples = 100
    kT = numpy.zeros(num_samples)
    for sample in range(num_samples):
        kT[sample] = thermo.kinetic_temperature
        sim.run(1)
    avg_kT = numpy.mean(kT)

    assert avg_kT == pytest.approx(1.5, 0.1)
