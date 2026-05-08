# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2025, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

import hoomd
import pytest
import numpy


def test_setters_getters(simulation_factory, one_particle_snapshot_factory):

    # make one particle test configuration
    sim = simulation_factory(one_particle_snapshot_factory(position=[0,0,0], L=20))

    u = hoomd.azplugins.update.dynamic_bond(nlist=hoomd.md.nlist.Cell(buffer=0.4),
                                                r_cut=1,
                                                trigger = hoomd.trigger.Periodic(period=10),
                                                bond_type=0,
                                                group_1=hoomd.filter.All(),
                                                group_2=hoomd.filter.All(),
                                                max_bonds_group_1=0,
                                                max_bonds_group_2=0)

    assert numpy.equal(u.r_cut, 1)
    u.r_cut = 1.5
    assert numpy.equal(u.r_cut, 1.5)

    u.max_bonds_group_1 = 3
    assert numpy.equal(u.max_bonds_group_1,3)

    u.max_bonds_group_2 = 7
    assert numpy.equal(u.max_bonds_group_2,7)

    # todo: this check doesn't work but it definitely throws an error - maybe wrong way to test?
    # check the test of box size large enough for cutoff
    #with pytest.raises(RuntimeError):
    #    u.r_cut=15.0

def test_form_bonds_same_group(simulation_factory):

    snap = hoomd.Snapshot()
    if snap.communicator.rank == 0:
        snap.configuration.box = [20, 20, 20, 0, 0, 0]
        snap.particles.N = 6
        snap.particles.types = ["A","B"]
        snap.particles.position[:,0] = (0,1,2,3,4,6)
        snap.particles.position[:,1] = (0,0,0,0,0,0)
        snap.particles.position[:,2] = (0,0,0,0,0,0)
        # dynamic bond operates on groups, so typeids should not matter at all
        snap.particles.typeid[:] = [0,0,1,1,0,0]
        snap.bonds.types = ['bond']

    sim = simulation_factory(snap)
    nl=hoomd.md.nlist.Cell(buffer=0.4)

    group =hoomd.filter.Tags([0,1,2,3,4,5])
    form_bonds = hoomd.azplugins.update.dynamic_bond(nlist=nl,
                                                r_cut=1.1,
                                                trigger = hoomd.trigger.Periodic(period=1),
                                                bond_type=0,
                                                group_1=group,
                                                group_2=group,
                                                max_bonds_group_1=2,
                                                max_bonds_group_2=2)

    # test bond formation. All are in the same group, so bonds should be formed
    # between 0-1, 1-2, 2-3, and 3-4 (but not 5, too far away)
    integrator = hoomd.md.Integrator(dt=0)
    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    integrator.methods.append(nve)
    sim.operations.updaters.append(form_bonds)
    sim.run(1)
    s = sim.state.get_snapshot()

    numpy.testing.assert_almost_equal(4, s.bonds.N)
    numpy.testing.assert_array_almost_equal([0,1], s.bonds.group[0])
    numpy.testing.assert_array_almost_equal([1,2], s.bonds.group[1])
    numpy.testing.assert_array_almost_equal([2,3], s.bonds.group[2])
    numpy.testing.assert_array_almost_equal([3,4], s.bonds.group[3])


    # same as before, we will have formed bonds 0-1-2-3-4 in the first step, so nothing
    # new should be formed
    sim.run(1)
    s = sim.state.get_snapshot()

    numpy.testing.assert_almost_equal(4, s.bonds.N)
    numpy.testing.assert_array_almost_equal([0,1], s.bonds.group[0])
    numpy.testing.assert_array_almost_equal([1,2], s.bonds.group[1])
    numpy.testing.assert_array_almost_equal([2,3], s.bonds.group[2])
    numpy.testing.assert_array_almost_equal([3,4], s.bonds.group[3])


    # now put particle 5 in range of partice 2 and 3, but no new bonds should
    # be formed since max_bonds_group_1=max_bonds_group_2=2 and that would be the third
    # bond on those particles
    s.particles.position[5]=(2.5,0,0)
    sim.state.set_snapshot(s)
    sim.run(1)
    s = sim.state.get_snapshot()

    numpy.testing.assert_almost_equal(4, s.bonds.N)
    numpy.testing.assert_array_almost_equal([0,1], s.bonds.group[0])
    numpy.testing.assert_array_almost_equal([1,2], s.bonds.group[1])
    numpy.testing.assert_array_almost_equal([2,3], s.bonds.group[2])
    numpy.testing.assert_array_almost_equal([3,4], s.bonds.group[3])


def test_form_bonds_same_group_priority(simulation_factory):

    snap = hoomd.Snapshot()
    if snap.communicator.rank == 0:
        snap.configuration.box = [20, 20, 20, 0, 0, 0]
        snap.particles.N = 6
        snap.particles.types = ["A","B"]
        snap.particles.position[:,0] = (0,1,2,3,4,2.5)
        snap.particles.position[:,1] = (0,0,0,0,0,0)
        snap.particles.position[:,2] = (0,0,0,0,0,0)
        # dynamic bond operates on groups, so typeids should not matter at all
        snap.particles.typeid[:] = [0,0,1,1,0,0]
        snap.bonds.types = ['bond']

    sim = simulation_factory(snap)
    nl=hoomd.md.nlist.Cell(buffer=0.4)

    group =hoomd.filter.Tags([0,1,2,3,4,5])
    form_bonds = hoomd.azplugins.update.dynamic_bond(nlist=nl,
                                                r_cut=1.1,
                                                trigger = hoomd.trigger.Periodic(period=1),
                                                bond_type=0,
                                                group_1=group,
                                                group_2=group,
                                                max_bonds_group_1=2,
                                                max_bonds_group_2=2)

    # test bond formation. All are in the same group, so bonds should be formed
    # between 0-1, 1-2, 2-3, and 3-4 (but not 5, too far away)
    integrator = hoomd.md.Integrator(dt=0)
    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    integrator.methods.append(nve)
    sim.operations.updaters.append(form_bonds)
    sim.run(1)
    s = sim.state.get_snapshot()

    # here, particle 5 is in range of partice 2 and 3, so in principle the bonds
    # 0-1, 1-2 ,2-3, 3-4 (all length 1), 2-5 and 3-5 (length 0.5) can be formed.
    # The all_possible_bonds array  is sorted by bond distance, so bonds 2-5
    # and 3-5 are formed first. Then in order of index 0-1, and 1-2.
    # Particle 2 now has two bonds, so 2-3 can't be formed, but 3-4 can.
    numpy.testing.assert_almost_equal(5, s.bonds.N)
    numpy.testing.assert_array_almost_equal([2,5], s.bonds.group[0])
    numpy.testing.assert_array_almost_equal([3,5], s.bonds.group[1])
    numpy.testing.assert_array_almost_equal([0,1], s.bonds.group[2])
    numpy.testing.assert_array_almost_equal([1,2], s.bonds.group[3])
    numpy.testing.assert_array_almost_equal([3,4], s.bonds.group[4])


def test_update_bond_two_groups(simulation_factory):

    snap = hoomd.Snapshot()
    if snap.communicator.rank == 0:
        snap.configuration.box = [20, 20, 20, 0, 0, 0]
        snap.particles.N = 4
        snap.particles.types = ["A","B"]
        snap.particles.position[:,0] = (0,0.9,1.1,2)
        snap.particles.position[:,1] = (0,0,0,0)
        snap.particles.position[:,2] = (0,0,0,0)
        snap.particles.typeid[:] = [1,0,1,0]
        snap.bonds.types = ['bond']

    sim = simulation_factory(snap)
    nl=hoomd.md.nlist.Cell(buffer=0.4)

    group_1 = hoomd.filter.Tags([0,1])
    group_2 = hoomd.filter.Tags([2,3])
    form_bonds = hoomd.azplugins.update.dynamic_bond(nlist=nl,
                                                r_cut=1.0,
                                                trigger = hoomd.trigger.Periodic(period=1),
                                                bond_type=0,
                                                group_1=group_1,
                                                group_2=group_2,
                                                max_bonds_group_1=1,
                                                max_bonds_group_2=2)

    # test bond formation between particle 1-2
    # particle 0,1 are in the same group, so even if their distance is
    # below r_cut, no bond should be formed, same for goes for particle 2,3
    integrator = hoomd.md.Integrator(dt=0)
    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    integrator.methods.append(nve)
    sim.operations.updaters.append(form_bonds)
    sim.run(1)
    s = sim.state.get_snapshot()

    numpy.testing.assert_almost_equal(1, s.bonds.N)
    numpy.testing.assert_array_almost_equal([1,2], s.bonds.group[0])

    # after this bond 1-2 is formed, there shouldn't be a second one formed
    # e.g. a dublicate, even when the simulation continues to run
    sim.run(10)
    s = sim.state.get_snapshot()

    numpy.testing.assert_almost_equal(1, s.bonds.N)
    numpy.testing.assert_array_almost_equal([1,2], s.bonds.group[0])





