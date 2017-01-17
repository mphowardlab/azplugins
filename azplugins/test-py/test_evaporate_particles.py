# Copyright (c) 2016-2017, Panagiotopoulos Group, Princeton University
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: mphoward

from hoomd import *
from hoomd import md
context.initialize()
try:
    from hoomd import azplugins
except ImportError:
    import azplugins
import unittest
import numpy as np

class evaporate_particles_tests(unittest.TestCase):
    def setUp(self):
        snap = data.make_snapshot(N=4, box=data.boxdim(L=20), particle_types=['A','B','C'])
        if comm.get_rank() == 0:
            snap.particles.position[:,0] = (-5.,-5.,5.,5.)
            snap.particles.position[:,2] = (-7.5,2.5,4.5,3.)
            snap.particles.typeid[:] = [0,1,0,2]

        if comm.get_num_ranks() > 1:
            comm.decomposition(nx=2, ny=1, nz=1)

        self.s = init.read_snapshot(snap)
        self.u = azplugins.evaporate.particles(solvent='A', evaporated='B', lo=-5.0, hi=5.0, seed=42)

    # test for basic initialization and call of cpp updater
    def test_basic(self):
        self.assertEqual(self.u.cpp_updater.outside, 0)
        self.assertEqual(self.u.cpp_updater.inside, 1)
        self.assertAlmostEqual(self.u.cpp_updater.lo, -5.0, 5)
        self.assertAlmostEqual(self.u.cpp_updater.hi, 5.0, 5)

        # should not throw
        self.u.cpp_updater.update(0)

    # test error handling for solvent type parameter
    def test_set_solvent(self):
        # set params should switch inside type and leave others alone
        self.u.set_params(solvent='C')
        self.assertEqual(self.u.cpp_updater.outside, 2)
        self.assertEqual(self.u.cpp_updater.inside, 1)
        self.assertAlmostEqual(self.u.cpp_updater.lo, -5.0, 5)
        self.assertAlmostEqual(self.u.cpp_updater.hi, 5.0, 5)

        # update call should still be safe
        self.u.cpp_updater.update(0)

        # cannot have same type inside and outside types
        with self.assertRaises(ValueError):
            self.u.set_params(solvent='B')

        # cannot set a nonexistent type
        with self.assertRaises(ValueError):
            self.u.set_params(solvent='D')

    # test error handling for evaporated type parameter
    def test_set_evaporated(self):
        # set params should switch outside type and leave others alone
        self.u.set_params(evaporated='C')
        self.assertEqual(self.u.cpp_updater.outside, 0)
        self.assertEqual(self.u.cpp_updater.inside, 2)
        self.assertAlmostEqual(self.u.cpp_updater.lo, -5.0, 5)
        self.assertAlmostEqual(self.u.cpp_updater.hi, 5.0, 5)

        # update call should still be safe
        self.u.cpp_updater.update(0)

        # cannot have same type inside and outside types
        with self.assertRaises(ValueError):
            self.u.set_params(evaporated='A')

        # cannot set a nonexistent type
        with self.assertRaises(ValueError):
            self.u.set_params(evaporated='D')

    # test error handling for lo parameter
    def test_set_lo(self):
        # set params should update lo of region and leave others alone
        self.u.set_params(lo=-7.0)
        self.assertEqual(self.u.cpp_updater.outside, 0)
        self.assertEqual(self.u.cpp_updater.inside, 1)
        self.assertAlmostEqual(self.u.cpp_updater.lo, -7.0, 5)
        self.assertAlmostEqual(self.u.cpp_updater.hi, 5.0, 5)

        # update call is still safe
        self.u.cpp_updater.update(0)

        # region cannot lie above hi
        with self.assertRaises(ValueError):
            self.u.set_params(lo=6.0)

        # region cannot lie outside box, caught at runtime
        self.u.set_params(lo=-11.0)
        with self.assertRaises(RuntimeError):
            self.u.cpp_updater.update(1)

    # test error handling for hi parameter
    def test_set_hi(self):
        # set params should update hi of region and leave others alone
        self.u.set_params(hi=7.0)
        self.assertEqual(self.u.cpp_updater.outside, 0)
        self.assertEqual(self.u.cpp_updater.inside, 1)
        self.assertAlmostEqual(self.u.cpp_updater.lo, -5.0, 5)
        self.assertAlmostEqual(self.u.cpp_updater.hi, 7.0, 5)

        # update call is still safe
        self.u.cpp_updater.update(0)

        # region cannot lie below lo
        with self.assertRaises(ValueError):
            self.u.set_params(hi=-6.0)

        # region cannot lie outside box, caught at runtime
        self.u.set_params(hi=11.0)
        with self.assertRaises(RuntimeError):
            self.u.cpp_updater.update(1)

    # test details of the type changing algorithm
    def test_change_types(self):
        # require initial types are set correctly
        snap = self.s.take_snapshot(all=True)
        if comm.get_rank() == 0:
            np.testing.assert_array_equal(snap.particles.typeid[:], [0,1,0,2])

        # first run should evaporate the last particle
        run(1)
        snap = self.s.take_snapshot(all=True)
        if comm.get_rank() == 0:
            np.testing.assert_array_equal(snap.particles.typeid, [0,1,1,2])

        # make sure multiple particles can be evaporated at once
        if comm.get_rank() == 0:
            snap.particles.typeid[:-1] = (0,0,0)
        self.s.restore_snapshot(snap)
        run(1)
        snap = self.s.take_snapshot(all=True)
        if comm.get_rank() == 0:
            np.testing.assert_array_equal(snap.particles.typeid, [0,1,1,2])

        # make sure at least one particle gets evaporated with limit set
        # this is a random choice of the number generator, so we just count the number of ones
        self.u.set_params(Nmax=1)
        if comm.get_rank() == 0:
            snap.particles.typeid[:-1] = (0,0,0)
        self.s.restore_snapshot(snap)
        run(1)
        snap = self.s.take_snapshot(all=True)
        if comm.get_rank() == 0:
            self.assertEqual(list(snap.particles.typeid).count(1), 1)

    # test that nothing quirky happens with empty region
    def test_empty_region(self):
        # check first with particles in region, but none of type to process
        snap = self.s.take_snapshot(all=True)
        if comm.get_rank() == 0:
            snap.particles.typeid[:] = (2,2,2,2)
        self.s.restore_snapshot(snap)
        run(1)
        snap = self.s.take_snapshot(all=True)
        if comm.get_rank() == 0:
            np.testing.assert_array_equal(snap.particles.typeid, [2,2,2,2])

        # now check with particles of solvent type, but outside evaporation region
        if comm.get_rank() == 0:
            snap.particles.position[:,2] = (-7.5, -7.5, -7.5, -7.5)
            snap.particles.typeid[:] = (0,0,0,0)
        self.s.restore_snapshot(snap)
        run(1)
        snap = self.s.take_snapshot(all=True)
        if comm.get_rank() == 0:
            np.testing.assert_array_equal(snap.particles.typeid, (0,0,0,0))

    # test box change signal
    def test_box_change(self):
        snap = self.s.take_snapshot(all=True)
        if comm.get_rank() == 0:
            snap.particles.position[:,2] = [-2., 0., 2. ,0.]
        self.s.restore_snapshot(snap)
        run(1)

        # shrink box smaller than region, which should trigger signal to check
        # box and cause a runtime error
        update.box_resize(L=5.0, period=None)
        with self.assertRaises(RuntimeError):
            run(1)

    def tearDown(self):
        del self.s, self.u
        context.initialize()

class evaporate_particles_big_test(unittest.TestCase):
    # put all particles into the deletion zone
    def setUp(self):
        snap = data.make_snapshot(N=5000, box=data.boxdim(L=20), particle_types=['A','B'])
        if comm.get_rank() == 0:
            # some in -x, some in +x in MPI
            snap.particles.position[:500,0] = -5.
            snap.particles.position[500:,0] = 5.
            snap.particles.position[:,2] = 0.
            snap.particles.typeid[:] = np.zeros(snap.particles.N)

        if comm.get_num_ranks() > 1:
            comm.decomposition(nx=2, ny=1, nz=1)

        self.s = init.read_snapshot(snap)
        self.u = azplugins.evaporate.particles(solvent='A', evaporated='B', lo=-5.0, hi=5.0, seed=771991)

    # test that all particles can be evaporated at once
    def test_all(self):
        run(1)
        snap = self.s.take_snapshot(all=True)
        if comm.get_rank() == 0:
            np.testing.assert_array_equal(snap.particles.typeid, np.ones(snap.particles.N))

    def test_limit(self):
        self.u.set_params(Nmax=50)

        run(1)
        snap = self.s.take_snapshot(all=True)
        if comm.get_rank() == 0:
            self.assertEqual(list(snap.particles.typeid).count(1), 50)

        run(1)
        snap = self.s.take_snapshot(all=True)
        if comm.get_rank() == 0:
            self.assertEqual(list(snap.particles.typeid).count(1), 100)

        self.u.set_params(Nmax=900)
        run(1)
        snap = self.s.take_snapshot(all=True)
        if comm.get_rank() == 0:
            self.assertEqual(list(snap.particles.typeid).count(1), 1000)

        self.u.set_params(Nmax=4000)
        run(1)
        snap = self.s.take_snapshot(all=True)
        if comm.get_rank() == 0:
            self.assertEqual(list(snap.particles.typeid).count(1), 5000)
            np.testing.assert_array_equal(snap.particles.typeid, np.ones(snap.particles.N))

        run(1)
        snap = self.s.take_snapshot(all=True)
        if comm.get_rank() == 0:
            np.testing.assert_array_equal(snap.particles.typeid, np.ones(snap.particles.N))

    def tearDown(self):
        del self.s, self.u
        context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
