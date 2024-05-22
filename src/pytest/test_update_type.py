# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2022, Auburn University
# This file is part of the azplugins project, released under the Modified BSD License.

import hoomd
from hoomd import md
hoomd.context.initialize()
try:
    from hoomd import azplugins
except ImportError:
    import azplugins
import unittest
import numpy as np

class update_type_tests(unittest.TestCase):
    def setUp(self):
        snap = hoomd.data.make_snapshot(N=3, box=hoomd.data.boxdim(L=20), particle_types=['A','B','C'])
        if hoomd.comm.get_rank() == 0:
            snap.particles.position[:,2] = (-7.5,0.,7.5)
            snap.particles.typeid[:] = [0,1,0]

        self.s = hoomd.init.read_snapshot(snap)
        self.u = azplugins.update.types(inside='A', outside='B', lo=-5.0, hi=5.0)

    # test for basic initialization and call of cpp updater
    def test_basic(self):
        self.assertEqual(self.u.cpp_updater.inside, 0)
        self.assertEqual(self.u.cpp_updater.outside, 1)
        self.assertAlmostEqual(self.u.cpp_updater.lo, -5.0, 5)
        self.assertAlmostEqual(self.u.cpp_updater.hi, 5.0, 5)

        # should not throw
        self.u.cpp_updater.update(0)

    # test error handling for inside type parameter
    def test_set_inside(self):
        # set params should switch inside type and leave others alone
        self.u.set_params(inside='C')
        self.assertEqual(self.u.cpp_updater.inside, 2)
        self.assertEqual(self.u.cpp_updater.outside, 1)
        self.assertAlmostEqual(self.u.cpp_updater.lo, -5.0, 5)
        self.assertAlmostEqual(self.u.cpp_updater.hi, 5.0, 5)

        # update call should still be safe
        self.u.cpp_updater.update(0)

        # cannot have same type inside and outside types
        with self.assertRaises(ValueError):
            self.u.set_params(inside='B')

        # cannot set a nonexistent type
        with self.assertRaises(ValueError):
            self.u.set_params(inside='D')

    # test error handling for outside type parameter
    def test_set_outside(self):
        # set params should switch outside type and leave others alone
        self.u.set_params(outside='C')
        self.assertEqual(self.u.cpp_updater.inside, 0)
        self.assertEqual(self.u.cpp_updater.outside, 2)
        self.assertAlmostEqual(self.u.cpp_updater.lo, -5.0, 5)
        self.assertAlmostEqual(self.u.cpp_updater.hi, 5.0, 5)

        # update call should still be safe
        self.u.cpp_updater.update(0)

        # cannot have same type inside and outside types
        with self.assertRaises(ValueError):
            self.u.set_params(outside='A')

        # cannot set a nonexistent type
        with self.assertRaises(ValueError):
            self.u.set_params(outside='D')

    # test error handling for lo parameter
    def test_set_lo(self):
        # set params should update lo of region and leave others alone
        self.u.set_params(lo=-7.0)
        self.assertEqual(self.u.cpp_updater.inside, 0)
        self.assertEqual(self.u.cpp_updater.outside, 1)
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
        self.assertEqual(self.u.cpp_updater.inside, 0)
        self.assertEqual(self.u.cpp_updater.outside, 1)
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
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_equal(snap.particles.typeid[:], [0,1,0])

        # first run should flip all of the types
        hoomd.run(1)
        snap = self.s.take_snapshot(all=True)
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_equal(snap.particles.typeid, [1,0,1])

        # migrate position, which should trigger a type change
        if hoomd.comm.get_rank() == 0:
            snap.particles.position[:,2] = [7.5, -7.5, 0]
        self.s.restore_snapshot(snap)
        hoomd.run(1)
        snap = self.s.take_snapshot(all=True)
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_equal(snap.particles.typeid, [1,1,0])

        # reset types, but now make one particle be a type that is not included
        if hoomd.comm.get_rank() == 0:
            snap.particles.typeid[:] = (0,1,2)
        self.s.restore_snapshot(snap)
        hoomd.run(1)
        snap = self.s.take_snapshot(all=True)
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_equal(snap.particles.typeid, [1,1,2])

    # test box change signal
    def test_box_change(self):
        snap = self.s.take_snapshot(all=True)
        if hoomd.comm.get_rank() == 0:
            snap.particles.position[:,2] = [-2., 0., 2.]
        self.s.restore_snapshot(snap)
        hoomd.run(1)

        # shrink box smaller than region, which should trigger signal to check
        # box and cause a runtime error
        hoomd.update.box_resize(L=5.0, period=None)
        with self.assertRaises(RuntimeError):
            hoomd.run(1)

    def tearDown(self):
        del self.s, self.u
        hoomd.context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
