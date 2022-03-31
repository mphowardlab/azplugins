# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2022, Auburn University
# This file is part of the azplugins project, released under the Modified BSD License.

import hoomd
from hoomd import md
from hoomd import mpcd
hoomd.context.initialize()
try:
    from hoomd import azplugins
except ImportError:
    import azplugins
import azplugins.mpcd
import unittest
import numpy as np

class analyze_group_velocity_tests(unittest.TestCase):
    def setUp(self):
        # empty MD system
        md_snap = hoomd.data.make_snapshot(N=0, box=hoomd.data.boxdim(L=10))
        hoomd.init.read_snapshot(md_snap)

        # MPCD system
        snap = mpcd.data.make_snapshot(N=2)
        if hoomd.comm.get_rank() == 0:
            snap.particles.position[:] = [[0,0,-5],[0,0,5]]
            snap.particles.velocity[:] = [[1,-2,3],[-3,6,-9]]
        self.s = mpcd.init.read_snapshot(snap)

    def test_compute(self):
        azplugins.mpcd.compute_velocity()
        log = hoomd.analyze.log(
            filename=None,
            quantities=['mpcd_vx','mpcd_vy','mpcd_vz'],
            period=1
            )
        hoomd.run(1)
        v = [
            log.query('mpcd_vx'),
            log.query('mpcd_vy'),
            log.query('mpcd_vz')
            ]
        np.testing.assert_allclose(v, [-1,2,-3])

    def test_compute_suffix(self):
        azplugins.mpcd.compute_velocity(suffix='_foo')
        log = hoomd.analyze.log(
            filename=None,
            quantities=['mpcd_vx_foo','mpcd_vy_foo','mpcd_vz_foo'],
            period=1
            )
        hoomd.run(1)
        v = [
            log.query('mpcd_vx_foo'),
            log.query('mpcd_vy_foo'),
            log.query('mpcd_vz_foo')
            ]
        np.testing.assert_allclose(v, [-1,2,-3])

    def test_unique_suffix(self):
        azplugins.mpcd.compute_velocity(suffix='_1')
        with self.assertRaises(ValueError):
            azplugins.mpcd.compute_velocity(suffix='_1')

    def tearDown(self):
        del self.s
        hoomd.context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
