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

class analyze_group_velocity_tests(unittest.TestCase):
    def setUp(self):
        snap = hoomd.data.make_snapshot(N=2, box=hoomd.data.boxdim(Lx=10,Ly=10,Lz=10), particle_types=['A','B'])
        if hoomd.comm.get_rank() == 0:
            snap.particles.position[:] = [[0,0,-5],[0,0,5]]
            snap.particles.velocity[:] = [[1,-2,3],[-2,4,-6]]
            snap.particles.typeid[:] = [0,1]
            snap.particles.mass[:] = [1,2]

        self.s = hoomd.init.read_snapshot(snap)

    def test_compute_all(self):
        all_ = hoomd.group.all()
        azplugins.analyze.group_velocity(group=all_)
        log = hoomd.analyze.log(filename=None, quantities=['vx_all','vy_all','vz_all'], period=1)
        hoomd.run(1)
        v = [log.query('vx_all'), log.query('vy_all'), log.query('vz_all')]
        np.testing.assert_allclose(v, [-1,2,-3])

    def test_compute_subset(self):
        typeA = hoomd.group.type('A',name='A')
        azplugins.analyze.group_velocity(group=typeA)
        log = hoomd.analyze.log(filename=None, quantities=['vx_A','vy_A','vz_A'], period=1)
        hoomd.run(1)
        v = [log.query('vx_A'), log.query('vy_A'), log.query('vz_A')]
        np.testing.assert_allclose(v, [1,-2,3])

    def test_compute_suffix(self):
        typeB = hoomd.group.type('B',name='B')
        azplugins.analyze.group_velocity(group=typeB,suffix='_foo')
        log = hoomd.analyze.log(filename=None, quantities=['vx_foo','vy_foo','vz_foo'], period=1)
        hoomd.run(1)
        v = [log.query('vx_foo'), log.query('vy_foo'), log.query('vz_foo')]
        np.testing.assert_allclose(v, [-2,4,-6])

    def test_unique_suffix(self):
        all_ = hoomd.group.all()
        azplugins.analyze.group_velocity(group=all_,suffix='_1')
        with self.assertRaises(ValueError):
            azplugins.analyze.group_velocity(group=all_,suffix='_1')

    def tearDown(self):
        del self.s
        hoomd.context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
