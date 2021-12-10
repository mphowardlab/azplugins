# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021, Auburn University
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

class flow_FlowProfiler_tests(unittest.TestCase):
    def setUp(self):
        snap = hoomd.data.make_snapshot(N=5, box=hoomd.data.boxdim(L=10), particle_types=['A','B','C'])
        if hoomd.comm.get_rank() == 0:
            snap.particles.position[:,2] = (-4.25,2.25,3.0,4.25,4.2)
            snap.particles.velocity[:,0] = (2.0,1.0,-1.0,3.0,1.0)
            snap.particles.typeid[:] = [0,1,0,2,0]

        self.s = hoomd.init.read_snapshot(snap)
        self.u = azplugins.flow.FlowProfiler(system=self.s, bin_axis=2, flow_axis=0, bins=10, range=(-5,5), area=10**2)
        hoomd.analyze.callback(self.u, period=1)

    def test_binning(self):
        hoomd.run(1)

        # test that the bins are correct
        expected_bins =[-4.5, -3.5, -2.5, -1.5, -0.5,  0.5,  1.5,  2.5,  3.5,  4.5]
        np.testing.assert_allclose(self.u.centers, expected_bins)
        for i in range(10):
            self.assertAlmostEqual(expected_bins[i],self.u.centers[i])

        # test that the particles are binned correctly
        bin_volume=10*10*1.0
        expected_densities = np.array([1,0,0,0,0,0,0,1,1,2])/bin_volume
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_allclose(self.u.density, expected_densities)

        # test that the binned values are correct - last bin is averaged velocity 3.0 and 1.0
        expected_velocities = [2.0,0,0,0,0,0,0,1.0,-1.0,2.0]
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_allclose(self.u.velocity, expected_velocities)

    def test_missing_params(self):
        with self.assertRaises(TypeError):
            self.u = azplugins.flow.FlowProfiler(system=self.s, flow_axis=0, bins=10, range=(-5,5), area=10**2)
        with self.assertRaises(TypeError):
            self.u = azplugins.flow.FlowProfiler(system=self.s, bin_axis=2, bins=10, range=(-5,5), area=10**2)
        with self.assertRaises(TypeError):
            self.u = azplugins.flow.FlowProfiler(system=self.s, bin_axis=2, flow_axis=0)
        with self.assertRaises(ValueError):
            self.u = azplugins.flow.FlowProfiler(system=self.s, flow_axis=56, bin_axis=2,bins=10,range=(-5,5))
        with self.assertRaises(ValueError):
            self.u = azplugins.flow.FlowProfiler(system=self.s, flow_axis=0, bin_axis=28,bins=10,range=(-5,5))

    def tearDown(self):
        del self.s, self.u
        hoomd.context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
