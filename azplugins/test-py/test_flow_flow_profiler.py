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

class flow_FlowProfiler_tests(unittest.TestCase):
    def setUp(self):
        snap = hoomd.data.make_snapshot(N=5, box=hoomd.data.boxdim(L=10), particle_types=['A','B','C'])
        if hoomd.comm.get_rank() == 0:
            snap.particles.position[:,2] = (-4.25,2.25,3.0,4.25,4.2)
            snap.particles.velocity[:,0] = (2.0,1.0,-1.0,3.0,1.0)
            snap.particles.typeid[:] = [0,1,0,2,0]
            snap.particles.mass[:] = (0.5,0.5,0.5,0.5,1.0)

        self.s = hoomd.init.read_snapshot(snap)
        self.u = azplugins.flow.FlowProfiler(system=self.s, axis=2, bins=10, range=(-5,5), area=10**2)
        hoomd.analyze.callback(self.u, period=1)

    def test_binning(self):
        hoomd.run(1)

        # test that the bins are correct
        expected_bins =[-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5]
        np.testing.assert_allclose(self.u.edges, expected_bins)
        for i in range(10):
            self.assertAlmostEqual(expected_bins[i],self.u.edges[i])

        # test that the centers are correct
        expected_centers =[-4.5, -3.5, -2.5, -1.5,-0.5,  0.5,  1.5,  2.5,  3.5,  4.5]
        np.testing.assert_allclose(self.u.centers, expected_centers)
        for i in range(10):
            self.assertAlmostEqual(expected_centers[i],self.u.centers[i])

        # test that the particles are binned correctly - number density
        bin_volume=10*10*1.0
        expected_densities = np.array([1,0,0,0,0,0,0,1,1,2])/bin_volume
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_allclose(self.u.number_density, expected_densities)

        # test that the binned values are correct - last bin is averaged velocity 3.0 and 1.0
        expected_velocities = [[2,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],
                               [1,0,0],[-1,0,0],[(3.0+1.0)/2,0,0]]

        if hoomd.comm.get_rank() == 0:
            np.testing.assert_allclose(self.u.number_velocity, expected_velocities)

    def test_mass_averaging(self):
        hoomd.run(1)
        bin_volume=10*10*1.0
        expected_densities = np.array([0.5,0,0,0,0,0,0,0.5,0.5,1+0.5])/bin_volume # last particle is heavier
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_allclose(self.u.mass_density, expected_densities)

        # last bin is mass averaged velocity 3.0 and 1.0 with masses 0.5 and 1.0
        expected_velocities = [[2,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],
                               [1,0,0],[-1,0,0],[(3.0*0.5+1.0*1.0)/(0.5+1.0),0,0]]
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_allclose(self.u.mass_velocity, expected_velocities)

    def test_temperature(self):
        hoomd.run(1)
        # Temperature is calculated via kT= 2(KE-KE_cm)/(3*(n-1)) for each bin
        # all bins have either one or zero particles except for the last one with two. The only temperature we need to
        # calculate is the one for the last bin. KE_cm for the last bin is 2.08333333=0.5*((3.0*0.5+1.0*1.0)/(0.5+1.0))^2*(0.5+1.0)
        # KE for this bin is 2.75=0.5*(0.5*3^2+1^2*1), and 2*(2.75-2.08333333)/3=0.4444=kT for last bin.

        expected_temperature = np.array([0,0,0,0,0,0,0,0,0,0.444444444])
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_allclose(self.u.kT, expected_temperature)


    def test_missing_params(self):
        with self.assertRaises(TypeError):
            self.u = azplugins.flow.FlowProfiler(system=self.s, bins=10.5, range=(-5,5), area=10**2)
        with self.assertRaises(ValueError):
            self.u = azplugins.flow.FlowProfiler(system=self.s, axis=3.0, bins=10, range=(-5,5), area=10**2)
        with self.assertRaises(TypeError):
            self.u = azplugins.flow.FlowProfiler(system=self.s, axis='x')
        with self.assertRaises(TypeError):
            self.u = azplugins.flow.FlowProfiler(system=self.s)
        with self.assertRaises(ValueError):
            self.u = azplugins.flow.FlowProfiler(system=self.s, axis=28,bins=10,range=(-5,5))

    def tearDown(self):
        del self.s, self.u
        hoomd.context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
