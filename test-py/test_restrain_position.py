# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021, Auburn University
# This file is part of the azplugins project, released under the Modified BSD License.

import hoomd
import hoomd
from hoomd import md
hoomd.context.initialize()
try:
    from hoomd import azplugins
except ImportError:
    import azplugins
import unittest

import numpy as np

# azplugins.restrain.position
class restrain_position_tests(unittest.TestCase):
    def setUp(self):
        snap = hoomd.data.make_snapshot(N=100, box=hoomd.data.boxdim(L=20), particle_types=['A'])
        self.s = hoomd.init.read_snapshot(snap)

    # basic test of creation
    def test_basic(self):
        springs = azplugins.restrain.position(hoomd.group.all(),k=1,r_cut=1000.0)
        springs = azplugins.restrain.position(hoomd.group.all(),k=1.,r_cut=1000.0)
        springs = azplugins.restrain.position(hoomd.group.all(),k=(1,2,3),r_cut=1000.0)
        springs = azplugins.restrain.position(hoomd.group.all(),k=(1.,2.,3.),r_cut=1000.0)

    # test creation with bad inputs
    def test_bad_args(self):
        self.assertRaises(ValueError, azplugins.restrain.position, hoomd.group.all(), k="one", r_cut=1000.0 )
        self.assertRaises(ValueError, azplugins.restrain.position, hoomd.group.all(), k=(1.,2,"three"), r_cut=1000.0)

    # test potential calculation
    def test_potential(self):
        k = 1.
        r_cut=1000.0
        for i in range(10):
            self.s.particles[i].position = [0.,0.,0.]
        springs = azplugins.restrain.position(hoomd.group.all(),k=k,r_cut=r_cut)
        for i in range(10):
            self.s.particles[i].position = [i,i,i]
        # integrator with zero timestep to compute forces
        md.integrate.mode_standard(dt=0)
        md.integrate.nve(group = hoomd.group.all())
        hoomd.run(0)
        for i in range(10):
            expected_energy = 3.0*i*i*0.5*k
            self.assertAlmostEqual(self.s.particles[i].net_energy,expected_energy)
            expected_force  = -i*k*np.ones( (3) )
            np.testing.assert_array_almost_equal(self.s.particles[i].net_force,expected_force)

    # test potential calculation with (uneven) tuple
    def test_potential_tuple(self):
        k = (1., 2., 3.)
        r_cut=1000.0
        for i in range(10):
            self.s.particles[i].position = [0.,0.,0.]
        springs = azplugins.restrain.position(hoomd.group.all(),k=k,r_cut=1000.0)
        for i in range(10):
            x = i
            y = int(i/2)
            z = int(i/3)
            self.s.particles[i].position = [x,y,z]
        # integrator with zero timestep to compute forces
        md.integrate.mode_standard(dt=0)
        md.integrate.nve(group = hoomd.group.all())
        hoomd.run(0)
        for i in range(10):
            x = i
            y = int(i/2)
            z = int(i/3)
            expected_energy = x*x*0.5*k[0] + y*y*0.5*k[1] + z*z*0.5*k[2]
            self.assertAlmostEqual(self.s.particles[i].net_energy,expected_energy)
            expected_force = np.asarray([-x*k[0], -y*k[1], -z*k[2]])
            np.testing.assert_array_almost_equal(self.s.particles[i].net_force,expected_force)

    def tearDown(self):
        del self.s
        hoomd.context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
