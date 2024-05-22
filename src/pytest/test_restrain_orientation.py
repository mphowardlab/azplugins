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

# define quaternion rotation operations
def normalize(v, tolerance=0.00001):
    mag2 = np.sum(n * n for n in v)
    if np.abs(mag2 - 1.0) > tolerance:
        mag = sqrt(mag2)
        v = np.array(n / mag for n in v)
    return v

def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.array([w, x, y, z])

def q_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def qv_mult(q1, v1):
    q2 = (0.0,) + v1
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]

def axisangle_to_q(v, theta):
    v = normalize(v)
    x, y, z = v
    theta /= 2
    w = np.cos(theta)
    x = x * np.sin(theta)
    y = y * np.sin(theta)
    z = z * np.sin(theta)
    return np.array([w, x, y, z])

# azplugins.restrain.orientation
class restrain_orientation_tests(unittest.TestCase):
    def setUp(self):
        snap = hoomd.data.make_snapshot(N=100, box=hoomd.data.boxdim(L=20), particle_types=['A'])
        snap.particles.moment_inertia[:] = (0.1,0.1,0.1) # 2/5 m r^2 = 2/20 m sigma^2
        snap.particles.angmom[:] = 0
        self.s = hoomd.init.read_snapshot(snap)

    # basic test of creation
    def test_basic(self):
        field = azplugins.restrain.orientation(hoomd.group.all(),k=1)
        field = azplugins.restrain.orientation(hoomd.group.all(),k=1.)

    # test creation with bad inputs
    def test_bad_args(self):
        self.assertRaises(ValueError, azplugins.restrain.orientation, hoomd.group.all(), k="one")

    # test potential calculation
    def test_potential(self):
        k = 1.
        r = axisangle_to_q((0,0,1),0)
        for i in range(len(self.s.particles)):
            self.s.particles[i].orientation = r
        field = azplugins.restrain.orientation(hoomd.group.all(),k=k)
        for i in range(10):
            rads = (i * 10.)/180.*np.pi
            p = axisangle_to_q((0,0,1),rads)
            self.s.particles[i].orientation = p
        # integrator with zero timestep to compute forces
        md.integrate.mode_standard(dt=0)
        md.integrate.nve(group = hoomd.group.all())
        hoomd.run(0)
        for i in range(10):
            rads = (i * 10.)/180.*np.pi
            p = axisangle_to_q((0,0,1),rads)
            n_i = qv_mult(p,(1,0,0))
            n_ref = qv_mult(r,(1,0,0))
            gamma = np.dot(n_i,n_ref)
            expected_energy = k * ( 1. - gamma * gamma )
            self.assertAlmostEqual(self.s.particles[i].net_energy,expected_energy, 5)
            expected_torque = 2. * k * gamma * np.cross(n_i,n_ref)
            np.testing.assert_array_almost_equal(self.s.particles[i].net_torque,expected_torque, 5)

    def tearDown(self):
        del self.s
        hoomd.context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
