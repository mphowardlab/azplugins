# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2022, Auburn University
# This file is part of the azplugins project, released under the Modified BSD License.

import numpy as np

import hoomd
hoomd.context.initialize()
from hoomd import md
try:
    from hoomd import azplugins
except ImportError:
    import azplugins
import unittest

# azplugins.restrain.sphere
class restrain_sphere_tests(unittest.TestCase):
    """Tests azplugins.restrain.sphere"""

    def setUp(self):
        snap = hoomd.data.make_snapshot(N=3, box=hoomd.data.boxdim(L=7.0))
        if hoomd.comm.get_rank() == 0:
            snap.particles.position[0] = [-1,1,1]
            snap.particles.position[1] = [1,3,1]
            snap.particles.position[2] = [1,1,3]
            snap.particles.image[2] = [0,0,-1]
        hoomd.init.read_snapshot(snap)

        # dummy integrator
        all_ = hoomd.group.all()
        md.integrate.mode_standard(dt=0)
        md.integrate.nve(group=all_)

    def test_create(self):
        """Test object creation and updating."""
        f = azplugins.restrain.sphere(group=hoomd.group.all(), radius=2, origin=(0,0,0), k=2.0)

        f.set_params(k=5.0)
        f.set_params(k=8)

        f.set_params(radius=3.)
        f.set_params(radius=4)

        f.set_params(origin=(0,0,1))
        f.set_params(origin=[0,0,1])
        f.set_params(origin=np.array([0,0,1]))

        f.set_params(radius=2., origin=(0,0,0), k=10.0)

    def test_force(self):
        """Test forces computed on particles."""
        group = hoomd.group.all()

        # compute forces
        f = azplugins.restrain.sphere(group=group, radius=1, origin=(1,1,1), k=2.0)
        hoomd.run(1)
        np.testing.assert_array_almost_equal(f.forces[0].force, (2.,0,0))
        np.testing.assert_array_almost_equal(f.forces[1].force, (0,-2.,0))
        np.testing.assert_array_almost_equal(f.forces[2].force, (0,0,8.))
        self.assertAlmostEqual(f.forces[0].energy, 1.)
        self.assertAlmostEqual(f.forces[1].energy, 1.)
        self.assertAlmostEqual(f.forces[2].energy, 16.)
        np.testing.assert_array_almost_equal(f.forces[0].virial, (-2.,2.,2.,0,0,0))
        np.testing.assert_array_almost_equal(f.forces[1].virial, (0,0,0,-6.,-2.,0))
        np.testing.assert_array_almost_equal(f.forces[2].virial, (0,0,0,0,0,24.))

        # change the spring constant
        f.set_params(k=1.0)
        hoomd.run(1)
        np.testing.assert_array_almost_equal(f.forces[0].force, (1.,0,0))
        np.testing.assert_array_almost_equal(f.forces[1].force, (0,-1.,0))
        np.testing.assert_array_almost_equal(f.forces[2].force, (0,0,4.))
        self.assertAlmostEqual(f.forces[0].energy, 0.5)
        self.assertAlmostEqual(f.forces[1].energy, 0.5)
        self.assertAlmostEqual(f.forces[2].energy, 8.0)

        # change the radius
        f.set_params(k=2.0, radius=2)
        hoomd.run(1)
        np.testing.assert_array_almost_equal(f.forces[0].force, (0,0,0))
        np.testing.assert_array_almost_equal(f.forces[1].force, (0,0,0))
        np.testing.assert_array_almost_equal(f.forces[2].force, (0,0,6.))
        self.assertAlmostEqual(f.forces[0].energy, 0.)
        self.assertAlmostEqual(f.forces[1].energy, 0.)
        self.assertAlmostEqual(f.forces[2].energy, 9.)

    def test_group(self):
        """Test forces on subgroup of prticles."""
        # leave out particle 0
        group = hoomd.group.tags(1,2)

        # compute forces
        f = azplugins.restrain.sphere(group=group, radius=1, origin=(1,1,1), k=2.0)
        hoomd.run(1)
        np.testing.assert_array_almost_equal(f.forces[0].force, (0,0,0))
        np.testing.assert_array_almost_equal(f.forces[1].force, (0,-2.,0))
        np.testing.assert_array_almost_equal(f.forces[2].force, (0,0,8.))
        self.assertAlmostEqual(f.forces[0].energy, 0.)
        self.assertAlmostEqual(f.forces[1].energy, 1.)
        self.assertAlmostEqual(f.forces[2].energy, 16.)
        np.testing.assert_array_almost_equal(f.forces[0].virial, (0,0,0,0,0,0))
        np.testing.assert_array_almost_equal(f.forces[1].virial, (0,0,0,-6.,-2.,0))
        np.testing.assert_array_almost_equal(f.forces[2].virial, (0,0,0,0,0,24.))

    def tearDown(self):
        hoomd.context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
