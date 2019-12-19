# Copyright (c) 2018-2019, Michael P. Howard
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: mphoward

import numpy as np

import hoomd
hoomd.context.initialize()
from hoomd import md
try:
    from hoomd import azplugins
except ImportError:
    import azplugins
import unittest

# azplugins.restrain.cylinder
class restrain_cylinder_tests(unittest.TestCase):
    """Tests azplugins.restrain.cylinder"""

    def setUp(self):
        snap = hoomd.data.make_snapshot(N=3, box=hoomd.data.boxdim(L=7.0))
        if hoomd.comm.get_rank() == 0:
            snap.particles.position[0] = [-1,0,0]
            snap.particles.position[1] = [0,0,1]
            snap.particles.position[2] = [2,0,3]
            snap.particles.image[2] = [-1,0,-1]
        hoomd.init.read_snapshot(snap)

        # dummy integrator
        all_ = hoomd.group.all()
        md.integrate.mode_standard(dt=0)
        md.integrate.nve(group=all_)

    def test_create(self):
        """Test object creation and updating."""
        f = azplugins.restrain.cylinder(group=hoomd.group.all(), radius=2, origin=(0,0,0), axis=(1,0,0), k=2.0)

        f.set_params(k=5.0)
        f.set_params(k=8)

        f.set_params(radius=3.)
        f.set_params(radius=4)

        f.set_params(origin=(0,0,1))
        f.set_params(origin=[0,0,1])
        f.set_params(origin=np.array([0,0,1]))

        f.set_params(axis=(0,0,1))
        f.set_params(axis=[0,0,1])
        f.set_params(axis=np.array([0,0,1]))

        f.set_params(radius=2., origin=(0,0,0), axis=(1,0,0), k=10.0)

    def test_force(self):
        """Test forces computed on particles."""
        group = hoomd.group.all()

        # compute forces
        f = azplugins.restrain.cylinder(group=group, radius=1, origin=(1,0,2), axis=(1,0,0), k=2.0)
        hoomd.run(1)
        np.testing.assert_array_almost_equal(f.forces[0].force, (0,0,2.), decimal=3)
        np.testing.assert_array_almost_equal(f.forces[1].force, (0,0,0), decimal=3)
        np.testing.assert_array_almost_equal(f.forces[2].force, (0,0,10.), decimal=3)
        self.assertAlmostEqual(f.forces[0].energy, 1.)
        self.assertAlmostEqual(f.forces[1].energy, 0.)
        self.assertAlmostEqual(f.forces[2].energy, 25.)
        np.testing.assert_array_almost_equal(f.forces[0].virial, (0,0,0,0,0,0), decimal=3)
        np.testing.assert_array_almost_equal(f.forces[1].virial, (0,0,0,0,0,0), decimal=3)
        np.testing.assert_array_almost_equal(f.forces[2].virial, (0,0,0,0,0,30.), decimal=3)

        # change the spring constant
        f.set_params(k=1.0)
        hoomd.run(1)
        np.testing.assert_array_almost_equal(f.forces[0].force, (0,0,1.), decimal=3)
        np.testing.assert_array_almost_equal(f.forces[1].force, (0,0,0), decimal=3)
        np.testing.assert_array_almost_equal(f.forces[2].force, (0,0,5.), decimal=3)
        self.assertAlmostEqual(f.forces[0].energy, 0.5)
        self.assertAlmostEqual(f.forces[1].energy, 0.)
        self.assertAlmostEqual(f.forces[2].energy, 12.5)

        # change the radius
        f.set_params(k=2.0, radius=2)
        hoomd.run(1)
        np.testing.assert_array_almost_equal(f.forces[0].force, (0,0,0), decimal=3)
        np.testing.assert_array_almost_equal(f.forces[1].force, (0,0,-2.), decimal=3)
        np.testing.assert_array_almost_equal(f.forces[2].force, (0,0,8.), decimal=3)
        self.assertAlmostEqual(f.forces[0].energy, 0.)
        self.assertAlmostEqual(f.forces[1].energy, 1.)
        self.assertAlmostEqual(f.forces[2].energy, 16.)
        np.testing.assert_array_almost_equal(f.forces[0].virial, (0,0,0,0,0,0), decimal=3)
        np.testing.assert_array_almost_equal(f.forces[1].virial, (0,0,0,0,0,-2.), decimal=3)
        np.testing.assert_array_almost_equal(f.forces[2].virial, (0,0,0,0,0,24.), decimal=3)

        # shift the cylinder
        f.set_params(radius=1, origin=(0,0,1))
        hoomd.run(1)
        np.testing.assert_array_almost_equal(f.forces[0].force, (0,0,0), decimal=3)
        np.testing.assert_array_almost_equal(f.forces[1].force, (0,0,0), decimal=3)
        np.testing.assert_array_almost_equal(f.forces[2].force, (0,0,8.), decimal=3)
        self.assertAlmostEqual(f.forces[0].energy, 0)
        self.assertAlmostEqual(f.forces[1].energy, 0.)
        self.assertAlmostEqual(f.forces[2].energy, 16.)

        # rotate the axis
        f.set_params(radius=1, origin=(1,0,2), axis=(0,0,1))
        hoomd.run(1)
        np.testing.assert_array_almost_equal(f.forces[0].force, (2.,0,0), decimal=3)
        np.testing.assert_array_almost_equal(f.forces[1].force, (0,0,0), decimal=3)
        np.testing.assert_array_almost_equal(f.forces[2].force, (10.,0,0), decimal=3)
        self.assertAlmostEqual(f.forces[0].energy, 1.)
        self.assertAlmostEqual(f.forces[1].energy, 0.)
        self.assertAlmostEqual(f.forces[2].energy, 25.)
        np.testing.assert_array_almost_equal(f.forces[0].virial, (-2.,0,0,0,0,0), decimal=3)
        np.testing.assert_array_almost_equal(f.forces[1].virial, (0,0,0,0,0,0), decimal=3)
        np.testing.assert_array_almost_equal(f.forces[2].virial, (20.,0,30.,0,0,0), decimal=3)

    def test_group(self):
        """Test forces on subgroup of prticles."""
        # leave out particle 0
        group = hoomd.group.tags(1,2)

        # compute forces
        f = azplugins.restrain.cylinder(group=group, radius=1, origin=(1,0,2), axis=(1,0,0), k=2.0)
        hoomd.run(1)
        np.testing.assert_array_almost_equal(f.forces[0].force, (0,0,0), decimal=3)
        np.testing.assert_array_almost_equal(f.forces[1].force, (0,0,0), decimal=3)
        np.testing.assert_array_almost_equal(f.forces[2].force, (0,0,10.), decimal=3)
        self.assertAlmostEqual(f.forces[0].energy, 0.)
        self.assertAlmostEqual(f.forces[1].energy, 0.)
        self.assertAlmostEqual(f.forces[2].energy, 25.)
        np.testing.assert_array_almost_equal(f.forces[0].virial, (0,0,0,0,0,0), decimal=3)
        np.testing.assert_array_almost_equal(f.forces[1].virial, (0,0,0,0,0,0), decimal=3)
        np.testing.assert_array_almost_equal(f.forces[2].virial, (0,0,0,0,0,30), decimal=3)

    def tearDown(self):
        hoomd.context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
