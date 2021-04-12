# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021, Auburn University
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

# azplugins.restrain.plane
class restrain_plane_tests(unittest.TestCase):
    """Tests azplugins.restrain.plane"""

    def setUp(self):
        snap = hoomd.data.make_snapshot(N=3, box=hoomd.data.boxdim(L=5.0))
        if hoomd.comm.get_rank() == 0:
            snap.particles.position[0] = [ 1,0,0]
            snap.particles.position[1] = [-1,0,2]
            snap.particles.position[2] = [ 2,0,0]
            snap.particles.image[2] = [-1,0,0]
        hoomd.init.read_snapshot(snap)

        # dummy integrator
        all_ = hoomd.group.all()
        md.integrate.mode_standard(dt=0)
        md.integrate.nve(group=all_)

    def test_create(self):
        """Test object creation and updating."""
        f = azplugins.restrain.plane(group=hoomd.group.all(), point=(0,0,0), normal=(1,0,0), k=2.0)

        f.set_params(k=5.0)
        f.set_params(k=8)

        f.set_params(point=(0,0,1))
        f.set_params(point=[0,0,1])
        f.set_params(point=np.array([0,0,1]))

        f.set_params(normal=(0,0,1))
        f.set_params(normal=[0,0,1])
        f.set_params(normal=np.array([0,0,1]))

        f.set_params(point=(0,0,0), normal=(1,0,0), k=10.0)

    def test_force(self):
        """Test forces computed on particles."""
        group = hoomd.group.all()

        # compute forces
        f = azplugins.restrain.plane(group=group, point=(0,0,0), normal=(1,0,0), k=2.0)
        hoomd.run(1)
        np.testing.assert_array_almost_equal(f.forces[0].force, (-2.,0,0))
        np.testing.assert_array_almost_equal(f.forces[1].force, ( 2.,0,0))
        np.testing.assert_array_almost_equal(f.forces[2].force, ( 6.,0,0))
        self.assertAlmostEqual(f.forces[0].energy, 1.)
        self.assertAlmostEqual(f.forces[1].energy, 1.)
        self.assertAlmostEqual(f.forces[2].energy, 9.)
        np.testing.assert_array_almost_equal(f.forces[0].virial, (-2.,0,0,0,0,0))
        np.testing.assert_array_almost_equal(f.forces[1].virial, (-2.,0,4.,0,0,0))
        np.testing.assert_array_almost_equal(f.forces[2].virial, (12.,0,0,0,0,0))

        # change the spring constant
        f.set_params(k=1.0)
        hoomd.run(1)
        np.testing.assert_array_almost_equal(f.forces[0].force, (-1.,0,0))
        np.testing.assert_array_almost_equal(f.forces[1].force, ( 1.,0,0))
        np.testing.assert_array_almost_equal(f.forces[2].force, ( 3.,0,0))
        self.assertAlmostEqual(f.forces[0].energy, 0.5)
        self.assertAlmostEqual(f.forces[1].energy, 0.5)
        self.assertAlmostEqual(f.forces[2].energy, 4.5)

        # shift the plane down
        f.set_params(point=(-1,0,0))
        hoomd.run(1)
        np.testing.assert_array_almost_equal(f.forces[0].force, (-2.,0,0))
        np.testing.assert_array_almost_equal(f.forces[1].force, ( 0.,0,0))
        np.testing.assert_array_almost_equal(f.forces[2].force, ( 2.,0,0))
        self.assertAlmostEqual(f.forces[0].energy, 2.0)
        self.assertAlmostEqual(f.forces[1].energy, 0.0)
        self.assertAlmostEqual(f.forces[2].energy, 2.0)

        # rotate the plane so that only particle 1 is off the line
        f.set_params(point=(0,0,0), normal=(0,0,1))
        hoomd.run(1)
        np.testing.assert_array_almost_equal(f.forces[0].force, (0,0,0))
        np.testing.assert_array_almost_equal(f.forces[1].force, (0,0,-2))
        np.testing.assert_array_almost_equal(f.forces[2].force, (0,0,0))
        self.assertAlmostEqual(f.forces[0].energy, 0.0)
        self.assertAlmostEqual(f.forces[1].energy, 2.0)
        self.assertAlmostEqual(f.forces[2].energy, 0.0)

    def test_group(self):
        """Test forces on subgroup of prticles."""
        # leave out particle 0
        group = hoomd.group.tags(1,2)

        # compute forces
        f = azplugins.restrain.plane(group=group, point=(0,0,0), normal=(1,0,0), k=2.0)
        hoomd.run(1)
        np.testing.assert_array_almost_equal(f.forces[0].force, ( 0.,0,0))
        np.testing.assert_array_almost_equal(f.forces[1].force, ( 2.,0,0))
        np.testing.assert_array_almost_equal(f.forces[2].force, ( 6.,0,0))
        self.assertAlmostEqual(f.forces[0].energy, 0.)
        self.assertAlmostEqual(f.forces[1].energy, 1.)
        self.assertAlmostEqual(f.forces[2].energy, 9.)
        np.testing.assert_array_almost_equal(f.forces[0].virial, (0,0,0,0,0,0))
        np.testing.assert_array_almost_equal(f.forces[1].virial, (-2.,0,4.,0,0,0))
        np.testing.assert_array_almost_equal(f.forces[2].virial, (12.,0,0,0,0,0))

    def tearDown(self):
        hoomd.context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
