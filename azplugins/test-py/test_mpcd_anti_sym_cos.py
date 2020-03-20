# Copyright (c) 2018-2020, Michael P. Howard
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: astatt

import unittest
import numpy as np
import hoomd
from hoomd import md
from hoomd import mpcd

try:
    from hoomd import azplugins
    import hoomd.azplugins.mpcd
except ImportError:
    import azplugins
    import azplugins.mpcd
import unittest

# unit tests for mpcd symmetric cosine channel  geometry
class mpcd_anti_sym_cos_test(unittest.TestCase):
    def setUp(self):

        hoomd.context.initialize()

        # set the decomposition in z for mpi builds
        if hoomd.comm.get_num_ranks() > 1:
            hoomd.comm.decomposition(nz=2)

        # default testing configuration
        hoomd.init.read_snapshot(hoomd.data.make_snapshot(N=0, box=hoomd.data.boxdim(L=20.)))

        # initialize the system from the starting snapshot
        # test vertical, diagonal, and horizontal collisions to wall
        snap = mpcd.data.make_snapshot(N=3)
        snap.particles.position[:] = [[0.,-3.0,5.85],[1.55,0.,5.5],[0.0,0.0,2.2]]
        snap.particles.velocity[:] = [[0,0.,1.],[1.,0.,0.],[-1.,-1.,-1.]]
        self.s = mpcd.init.read_snapshot(snap)

        mpcd.integrator(dt=0.1)

    # test creation can happen (with all parameters set)
    def test_create(self):
        azplugins.mpcd.anti_sym_cos(A=4., h=2. ,p=1, V=0.1, boundary="no_slip")

    # test for setting parameters
    def test_set_params(self):
      anti_sym_cos = azplugins.mpcd.anti_sym_cos(A=4.,h=2. ,p=1)
      self.assertAlmostEqual(anti_sym_cos.A, 4.)
      self.assertAlmostEqual(anti_sym_cos.V, 0.)
      self.assertEqual(anti_sym_cos.boundary, "no_slip")
      self.assertAlmostEqual(anti_sym_cos._cpp.geometry.getAmplitude(), 4.)
      self.assertAlmostEqual(anti_sym_cos._cpp.geometry.getVelocity(), 0.)
      self.assertEqual(anti_sym_cos._cpp.geometry.getBoundaryCondition(), mpcd._mpcd.boundary.no_slip)

      # change H and also ensure other parameters stay the same
      anti_sym_cos.set_params(A=2.)
      self.assertAlmostEqual(anti_sym_cos.A, 2.)
      self.assertAlmostEqual(anti_sym_cos.V, 0.)
      self.assertEqual(anti_sym_cos.boundary, "no_slip")
      self.assertAlmostEqual(anti_sym_cos._cpp.geometry.getAmplitude(), 2.)
      self.assertAlmostEqual(anti_sym_cos._cpp.geometry.getVelocity(), 0.)
      self.assertEqual(anti_sym_cos._cpp.geometry.getBoundaryCondition(), mpcd._mpcd.boundary.no_slip)

      # change V
      anti_sym_cos.set_params(V=0.1)
      self.assertAlmostEqual(anti_sym_cos.V, 0.1)
      self.assertAlmostEqual(anti_sym_cos._cpp.geometry.getVelocity(), 0.1)

      # change BCs
      anti_sym_cos.set_params(boundary="slip")
      self.assertEqual(anti_sym_cos.boundary, "slip")
      self.assertEqual(anti_sym_cos._cpp.geometry.getBoundaryCondition(), mpcd._mpcd.boundary.slip)

    # test for invalid boundary conditions being set
    def test_bad_boundary(self):
      anti_sym_cos = azplugins.mpcd.anti_sym_cos(A=4.,h=2. ,p=1)
      anti_sym_cos.set_params(boundary="no_slip")
      anti_sym_cos.set_params(boundary="slip")

      with self.assertRaises(ValueError):
          anti_sym_cos.set_params(boundary="invalid")

    # test that setting the cosine size too large raises an error
    def test_validate_box(self):
      # initial configuration is invalid
      anti_sym_cos = azplugins.mpcd.anti_sym_cos(A=10.,h=2. ,p=1)
      with self.assertRaises(RuntimeError):
          hoomd.run(1)

      # now it should be valid
      anti_sym_cos.set_params(A=4.,h=2. ,p=1)
      hoomd.run(2)

      # make sure we can invalidate it again
      anti_sym_cos.set_params(A=10.,h=2. ,p=1)
      with self.assertRaises(RuntimeError):
          hoomd.run(1)

    # test that particles out of bounds can be caught
    def test_out_of_bounds(self):
      anti_sym_cos = azplugins.mpcd.anti_sym_cos(A=2., h=1. ,p=1)
      with self.assertRaises(RuntimeError):
          hoomd.run(1)

      anti_sym_cos.set_params(A=5.,h=2. ,p=1)
      hoomd.run(1)

    # test basic stepping behavior with no slip boundary conditions
    def test_step_noslip(self):
      azplugins.mpcd.anti_sym_cos(A=4.,h=2. ,p=1, boundary='no_slip')

      # take one step, particle 1 hits the wall
      hoomd.run(1)
      snap = self.s.take_snapshot()
      if hoomd.comm.get_rank() == 0:
          np.testing.assert_array_almost_equal(snap.particles.position[0], [0,-3.0,5.95])
          np.testing.assert_array_almost_equal(snap.particles.position[1], [1.567225,0.0,5.5])
          np.testing.assert_array_almost_equal(snap.particles.position[2], [-0.1,-0.1,2.1])
          np.testing.assert_array_almost_equal(snap.particles.velocity[0], [0,0,1.])
          np.testing.assert_array_almost_equal(snap.particles.velocity[1], [-1,0,0])
          np.testing.assert_array_almost_equal(snap.particles.velocity[2], [-1,-1,-1])

      # particle 0 hits the highest spot and is reflected back
      hoomd.run(1)
      snap = self.s.take_snapshot()
      if hoomd.comm.get_rank() == 0:
          np.testing.assert_array_almost_equal(snap.particles.position[0], [0,-3.0,5.95])
          np.testing.assert_array_almost_equal(snap.particles.position[1], [1.467225,0.0,5.5])
          np.testing.assert_array_almost_equal(snap.particles.position[2], [-0.2,-0.2,2.0])
          np.testing.assert_array_almost_equal(snap.particles.velocity[0], [0,0,-1.])
          np.testing.assert_array_almost_equal(snap.particles.velocity[1], [-1,0,0])
          np.testing.assert_array_almost_equal(snap.particles.velocity[2], [-1,-1,-1])

      # particle 2 collides diagonally
      hoomd.run(1)
      snap = self.s.take_snapshot()
      if hoomd.comm.get_rank() == 0:
        np.testing.assert_array_almost_equal(snap.particles.position[0], [0,-3.0,5.85])
        np.testing.assert_array_almost_equal(snap.particles.position[1], [1.367225,0.0,5.5])
        np.testing.assert_array_almost_equal(snap.particles.position[2], [-0.11717,-0.11717,2.08283])
        np.testing.assert_array_almost_equal(snap.particles.velocity[0], [0,0,-1.])
        np.testing.assert_array_almost_equal(snap.particles.velocity[1], [-1,0,0])
        np.testing.assert_array_almost_equal(snap.particles.velocity[2], [1,1,1])
    #same as test above except for slip -> velcities differ
    def test_step_slip(self):
        azplugins.mpcd.anti_sym_cos(A=4.,h=2. ,p=1, boundary="slip")

        # take one step,  particle 1 hits the wall
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.position[0], [0,-3.0,5.95])
            np.testing.assert_array_almost_equal(snap.particles.position[1], [1.62764,0,5.463246])
            np.testing.assert_array_almost_equal(snap.particles.position[2], [-0.1,-0.1,2.1])
            np.testing.assert_array_almost_equal(snap.particles.velocity[0], [0,0,1.])
            np.testing.assert_array_almost_equal(snap.particles.velocity[1], [0.459737,0,-0.888055])
            np.testing.assert_array_almost_equal(snap.particles.velocity[2], [-1,-1,-1])

        # take one step,  particle 0 hits the wall (same as for no_slip, because it's vertical)
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.position[0], [0,-3.0,5.95])
            np.testing.assert_array_almost_equal(snap.particles.position[2], [-0.2,-0.2,2.0])
            np.testing.assert_array_almost_equal(snap.particles.velocity[0], [0,0,-1.])
            np.testing.assert_array_almost_equal(snap.particles.velocity[2], [-1,-1,-1])

        # take another step,  particle 2 hits the wall
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.position[2], [-0.313714,-0.3,2.066657])
            np.testing.assert_array_almost_equal(snap.particles.velocity[2], [-1.150016, -1.,0.823081])

    # test that virtual particle filler can be attached, removed, and updated
    def test_filler(self):
      # initialization of a filler
      anti_sym_cos = azplugins.mpcd.anti_sym_cos(A=4.,h=2. ,p=1)
      anti_sym_cos.set_filler(density=5., kT=1.0, seed=42, type='A')
      self.assertTrue(anti_sym_cos._filler is not None)

      # run should be able to setup the filler, although this all happens silently
      hoomd.run(1)

      # changing the geometry should still be OK with a run
      anti_sym_cos.set_params(V=1.0)
      hoomd.run(1)

      # changing filler should be allowed
      anti_sym_cos.set_filler(density=10., kT=1.5, seed=7)
      self.assertTrue(anti_sym_cos._filler is not None)
      hoomd.run(1)

      # assert an error is raised if we set a bad particle type
      with self.assertRaises(RuntimeError):
          anti_sym_cos.set_filler(density=5., kT=1.0, seed=42, type='B')

      # assert an error is raised if we set a bad density
      with self.assertRaises(RuntimeError):
          anti_sym_cos.set_filler(density=-1.0, kT=1.0, seed=42)

      # removing the filler should still allow a run
      anti_sym_cos.remove_filler()
      self.assertTrue(anti_sym_cos._filler is None)
      hoomd.run(1)


    def tearDown(self):
        del self.s

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
