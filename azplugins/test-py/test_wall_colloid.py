# Copyright (c) 2016, Panagiotopoulos Group, Princeton University
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: mphoward

from hoomd import *
from hoomd import md
context.initialize()
try:
    from hoomd import azplugins
except ImportError:
    import azplugins
import unittest

# azplugins.wall.colloid
class wall_colloid_tests(unittest.TestCase):
    def setUp(self):
        snap = data.make_snapshot(N=100, box=data.boxdim(L=20), particle_types=['A'])
        self.s = init.read_snapshot(snap)
        self.walls = md.wall.group()
        self.walls.add_plane((0,0,-5),(0,0,1))

    # basic test of creation
    def test_basic(self):
        coll = azplugins.wall.colloid(self.walls)
        coll.force_coeff.set('A', epsilon=1.0, sigma=1.0, r_cut=2.5, r_extrap=0.1)
        coll.update_coeffs()

    # test missing epsilon
    def test_set_missing_epsilon(self):
        coll = azplugins.wall.colloid(self.walls)
        coll.force_coeff.set('A', sigma=1.0)
        self.assertRaises(RuntimeError, coll.update_coeffs)

    # test missing sigma
    def test_set_missing_sigma(self):
        coll = azplugins.wall.colloid(self.walls)
        coll.force_coeff.set('A', epsilon=1.0)
        self.assertRaises(RuntimeError, coll.update_coeffs)

    # test missing coefficients
    def test_missing_A(self):
        coll = azplugins.wall.colloid(self.walls)
        self.assertRaises(RuntimeError, coll.update_coeffs)

    # test default coefficients
    def test_default_coeff(self):
        coll = azplugins.wall.colloid(self.walls)
        # (r_cut, and r_extrap are default)
        coll.force_coeff.set('A', epsilon=1.0, sigma=1.0)
        coll.update_coeffs()

    # test coeff list
    def test_coeff_list(self):
        coll = azplugins.wall.colloid(self.walls)
        coll.force_coeff.set(['A', 'B'], epsilon=1.0, sigma=1.0, r_cut=2.5, r_extrap=0.1)
        coll.update_coeffs()

    # test adding types
    def test_type_add(self):
        coll = azplugins.wall.colloid(self.walls)
        coll.force_coeff.set('A', epsilon=1.0, sigma=1.0)
        self.s.particles.types.add('B')
        self.assertRaises(RuntimeError, coll.update_coeffs)
        coll.force_coeff.set('B', epsilon=1.0, sigma=1.0)
        coll.update_coeffs()

    def tearDown(self):
        del self.s, self.walls
        context.initialize()

# test the validity of the pair potential
class potential_wall_colloid_tests(unittest.TestCase):
    def setUp(self):
        snap = data.make_snapshot(N=1, box=data.boxdim(L=20),particle_types=['A'])
        if comm.get_rank() == 0:
            snap.particles.position[0] = (1,1,-2)
            snap.particles.diameter[0] = 1.5
        init.read_snapshot(snap)

        # planar wall
        self.walls = md.wall.group()
        self.walls.add_plane((0,0,-5),(0,0,1))

        # integrator with zero timestep to compute forces
        md.integrate.mode_standard(dt=0)
        md.integrate.nve(group = group.all())

    # test the calculation of force and potential
    def test_potential(self):
        coll = azplugins.wall.colloid(self.walls)

        # by default, cutoff is 0 so there should be no interaction
        coll.force_coeff.set('A', epsilon=100.0, sigma=1.05)
        run(1)
        self.assertAlmostEqual(coll.forces[0].energy, 0)
        f0 = coll.forces[0].force
        self.assertAlmostEqual(f0[0], 0)
        self.assertAlmostEqual(f0[1], 0)
        self.assertAlmostEqual(f0[2], 0)

        # set the cutoff and evaluate the energy and force
        # need to shift the energy since this is the default mode
        coll.force_coeff.set('A', r_cut=6.0)
        run(1)
        U = -0.374977848076 - (-0.0442302367107)
        F = -0.394551653468
        self.assertAlmostEqual(coll.forces[0].energy, U, 5)
        self.assertAlmostEqual(coll.forces[0].force[2], F, 5)

        # check epsilon = 0 is zero
        coll.force_coeff.set('A', epsilon=0.0)
        run(1)
        self.assertAlmostEqual(coll.forces[0].energy, 0)
        self.assertAlmostEqual(coll.forces[0].force[2], 0)

        # check outside the cutoff is zero
        coll.force_coeff.set('A', epsilon=2.0, r_cut=2.0)
        run(1)
        self.assertAlmostEqual(coll.forces[0].energy, 0)
        self.assertAlmostEqual(coll.forces[0].force[2], 0)

    def tearDown(self):
        del self.walls
        context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
