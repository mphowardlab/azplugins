# -*- coding: iso-8859-1 -*-
# Maintainer: mphoward

from hoomd import *
from hoomd import md
from hoomd import azplugins
context.initialize()
import unittest
import os

# azplugins.wall.lj93
class wall_lj93_tests(unittest.TestCase):
    def setUp(self):
        snap = data.make_snapshot(N=100, box=data.boxdim(L=20), particle_types=['A'])
        self.s = init.read_snapshot(snap)
        self.walls = md.wall.group()
        self.walls.add_plane((0,0,-5),(0,0,1))

    # basic test of creation
    def test_basic(self):
        lj93 = azplugins.wall.lj93(self.walls)
        lj93.force_coeff.set('A', epsilon=1.0, sigma=1.0, r_cut=2.5, r_extrap=0.1)
        lj93.update_coeffs()

    # test missing epsilon
    def test_set_missing_epsilon(self):
        lj93 = azplugins.wall.lj93(self.walls)
        lj93.force_coeff.set('A', sigma=1.0)
        self.assertRaises(RuntimeError, lj93.update_coeffs)

    # test missing sigma
    def test_set_missing_sigma(self):
        lj93 = azplugins.wall.lj93(self.walls)
        lj93.force_coeff.set('A', epsilon=1.0)
        self.assertRaises(RuntimeError, lj93.update_coeffs)

    # test missing coefficients
    def test_missing_A(self):
        lj93 = azplugins.wall.lj93(self.walls)
        self.assertRaises(RuntimeError, lj93.update_coeffs)

    # test default coefficients
    def test_default_coeff(self):
        lj93 = azplugins.wall.lj93(self.walls)
        # (r_cut, and r_extrap are default)
        lj93.force_coeff.set('A', epsilon=1.0, sigma=1.0)
        lj93.update_coeffs()

    # test coeff list
    def test_coeff_list(self):
        lj93 = azplugins.wall.lj93(self.walls)
        lj93.force_coeff.set(['A', 'B'], epsilon=1.0, sigma=1.0, r_cut=2.5, r_extrap=0.1)
        lj93.update_coeffs()

    # test adding types
    def test_type_add(self):
        lj93 = azplugins.wall.lj93(self.walls)
        lj93.force_coeff.set('A', epsilon=1.0, sigma=1.0)
        self.s.particles.types.add('B')
        self.assertRaises(RuntimeError, lj93.update_coeffs)
        lj93.force_coeff.set('B', epsilon=1.0, sigma=1.0)
        lj93.update_coeffs()

    def tearDown(self):
        del self.s, self.walls
        context.initialize()

# test the validity of the pair potential
class potential_wall_lj93_tests(unittest.TestCase):
    def setUp(self):
        snap = data.make_snapshot(N=1, box=data.boxdim(L=20),particle_types=['A'])
        if comm.get_rank() == 0:
            snap.particles.position[0] = (1,1,-3.95)
        init.read_snapshot(snap)

        # planar wall
        self.walls = md.wall.group()
        self.walls.add_plane((0,0,-5),(0,0,1))

        # integrator with zero timestep to compute forces
        md.integrate.mode_standard(dt=0)
        md.integrate.nve(group = group.all())

    # test the calculation of force and potential
    def test_potential(self):
        lj93 = azplugins.wall.lj93(self.walls)

        # by default, cutoff is 0 so there should be no interaction
        lj93.force_coeff.set('A', epsilon=2.0, sigma=1.05)
        run(1)
        self.assertAlmostEqual(lj93.forces[0].energy, 0)
        f0 = lj93.forces[0].force
        self.assertAlmostEqual(f0[0], 0)
        self.assertAlmostEqual(f0[1], 0)
        self.assertAlmostEqual(f0[2], 0)

        # set the cutoff and evaluate the energy and force
        # need to shift the energy since this is the default mode
        lj93.force_coeff.set('A', r_cut=3.0)
        run(1)
        U = -1.7333333333333334 - (-0.08572898249635419)
        F = -3.4285714285714284
        self.assertAlmostEqual(lj93.forces[0].energy, U, 5)
        self.assertAlmostEqual(lj93.forces[0].force[2], F, 5)

        # check epsilon = 0 is zero
        lj93.force_coeff.set('A', epsilon=0.0)
        run(1)
        self.assertAlmostEqual(lj93.forces[0].energy, 0)
        self.assertAlmostEqual(lj93.forces[0].force[2], 0)

        # check outside the cutoff is zero
        lj93.force_coeff.set('A', epsilon=2.0, r_cut=0.8)
        run(1)
        self.assertAlmostEqual(lj93.forces[0].energy, 0)
        self.assertAlmostEqual(lj93.forces[0].force[2], 0)

    def tearDown(self):
        del self.walls
        context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
