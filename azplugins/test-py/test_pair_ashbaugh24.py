# Copyright (c) 2018-2019, Michael P. Howard
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: astatt

from hoomd import *
from hoomd import md
context.initialize()
try:
    from hoomd import azplugins
except ImportError:
    import azplugins
import unittest

# azplugins.pair.lj124
class pair_ash24_tests(unittest.TestCase):
    def setUp(self):
        # raw snapshot is fine, just needs to have the types
        snap = data.make_snapshot(N=100, box=data.boxdim(L=20), particle_types=['A'])
        self.s = init.read_snapshot(snap)
        self.nl = md.nlist.cell()
        context.current.sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        ash24 = azplugins.pair.ashbaugh24(r_cut=3.0, nlist = self.nl)
        ash24.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, lam=1.0, r_cut=1.5, r_on=1.2)
        ash24.update_coeffs()

    # test missing coefficients
    def test_set_missing_epsilon(self):
        ash24 = azplugins.pair.ashbaugh24(r_cut=1.5, nlist = self.nl)
        ash24.pair_coeff.set('A', 'A', sigma=1.0)
        self.assertRaises(RuntimeError, ash24.update_coeffs)

    # test missing coefficients
    def test_set_missing_sigma(self):
        ash24 = azplugins.pair.ashbaugh24(r_cut=1.5, nlist = self.nl)
        ash24.pair_coeff.set('A', 'A', epsilon=1.0)
        self.assertRaises(RuntimeError, ash24.update_coeffs)

    # test missing coefficients
    def test_missing_AA(self):
        ash24 = azplugins.pair.ashbaugh24(r_cut=1.5, nlist = self.nl)
        self.assertRaises(RuntimeError, ash24.update_coeffs)

    # test set params
    def test_set_params(self):
        ash24 = azplugins.pair.ashbaugh24(r_cut=1.5, nlist = self.nl)
        ash24.set_params(mode="no_shift")
        ash24.set_params(mode="shift")
        ash24.set_params(mode="xplor")
        self.assertRaises(RuntimeError, ash24.set_params, mode="unknown")

    # test max rcut
    def test_max_rcut(self):
        ash24 = azplugins.pair.ashbaugh24(r_cut=2.5, nlist = self.nl)
        ash24.pair_coeff.set('A', 'A', sigma=1.0, epsilon=1.0, lam=1.0)
        self.assertAlmostEqual(2.5, ash24.get_max_rcut())
        ash24.pair_coeff.set('A', 'A', r_cut = 3.0)
        self.assertAlmostEqual(3.0, ash24.get_max_rcut())

    # test specific nlist subscription
    def test_nlist_subscribe(self):
        ash24 = azplugins.pair.ashbaugh24(r_cut=1.5, nlist = self.nl)

        ash24.pair_coeff.set('A', 'A', sigma=1.0, epsilon=1.0, lam=1.0)
        self.nl.update_rcut()
        self.assertAlmostEqual(1.5, self.nl.r_cut.get_pair('A','A'))

        ash24.pair_coeff.set('A', 'A', r_cut=2.0)
        self.nl.update_rcut()
        self.assertAlmostEqual(2.0, self.nl.r_cut.get_pair('A','A'))

    # test coeff list
    def test_coeff_list(self):
        ash24 = azplugins.pair.ashbaugh24(r_cut=1.5, nlist = self.nl)
        ash24.pair_coeff.set(['A', 'B'], ['A', 'C'], epsilon=1.0, sigma=1.0, lam=1.0, r_cut=1.5, r_on=1.2)
        ash24.update_coeffs()

    # test adding types
    def test_type_add(self):
        ash24 = azplugins.pair.ashbaugh24(r_cut=1.5, nlist = self.nl)
        ash24.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, lam=1.0)
        self.s.particles.types.add('B')
        self.assertRaises(RuntimeError, ash24.update_coeffs)
        ash24.pair_coeff.set('A', 'B', epsilon=1.0, sigma=1.0, lam=1.0)
        ash24.pair_coeff.set('B', 'B', epsilon=1.0, sigma=1.0, lam=1.0)
        ash24.update_coeffs()

    def tearDown(self):
        del self.s, self.nl
        context.initialize()

# test the validity of the pair potential
class potential_ash2424_tests(unittest.TestCase):
    def setUp(self):
        snap = data.make_snapshot(N=2, box=data.boxdim(L=20),particle_types=['A'])
        if comm.get_rank() == 0:
            snap.particles.position[0] = (0,0,0)
            snap.particles.position[1] = (1.0,0,0)
        init.read_snapshot(snap)
        self.nl = md.nlist.cell()

    # test the calculation of force and potential
    def test_potential(self):
        ash24 = azplugins.pair.ashbaugh24(r_cut=1.5, nlist = self.nl)

        # test when it's in the wca part (below 2^(1/24)*sigma=1.0293*sigma, no potential shifting
        ash24.pair_coeff.set('A','A', epsilon=1.0, sigma=1.0, lam=0.0)
        ash24.set_params(mode="no_shift")

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = group.all())
        run(1)
        U = 1.0
        F = -96.0
        f0 = ash24.forces[0].force
        f1 = ash24.forces[1].force
        e0 = ash24.forces[0].energy
        e1 = ash24.forces[1].energy

        self.assertAlmostEqual(e0,0.5*U,3)
        self.assertAlmostEqual(e1,0.5*U,3)

        self.assertAlmostEqual(f0[0],F,3)
        self.assertAlmostEqual(f0[1],0)
        self.assertAlmostEqual(f0[2],0)

        self.assertAlmostEqual(f1[0],-F,3)
        self.assertAlmostEqual(f1[1],0)
        self.assertAlmostEqual(f1[2],0)

        # change lambda to check for shifting of energy (force stays the same)
        ash24.pair_coeff.set('A','A', lam=0.5)
        run(1)
        U = 0.5
        F =  -96.0
        self.assertAlmostEqual(ash24.forces[0].energy, 0.5*U, 3)
        self.assertAlmostEqual(ash24.forces[1].energy, 0.5*U, 3)
        self.assertAlmostEqual(ash24.forces[0].force[0], F, 3)
        self.assertAlmostEqual(ash24.forces[1].force[0], -F, 3)

        # change sigma so that now the particle is in the LJ region
        # when lambda = 0, then the potential and force are zero
        ash24.pair_coeff.set('A','A', sigma=0.9, lam=0.0)
        run(1)
        self.assertAlmostEqual(ash24.forces[0].energy, 0)
        self.assertAlmostEqual(ash24.forces[1].energy, 0)
        self.assertAlmostEqual(ash24.forces[0].force[0], 0)
        self.assertAlmostEqual(ash24.forces[1].force[0], 0)

        # partially switch on the LJ with lambda = 0.6
        ash24.pair_coeff.set('A','A', lam=0.6, sigma=0.9)
        run(1)
        U = -0.176169018326
        F = 3.86156575841
        self.assertAlmostEqual(ash24.forces[0].energy, 0.5*U, 3)
        self.assertAlmostEqual(ash24.forces[1].energy, 0.5*U, 3)
        self.assertAlmostEqual(ash24.forces[0].force[0], F, 3)
        self.assertAlmostEqual(ash24.forces[1].force[0], -F, 3)

        # and check shift - force is the same
        ash24.pair_coeff.set('A','A', sigma=1.0, epsilon=1.0, lam=1.0)
        ash24.pair_coeff.set('A', 'A', r_cut=1.1)

        ash24.set_params(mode='no_shift')
        run(1)
        U = 0
        F = -96.0

        self.assertAlmostEqual(ash24.forces[0].energy, 0.5*U, 3)
        self.assertAlmostEqual(ash24.forces[1].energy, 0.5*U, 3)
        self.assertAlmostEqual(ash24.forces[0].force[0], F, 3)
        self.assertAlmostEqual(ash24.forces[1].force[0], -F, 3)
        ash24.set_params(mode='shift')
        run(1)
        U = 0.364872603786
        F = -96.0


        self.assertAlmostEqual(ash24.forces[0].energy, 0.5*U, 3)
        self.assertAlmostEqual(ash24.forces[1].energy, 0.5*U, 3)
        self.assertAlmostEqual(ash24.forces[0].force[0], F, 3)
        self.assertAlmostEqual(ash24.forces[1].force[0], -F, 3)



    # test the cases where the potential should be zero
    def test_noninteract(self):
        ash24 = azplugins.pair.ashbaugh24(r_cut=0.9, nlist = self.nl)

        # outside cutoff
        ash24.pair_coeff.set('A','A', epsilon=1.0, sigma=1.0, lam=0.5)
        ash24.set_params(mode="no_shift")

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = group.all())
        run(1)
        self.assertAlmostEqual(ash24.forces[0].energy, 0)
        self.assertAlmostEqual(ash24.forces[1].energy, 0)
        self.assertAlmostEqual(ash24.forces[0].force[0], 0)
        self.assertAlmostEqual(ash24.forces[1].force[0], 0)

        # inside cutoff but epsilon = 0
        ash24.pair_coeff.set('A','A', epsilon=0.0, sigma=1.0, lam=0.5, r_cut=1.5)
        run(1)
        self.assertAlmostEqual(ash24.forces[0].energy, 0)
        self.assertAlmostEqual(ash24.forces[1].energy, 0)
        self.assertAlmostEqual(ash24.forces[0].force[0], 0)
        self.assertAlmostEqual(ash24.forces[1].force[0], 0)

    def tearDown(self):
        del self.nl
        context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
