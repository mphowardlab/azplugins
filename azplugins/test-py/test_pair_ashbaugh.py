# Copyright (c) 2018-2019, Michael P. Howard
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

# azplugins.pair.ashbaugh
class pair_ashbaugh_tests(unittest.TestCase):
    def setUp(self):
        # raw snapshot is fine, just needs to have the types
        snap = data.make_snapshot(N=100, box=data.boxdim(L=20), particle_types=['A'])
        self.s = init.read_snapshot(snap)
        self.nl = md.nlist.cell()
        context.current.sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        ash = azplugins.pair.ashbaugh(r_cut=3.0, nlist = self.nl)
        ash.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, alpha=1.0, lam=0.5, r_cut=2.5, r_on=2.0)
        ash.update_coeffs()

    # test missing coefficients
    def test_set_missing_epsilon(self):
        ash = azplugins.pair.ashbaugh(r_cut=3.0, nlist = self.nl)
        ash.pair_coeff.set('A', 'A', sigma=1.0, lam=0.5)
        self.assertRaises(RuntimeError, ash.update_coeffs)

    # test missing coefficients
    def test_set_missing_sigma(self):
        ash = azplugins.pair.ashbaugh(r_cut=3.0, nlist = self.nl)
        ash.pair_coeff.set('A', 'A', epsilon=1.0, lam=0.5)
        self.assertRaises(RuntimeError, ash.update_coeffs)

    # test missing coefficients
    def test_set_missing_lam(self):
        ash = azplugins.pair.ashbaugh(r_cut=3.0, nlist = self.nl)
        ash.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
        self.assertRaises(RuntimeError, ash.update_coeffs)

    # test missing coefficients
    def test_missing_AA(self):
        ash = azplugins.pair.ashbaugh(r_cut=3.0, nlist = self.nl)
        self.assertRaises(RuntimeError, ash.update_coeffs)

    # test set params
    def test_set_params(self):
        ash = azplugins.pair.ashbaugh(r_cut=3.0, nlist = self.nl)
        ash.set_params(mode="no_shift")
        ash.set_params(mode="shift")
        ash.set_params(mode="xplor")
        self.assertRaises(RuntimeError, ash.set_params, mode="blah")

    # test default coefficients
    def test_default_coeff(self):
        ash = azplugins.pair.ashbaugh(r_cut=3.0, nlist = self.nl)
        # (r_cut, and r_on are default)
        ash.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, lam=0.5)
        ash.update_coeffs()

    # test max rcut
    def test_max_rcut(self):
        ash = azplugins.pair.ashbaugh(r_cut=2.5, nlist = self.nl)
        ash.pair_coeff.set('A', 'A', sigma=1.0, epsilon=1.0)
        self.assertAlmostEqual(2.5, ash.get_max_rcut())
        ash.pair_coeff.set('A', 'A', r_cut = 2.0)
        self.assertAlmostEqual(2.0, ash.get_max_rcut())

    # test specific nlist subscription
    def test_nlist_subscribe(self):
        ash = azplugins.pair.ashbaugh(r_cut=2.5, nlist = self.nl)

        ash.pair_coeff.set('A', 'A', sigma = 1.0, epsilon=1.0)
        self.nl.update_rcut()
        self.assertAlmostEqual(2.5, self.nl.r_cut.get_pair('A','A'))

        ash.pair_coeff.set('A', 'A', r_cut = 2.0)
        self.nl.update_rcut()
        self.assertAlmostEqual(2.0, self.nl.r_cut.get_pair('A','A'))

    # test coeff list
    def test_coeff_list(self):
        ash = azplugins.pair.ashbaugh(r_cut=3.0, nlist = self.nl)
        ash.pair_coeff.set(['A', 'B'], ['A', 'C'], epsilon=1.0, sigma=1.0, lam=0.5, r_cut=2.5, r_on=2.0)
        ash.update_coeffs()

    # test adding types
    def test_type_add(self):
        ash = azplugins.pair.ashbaugh(r_cut=3.0, nlist = self.nl)
        ash.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, lam=0.5)
        self.s.particles.types.add('B')
        self.assertRaises(RuntimeError, ash.update_coeffs)
        ash.pair_coeff.set('A', 'B', epsilon=1.0, sigma=1.0, lam=0.5)
        ash.pair_coeff.set('B', 'B', epsilon=1.0, sigma=1.0, lam=0.5)
        ash.update_coeffs()

    def tearDown(self):
        del self.s, self.nl
        context.initialize()

# test the validity of the pair potential
class potential_ashbaugh_tests(unittest.TestCase):
    def setUp(self):
        snap = data.make_snapshot(N=2, box=data.boxdim(L=20),particle_types=['A'])
        if comm.get_rank() == 0:
            snap.particles.position[0] = (0,0,0)
            snap.particles.position[1] = (1.05,0,0)
        init.read_snapshot(snap)
        self.nl = md.nlist.cell()

    # test the calculation of force and potential
    def test_potential(self):
        ash = azplugins.pair.ashbaugh(r_cut=3.0, nlist = self.nl)

        # test when it's in the wca part, no potential shifting
        ash.pair_coeff.set('A','A', epsilon=2.0, sigma=1.05, lam=0.0)
        ash.set_params(mode="no_shift")

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = group.all())
        run(1)
        U = 2.0
        F = -45.7143
        f0 = ash.forces[0].force
        f1 = ash.forces[1].force
        e0 = ash.forces[0].energy
        e1 = ash.forces[1].energy

        self.assertAlmostEqual(e0,0.5*U,3)
        self.assertAlmostEqual(e1,0.5*U,3)

        self.assertAlmostEqual(f0[0],F,3)
        self.assertAlmostEqual(f0[1],0)
        self.assertAlmostEqual(f0[2],0)

        self.assertAlmostEqual(f1[0],-F,3)
        self.assertAlmostEqual(f1[1],0)
        self.assertAlmostEqual(f1[2],0)

        # change lambda to check for shifting of energy (force stays the same)
        ash.pair_coeff.set('A','A', lam=0.5)
        run(1)
        U = 1.0
        F = -45.7143
        self.assertAlmostEqual(ash.forces[0].energy, 0.5*U, 3)
        self.assertAlmostEqual(ash.forces[1].energy, 0.5*U, 3)
        self.assertAlmostEqual(ash.forces[0].force[0], F, 3)
        self.assertAlmostEqual(ash.forces[1].force[0], -F, 3)

        # change sigma so that now the particle is in the LJ region
        # when lambda = 0, then the potential and force are zero
        ash.pair_coeff.set('A','A', sigma=0.5, lam=0.0)
        run(1)
        self.assertAlmostEqual(ash.forces[0].energy, 0)
        self.assertAlmostEqual(ash.forces[1].energy, 0)
        self.assertAlmostEqual(ash.forces[0].force[0], 0)
        self.assertAlmostEqual(ash.forces[1].force[0], 0)

        # partially switch on the LJ with lambda = 0.5
        ash.pair_coeff.set('A','A', lam=0.5)
        run(1)
        U = -0.0460947
        F = 0.260291
        self.assertAlmostEqual(ash.forces[0].energy, 0.5*U, 3)
        self.assertAlmostEqual(ash.forces[1].energy, 0.5*U, 3)
        self.assertAlmostEqual(ash.forces[0].force[0], F, 3)
        self.assertAlmostEqual(ash.forces[1].force[0], -F, 3)

        # test that energy shifting works (bump up sigma so that at rcut = 3 the shift is reasonable)
        # check wca is shifted first
        ash.pair_coeff.set('A','A', sigma=1.05)
        ash.set_params(mode='shift')
        run(1)
        U = 1.00734
        F = -45.7143
        self.assertAlmostEqual(ash.forces[0].energy, 0.5*U, 3)
        self.assertAlmostEqual(ash.forces[1].energy, 0.5*U, 3)
        self.assertAlmostEqual(ash.forces[0].force[0], F, 3)
        self.assertAlmostEqual(ash.forces[1].force[0], -F, 3)

        # and check lj
        ash.pair_coeff.set('A','A', sigma=0.85)
        ash.set_params(mode='shift')
        run(1)
        U = -0.806849
        F = 2.81197
        self.assertAlmostEqual(ash.forces[0].energy, 0.5*U, 3)
        self.assertAlmostEqual(ash.forces[1].energy, 0.5*U, 3)
        self.assertAlmostEqual(ash.forces[0].force[0], F, 3)
        self.assertAlmostEqual(ash.forces[1].force[0], -F, 3)

    # test alpha parameter in potential. if potential is handled right,
    # coefficients are processed correctly and force will also be correct
    def test_alpha(self):
        ash = azplugins.pair.ashbaugh(r_cut=3.0, nlist = self.nl)
        ash.pair_coeff.set('A','A', epsilon=2.0, sigma=1.05, alpha=0.5, lam=0.5)
        ash.set_params(mode="no_shift")

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = group.all())
        run(1)

        U = 4.25
        self.assertAlmostEqual(ash.forces[0].energy,0.5*U,3)
        self.assertAlmostEqual(ash.forces[1].energy,0.5*U,3)

        ash.pair_coeff.set('A','A', sigma=0.5)
        run(1)
        U = -0.022775444603705584
        self.assertAlmostEqual(ash.forces[0].energy,0.5*U,3)
        self.assertAlmostEqual(ash.forces[1].energy,0.5*U,3)

    # test the cases where the potential should be zero
    def test_noninteract(self):
        ash = azplugins.pair.ashbaugh(r_cut=1.0, nlist = self.nl)

        # outside cutoff
        ash.pair_coeff.set('A','A', epsilon=1.0, sigma=1.0, lam=0.5)
        ash.set_params(mode="no_shift")

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = group.all())
        run(1)
        self.assertAlmostEqual(ash.forces[0].energy, 0)
        self.assertAlmostEqual(ash.forces[1].energy, 0)
        self.assertAlmostEqual(ash.forces[0].force[0], 0)
        self.assertAlmostEqual(ash.forces[1].force[0], 0)

        # inside cutoff but epsilon = 0
        ash.pair_coeff.set('A','A', epsilon=0.0, sigma=1.0, lam=0.5, r_cut=3.0)
        run(1)
        self.assertAlmostEqual(ash.forces[0].energy, 0)
        self.assertAlmostEqual(ash.forces[1].energy, 0)
        self.assertAlmostEqual(ash.forces[0].force[0], 0)
        self.assertAlmostEqual(ash.forces[1].force[0], 0)

    def tearDown(self):
        del self.nl
        context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
