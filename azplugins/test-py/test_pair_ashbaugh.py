# -*- coding: iso-8859-1 -*-
# Maintainer: mphoward

from hoomd import *
from hoomd import md
from hoomd import azplugins
context.initialize()
import unittest
import os

# azplugins.pair.ashbaugh
class pair_ashbaugh_tests(unittest.TestCase):
    def setUp(self):
        print
        # raw snapshot is fine, just needs to have the types
        snap = data.make_snapshot(N=100, box=data.boxdim(L=20), particle_types=['A'])
        self.s = init.read_snapshot(snap)
        self.nl = md.nlist.cell()
        context.current.sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        lj = azplugins.pair.ashbaugh(r_cut=3.0, nlist = self.nl);
        lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, lam=0.5, r_cut=2.5, r_on=2.0);
        lj.update_coeffs();

    # test missing coefficients
    def test_set_missing_epsilon(self):
        lj = azplugins.pair.ashbaugh(r_cut=3.0, nlist = self.nl);
        lj.pair_coeff.set('A', 'A', sigma=1.0, lam=0.5);
        self.assertRaises(RuntimeError, lj.update_coeffs);

    # test missing coefficients
    def test_set_missing_sigma(self):
        lj = azplugins.pair.ashbaugh(r_cut=3.0, nlist = self.nl);
        lj.pair_coeff.set('A', 'A', epsilon=1.0, lam=0.5);
        self.assertRaises(RuntimeError, lj.update_coeffs);

    # test missing coefficients
    def test_set_missing_lam(self):
        lj = azplugins.pair.ashbaugh(r_cut=3.0, nlist = self.nl);
        lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0);
        self.assertRaises(RuntimeError, lj.update_coeffs);

    # test missing coefficients
    def test_missing_AA(self):
        lj = azplugins.pair.ashbaugh(r_cut=3.0, nlist = self.nl);
        self.assertRaises(RuntimeError, lj.update_coeffs);

    # test set params
    def test_set_params(self):
        lj = azplugins.pair.ashbaugh(r_cut=3.0, nlist = self.nl);
        lj.set_params(mode="no_shift");
        lj.set_params(mode="shift");
        lj.set_params(mode="xplor");
        self.assertRaises(RuntimeError, lj.set_params, mode="blah");

    # test default coefficients
    def test_default_coeff(self):
        lj = azplugins.pair.ashbaugh(r_cut=3.0, nlist = self.nl);
        # (r_cut, and r_on are default)
        lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, lam=0.5)
        lj.update_coeffs()

    # test max rcut
    def test_max_rcut(self):
        lj = azplugins.pair.ashbaugh(r_cut=2.5, nlist = self.nl);
        lj.pair_coeff.set('A', 'A', sigma=1.0, epsilon=1.0)
        self.assertAlmostEqual(2.5, lj.get_max_rcut());
        lj.pair_coeff.set('A', 'A', r_cut = 2.0)
        self.assertAlmostEqual(2.0, lj.get_max_rcut());

    # test specific nlist subscription
    def test_nlist_subscribe(self):
        lj = azplugins.pair.ashbaugh(r_cut=2.5, nlist = self.nl);

        lj.pair_coeff.set('A', 'A', sigma = 1.0, epsilon=1.0)
        self.nl.update_rcut();
        self.assertAlmostEqual(2.5, self.nl.r_cut.get_pair('A','A'));

        lj.pair_coeff.set('A', 'A', r_cut = 2.0)
        self.nl.update_rcut();
        self.assertAlmostEqual(2.0, self.nl.r_cut.get_pair('A','A'));

    # test coeff list
    def test_coeff_list(self):
        lj = azplugins.pair.ashbaugh(r_cut=3.0, nlist = self.nl);
        lj.pair_coeff.set(['A', 'B'], ['A', 'C'], epsilon=1.0, sigma=1.0, lam=0.5, r_cut=2.5, r_on=2.0);
        lj.update_coeffs();

    # test adding types
    def test_type_add(self):
        lj = azplugins.pair.ashbaugh(r_cut=3.0, nlist = self.nl);
        lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, lam=0.5);
        self.s.particles.types.add('B')
        self.assertRaises(RuntimeError, lj.update_coeffs);
        lj.pair_coeff.set('A', 'B', epsilon=1.0, sigma=1.0, lam=0.5)
        lj.pair_coeff.set('B', 'B', epsilon=1.0, sigma=1.0, lam=0.5)
        lj.update_coeffs();

    # test for invalid lambda values
    def test_invalid_lambda(self):
        lj = azplugins.pair.ashbaugh(r_cut=3.0, nlist=self.nl)
        lj.pair_coeff.set('A','A', epsilon=1.0, sigma=1.0, lam=-0.1)
        self.assertRaises(RuntimeError, lj.update_coeffs)

        lj.pair_coeff.set('A','A', lam=1.1)
        self.assertRaises(RuntimeError, lj.update_coeffs)

    def tearDown(self):
        del self.s, self.nl
        context.initialize();

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
        lj = azplugins.pair.ashbaugh(r_cut=3.0, nlist = self.nl)

        # test when it's in the wca part, no potential shifting
        lj.pair_coeff.set('A','A', epsilon=2.0, sigma=1.05, lam=0.0)
        lj.set_params(mode="no_shift")

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = group.all())
        run(1)
        f0 = lj.forces[0].force
        f1 = lj.forces[1].force
        e0 = lj.forces[0].energy
        e1 = lj.forces[1].energy

        self.assertAlmostEqual(e0,0.5*2.0,3)
        self.assertAlmostEqual(e1,0.5*2.0,3)

        self.assertAlmostEqual(f0[0],-45.7143,3)
        self.assertAlmostEqual(f0[1],0)
        self.assertAlmostEqual(f0[2],0)

        self.assertAlmostEqual(f1[0],45.7143,3)
        self.assertAlmostEqual(f1[1],0)
        self.assertAlmostEqual(f1[2],0)

        # change lambda to check for shifting of energy (force stays the same)
        lj.pair_coeff.set('A','A', lam=0.5)
        run(1)
        self.assertAlmostEqual(lj.forces[0].energy, 0.5*1.0,3)
        self.assertAlmostEqual(lj.forces[1].energy, 0.5*1.0,3)
        self.assertAlmostEqual(lj.forces[0].force[0], -45.7143,3)
        self.assertAlmostEqual(lj.forces[1].force[0], 45.7143,3)

        # change sigma so that now the particle is in the LJ region
        # when lambda = 0, then the potential and force are zero
        lj.pair_coeff.set('A','A', sigma=0.5, lam=0.0)
        run(1)
        self.assertAlmostEqual(lj.forces[0].energy, 0)
        self.assertAlmostEqual(lj.forces[1].energy, 0)
        self.assertAlmostEqual(lj.forces[0].force[0], 0)
        self.assertAlmostEqual(lj.forces[1].force[0], 0)

        # partially switch on the LJ with lambda = 0.5
        lj.pair_coeff.set('A','A', lam=0.5)
        run(1)
        self.assertAlmostEqual(lj.forces[0].energy, 0.5*-0.0460947, 3)
        self.assertAlmostEqual(lj.forces[1].energy, 0.5*-0.0460947, 3)
        self.assertAlmostEqual(lj.forces[0].force[0], 0.260291, 3)
        self.assertAlmostEqual(lj.forces[1].force[0], -0.260291, 3)

        # test that energy shifting works (bump up sigma so that at rcut = 3 the shift is reasonable)
        # check wca is shifted first
        lj.pair_coeff.set('A','A', sigma=1.05)
        lj.set_params(mode='shift')
        run(1)
        self.assertAlmostEqual(lj.forces[0].energy, 0.5*1.00734, 3)
        self.assertAlmostEqual(lj.forces[1].energy, 0.5*1.00734, 3)
        self.assertAlmostEqual(lj.forces[0].force[0], -45.7143, 3)
        self.assertAlmostEqual(lj.forces[1].force[0], 45.7143, 3)

        # and check lj
        lj.pair_coeff.set('A','A', sigma=0.85)
        lj.set_params(mode='shift')
        run(1)
        self.assertAlmostEqual(lj.forces[0].energy, 0.5*-0.806849, 3)
        self.assertAlmostEqual(lj.forces[1].energy, 0.5*-0.806849, 3)
        self.assertAlmostEqual(lj.forces[0].force[0], 2.81197, 3)
        self.assertAlmostEqual(lj.forces[1].force[0], -2.81197, 3)

    # test the cases where the potential should be zero
    def test_noninteract(self):
        lj = azplugins.pair.ashbaugh(r_cut=1.0, nlist = self.nl)

        # outside cutoff
        lj.pair_coeff.set('A','A', epsilon=1.0, sigma=1.0, lam=0.5)
        lj.set_params(mode="no_shift")

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = group.all())
        run(1)
        self.assertAlmostEqual(lj.forces[0].energy, 0)
        self.assertAlmostEqual(lj.forces[1].energy, 0)
        self.assertAlmostEqual(lj.forces[0].force[0], 0)
        self.assertAlmostEqual(lj.forces[1].force[0], 0)

        # inside cutoff but epsilon = 0
        lj.pair_coeff.set('A','A', epsilon=0.0, sigma=1.0, lam=0.5, r_cut=3.0)
        run(1)
        self.assertAlmostEqual(lj.forces[0].energy, 0)
        self.assertAlmostEqual(lj.forces[1].energy, 0)
        self.assertAlmostEqual(lj.forces[0].force[0], 0)
        self.assertAlmostEqual(lj.forces[1].force[0], 0)

    def tearDown(self):
        del self.nl
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
