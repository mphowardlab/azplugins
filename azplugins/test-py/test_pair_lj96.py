# Copyright (c) 2016-2018, Panagiotopoulos Group, Princeton University
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: sjiao

from hoomd import *
from hoomd import md
context.initialize()
try:
    from hoomd import azplugins
except ImportError:
    import azplugins
import unittest

# azplugins.pair.lj96
class pair_lj96_tests(unittest.TestCase):
    def setUp(self):
        # raw snapshot is fine, just needs to have the types
        snap = data.make_snapshot(N=100, box=data.boxdim(L=20), particle_types=['A'])
        self.s = init.read_snapshot(snap)
        self.nl = md.nlist.cell()
        context.current.sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        lj96 = azplugins.pair.lj96(r_cut=3.0, nlist = self.nl)
        lj96.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, alpha=1.0, r_cut=2.5, r_on=2.0)
        lj96.update_coeffs()

    # test missing coefficients
    def test_set_missing_epsilon(self):
        lj96 = azplugins.pair.lj96(r_cut=3.0, nlist = self.nl)
        lj96.pair_coeff.set('A', 'A', sigma=1.0)
        self.assertRaises(RuntimeError, lj96.update_coeffs)

    # test missing coefficients
    def test_set_missing_sigma(self):
        lj96 = azplugins.pair.lj96(r_cut=3.0, nlist = self.nl)
        lj96.pair_coeff.set('A', 'A', epsilon=1.0)
        self.assertRaises(RuntimeError, lj96.update_coeffs)

    # test missing coefficients
    def test_missing_AA(self):
        lj96 = azplugins.pair.lj96(r_cut=3.0, nlist = self.nl)
        self.assertRaises(RuntimeError, lj96.update_coeffs)

    # test set params
    def test_set_params(self):
        lj96 = azplugins.pair.lj96(r_cut=3.0, nlist = self.nl)
        lj96.set_params(mode="no_shift")
        lj96.set_params(mode="shift")
        lj96.set_params(mode="xplor")
        self.assertRaises(RuntimeError, lj96.set_params, mode="blah")

    # test default coefficients
    def test_default_coeff(self):
        lj96 = azplugins.pair.lj96(r_cut=3.0, nlist = self.nl)
        # (r_cut, and r_on are default)
        lj96.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
        lj96.update_coeffs()

    # test max rcut
    def test_max_rcut(self):
        lj96 = azplugins.pair.lj96(r_cut=2.5, nlist = self.nl)
        lj96.pair_coeff.set('A', 'A', sigma=1.0, epsilon=1.0)
        self.assertAlmostEqual(2.5, lj96.get_max_rcut())
        lj96.pair_coeff.set('A', 'A', r_cut = 2.0)
        self.assertAlmostEqual(2.0, lj96.get_max_rcut())

    # test specific nlist subscription
    def test_nlist_subscribe(self):
        lj96 = azplugins.pair.lj96(r_cut=2.5, nlist = self.nl)

        lj96.pair_coeff.set('A', 'A', sigma = 1.0, epsilon=1.0)
        self.nl.update_rcut()
        self.assertAlmostEqual(2.5, self.nl.r_cut.get_pair('A','A'))

        lj96.pair_coeff.set('A', 'A', r_cut = 2.0)
        self.nl.update_rcut()
        self.assertAlmostEqual(2.0, self.nl.r_cut.get_pair('A','A'))

    # test coeff list
    def test_coeff_list(self):
        lj96 = azplugins.pair.lj96(r_cut=3.0, nlist = self.nl)
        lj96.pair_coeff.set(['A', 'B'], ['A', 'C'], epsilon=1.0, sigma=1.0, r_cut=2.5, r_on=2.0)
        lj96.update_coeffs()

    # test adding types
    def test_type_add(self):
        lj96 = azplugins.pair.lj96(r_cut=3.0, nlist = self.nl)
        lj96.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
        self.s.particles.types.add('B')
        self.assertRaises(RuntimeError, lj96.update_coeffs)
        lj96.pair_coeff.set('A', 'B', epsilon=1.0, sigma=1.0)
        lj96.pair_coeff.set('B', 'B', epsilon=1.0, sigma=1.0)
        lj96.update_coeffs()

    def tearDown(self):
        del self.s, self.nl
        context.initialize()

# test the validity of the pair potential
class potential_lj96_tests(unittest.TestCase):
    def setUp(self):
        snap = data.make_snapshot(N=2, box=data.boxdim(L=20),particle_types=['A'])
        if comm.get_rank() == 0:
            snap.particles.position[0] = (0,0,0)
            snap.particles.position[1] = (1.05,0,0)
        init.read_snapshot(snap)
        self.nl = md.nlist.cell()

    # test the calculation of force and potential
    def test_potential(self):
        lj96 = azplugins.pair.lj96(r_cut=3.0, nlist = self.nl)
        lj96.pair_coeff.set('A','A', epsilon=2.0, sigma=1.05)
        lj96.set_params(mode="no_shift")

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = group.all())
        run(1)
        U = 0.0
        F = -38.5714
        f0 = lj96.forces[0].force
        f1 = lj96.forces[1].force
        e0 = lj96.forces[0].energy
        e1 = lj96.forces[1].energy

        self.assertAlmostEqual(e0,0.5*U,3)
        self.assertAlmostEqual(e1,0.5*U,3)

        self.assertAlmostEqual(f0[0],F,3)
        self.assertAlmostEqual(f0[1],0)
        self.assertAlmostEqual(f0[2],0)

        self.assertAlmostEqual(f1[0],-F,3)
        self.assertAlmostEqual(f1[1],0)
        self.assertAlmostEqual(f1[2],0)

        lj96.pair_coeff.set('A','A', sigma=1.05)
        lj96.set_params(mode='shift')
        run(1)
        U = 0.0238
        F = -38.5714
        self.assertAlmostEqual(lj96.forces[0].energy, 0.5*U, 3)
        self.assertAlmostEqual(lj96.forces[1].energy, 0.5*U, 3)
        self.assertAlmostEqual(lj96.forces[0].force[0], F, 3)
        self.assertAlmostEqual(lj96.forces[1].force[0], -F, 3)

        lj96.pair_coeff.set('A','A', sigma=0.85)
        lj96.set_params(mode='shift')
        run(1)
        U = -1.7770
        F = 4.4343
        self.assertAlmostEqual(lj96.forces[0].energy, 0.5*U, 3)
        self.assertAlmostEqual(lj96.forces[1].energy, 0.5*U, 3)
        self.assertAlmostEqual(lj96.forces[0].force[0], F, 3)
        self.assertAlmostEqual(lj96.forces[1].force[0], -F, 3)

    # test alpha parameter in potential. if potential is handled right,
    # coefficients are processed correctly and force will also be correct
    def test_alpha(self):
        lj96 = azplugins.pair.lj96(r_cut=3.0, nlist = self.nl)
        lj96.pair_coeff.set('A','A', epsilon=2.0, sigma=1.05, alpha=0.5)
        lj96.set_params(mode="no_shift")

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = group.all())
        run(1)

        U = 6.7500
        self.assertAlmostEqual(lj96.forces[0].energy,0.5*U,3)
        self.assertAlmostEqual(lj96.forces[1].energy,0.5*U,3)

        lj96.pair_coeff.set('A','A', sigma=0.5)
        run(1)
        U = -0.0617
        self.assertAlmostEqual(lj96.forces[0].energy,0.5*U,3)
        self.assertAlmostEqual(lj96.forces[1].energy,0.5*U,3)

    # test the cases where the potential should be zero
    def test_noninteract(self):
        lj96 = azplugins.pair.lj96(r_cut=1.0, nlist = self.nl)

        # outside cutoff
        lj96.pair_coeff.set('A','A', epsilon=1.0, sigma=1.0)
        lj96.set_params(mode="no_shift")

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = group.all())
        run(1)
        self.assertAlmostEqual(lj96.forces[0].energy, 0)
        self.assertAlmostEqual(lj96.forces[1].energy, 0)
        self.assertAlmostEqual(lj96.forces[0].force[0], 0)
        self.assertAlmostEqual(lj96.forces[1].force[0], 0)

        # inside cutoff but epsilon = 0
        lj96.pair_coeff.set('A','A', epsilon=0.0, sigma=1.0, r_cut=3.0)
        run(1)
        self.assertAlmostEqual(lj96.forces[0].energy, 0)
        self.assertAlmostEqual(lj96.forces[1].energy, 0)
        self.assertAlmostEqual(lj96.forces[0].force[0], 0)
        self.assertAlmostEqual(lj96.forces[1].force[0], 0)

    def tearDown(self):
        del self.nl
        context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
