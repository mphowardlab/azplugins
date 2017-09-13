# Copyright (c) 2016-2017, Panagiotopoulos Group, Princeton University
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

# azplugins.pair.slj
class pair_slj_tests(unittest.TestCase):
    def setUp(self):
        # raw snapshot is fine, just needs to have the types
        snap = data.make_snapshot(N=100, box=data.boxdim(L=20), particle_types=['A'])
        self.s = init.read_snapshot(snap)
        self.nl = md.nlist.cell()
        context.current.sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        slj = azplugins.pair.slj(r_cut=3.0, nlist = self.nl)
        slj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, alpha=1.0, delta=1.0, r_cut=2.5, r_on=2.0)
        slj.update_coeffs()

    # test missing epsilon coefficient
    def test_set_missing_epsilon(self):
        slj = azplugins.pair.slj(r_cut=3.0, nlist = self.nl)
        slj.pair_coeff.set('A', 'A', sigma=1.0, delta=1.0)
        self.assertRaises(RuntimeError, slj.update_coeffs)

    # test missing sigma coefficient
    def test_set_missing_sigma(self):
        slj = azplugins.pair.slj(r_cut=3.0, nlist = self.nl)
        slj.pair_coeff.set('A', 'A', epsilon=1.0, delta=1.0)
        self.assertRaises(RuntimeError, slj.update_coeffs)

    # test missing delta coefficient
    def test_set_missing_delta(self):
        slj = azplugins.pair.slj(r_cut=3.0, nlist = self.nl)
        slj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
        self.assertRaises(RuntimeError, slj.update_coeffs)

    # test missing type coefficients
    def test_missing_AA(self):
        slj = azplugins.pair.slj(r_cut=3.0, nlist = self.nl)
        self.assertRaises(RuntimeError, slj.update_coeffs)

    # test set params
    def test_set_params(self):
        slj = azplugins.pair.slj(r_cut=3.0, nlist = self.nl)
        slj.set_params(mode="no_shift")
        slj.set_params(mode="shift")
        slj.set_params(mode="xplor")
        self.assertRaises(RuntimeError, slj.set_params, mode="blah")

    # test default coefficients
    def test_default_coeff(self):
        slj = azplugins.pair.slj(r_cut=3.0, nlist = self.nl)
        # (r_cut, and r_on are default)
        slj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, delta=1.0)
        slj.update_coeffs()

    # test max rcut
    def test_max_rcut(self):
        slj = azplugins.pair.slj(r_cut=2.5, nlist = self.nl)
        slj.pair_coeff.set('A', 'A', sigma=1.0, epsilon=1.0, delta=1.0)
        self.assertAlmostEqual(2.5, slj.get_max_rcut())
        slj.pair_coeff.set('A', 'A', r_cut = 2.0)
        self.assertAlmostEqual(2.0, slj.get_max_rcut())

    # test specific nlist subscription
    def test_nlist_subscribe(self):
        slj = azplugins.pair.slj(r_cut=2.5, nlist = self.nl)

        slj.pair_coeff.set('A', 'A', sigma = 1.0, epsilon=1.0, delta=1.0)
        self.nl.update_rcut()
        self.assertAlmostEqual(2.5, self.nl.r_cut.get_pair('A','A'))

        slj.pair_coeff.set('A', 'A', r_cut = 2.0)
        self.nl.update_rcut()
        self.assertAlmostEqual(2.0, self.nl.r_cut.get_pair('A','A'))

    # test coeff list
    def test_coeff_list(self):
        slj = azplugins.pair.slj(r_cut=3.0, nlist = self.nl)
        slj.pair_coeff.set(['A', 'B'], ['A', 'C'], epsilon=1.0, sigma=1.0, delta=1.0, r_cut=2.5, r_on=2.0)
        slj.update_coeffs()

    # test adding types
    def test_type_add(self):
        slj = azplugins.pair.slj(r_cut=3.0, nlist = self.nl)
        slj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, delta=1.0)
        self.s.particles.types.add('B')
        self.assertRaises(RuntimeError, slj.update_coeffs)
        slj.pair_coeff.set('A', 'B', epsilon=1.0, sigma=1.0, delta=1.0)
        slj.pair_coeff.set('B', 'B', epsilon=1.0, sigma=1.0, delta=1.0)
        slj.update_coeffs()

    def tearDown(self):
        del self.s, self.nl
        context.initialize()

# test the validity of the pair potential
class potential_slj_tests(unittest.TestCase):
    def setUp(self):
        snap = data.make_snapshot(N=2, box=data.boxdim(L=20),particle_types=['A'])
        if comm.get_rank() == 0:
            snap.particles.position[0] = (0,0,0)
            snap.particles.position[1] = (1.05,0,0)
        init.read_snapshot(snap)
        self.nl = md.nlist.cell()

    # test the calculation of force and potential
    def test_potential(self):
        slj = azplugins.pair.slj(r_cut=2.0, nlist = self.nl)

        # some set of parameters which will give a potential of zero by shifting
        slj.pair_coeff.set('A','A', epsilon=2.0, sigma=0.95, delta=0.1)
        slj.set_params(mode="no_shift")

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = group.all())
        run(1)
        U = 0.0
        F = -50.5263
        f0 = slj.forces[0].force
        f1 = slj.forces[1].force
        e0 = slj.forces[0].energy
        e1 = slj.forces[1].energy

        self.assertAlmostEqual(e0,0.5*U,3)
        self.assertAlmostEqual(e1,0.5*U,3)

        self.assertAlmostEqual(f0[0],F,3)
        self.assertAlmostEqual(f0[1],0)
        self.assertAlmostEqual(f0[2],0)

        self.assertAlmostEqual(f1[0],-F,3)
        self.assertAlmostEqual(f1[1],0)
        self.assertAlmostEqual(f1[2],0)

        # shift the potential and check the energy (force stays the same)
        slj.set_params(mode='shift')
        run(1)
        U -= -0.123046875
        f0 = slj.forces[0].force
        f1 = slj.forces[1].force
        e0 = slj.forces[0].energy
        e1 = slj.forces[1].energy

        self.assertAlmostEqual(e0,0.5*U,3)
        self.assertAlmostEqual(e1,0.5*U,3)

        self.assertAlmostEqual(f0[0],F,3)
        self.assertAlmostEqual(f0[1],0)
        self.assertAlmostEqual(f0[2],0)

        self.assertAlmostEqual(f1[0],-F,3)
        self.assertAlmostEqual(f1[1],0)
        self.assertAlmostEqual(f1[2],0)

    # test alpha parameter in potential
    def test_alpha(self):
        slj = azplugins.pair.slj(r_cut=3.0, nlist = self.nl)
        slj.pair_coeff.set('A','A', epsilon=2.0, sigma=0.95, delta=0.1, alpha=0.5)
        slj.set_params(mode="no_shift")

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = group.all())
        run(1)

        U = 4.0
        F = 75.78947368421039
        self.assertAlmostEqual(slj.forces[0].energy,0.5*U,3)
        self.assertAlmostEqual(slj.forces[1].energy,0.5*U,3)
        self.assertAlmostEqual(slj.forces[0].force[0], -F,3)
        self.assertAlmostEqual(slj.forces[1].force[0], F,3)

    # test the cases where the potential should be zero
    def test_noninteract(self):
        slj = azplugins.pair.slj(r_cut=1.0, nlist = self.nl)

        # outside cutoff
        slj.pair_coeff.set('A','A', epsilon=1.0, sigma=0.95, delta=0.5)
        slj.set_params(mode="no_shift")

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = group.all())
        run(1)
        self.assertAlmostEqual(slj.forces[0].energy, 0)
        self.assertAlmostEqual(slj.forces[1].energy, 0)
        self.assertAlmostEqual(slj.forces[0].force[0], 0)
        self.assertAlmostEqual(slj.forces[1].force[0], 0)

        # inside cutoff but epsilon = 0
        slj.pair_coeff.set('A','A', epsilon=0.0, r_cut=3.0)
        run(1)
        self.assertAlmostEqual(slj.forces[0].energy, 0)
        self.assertAlmostEqual(slj.forces[1].energy, 0)
        self.assertAlmostEqual(slj.forces[0].force[0], 0)
        self.assertAlmostEqual(slj.forces[1].force[0], 0)

    def tearDown(self):
        del self.nl
        context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
