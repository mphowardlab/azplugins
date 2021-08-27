# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021, Auburn University
# This file is part of the azplugins project, released under the Modified BSD License.

import hoomd
from hoomd import md
hoomd.context.initialize()
try:
    from hoomd import azplugins
except ImportError:
    import azplugins
import unittest

# azplugins.pair.lj124
class pair_lj124_tests(unittest.TestCase):
    def setUp(self):
        # raw snapshot is fine, just needs to have the types
        snap = hoomd.data.make_snapshot(N=100, box=hoomd.data.boxdim(L=20), particle_types=['A'])
        self.s = hoomd.init.read_snapshot(snap)
        self.nl = md.nlist.cell()
        hoomd.context.current.sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        lj124 = azplugins.pair.lj124(r_cut=3.0, nlist = self.nl)
        lj124.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, alpha=1.0, r_cut=2.5, r_on=2.0)
        lj124.update_coeffs()

    # test missing coefficients
    def test_set_missing_epsilon(self):
        lj124 = azplugins.pair.lj124(r_cut=3.0, nlist = self.nl)
        lj124.pair_coeff.set('A', 'A', sigma=1.0)
        self.assertRaises(RuntimeError, lj124.update_coeffs)

    # test missing coefficients
    def test_set_missing_sigma(self):
        lj124 = azplugins.pair.lj124(r_cut=3.0, nlist = self.nl)
        lj124.pair_coeff.set('A', 'A', epsilon=1.0)
        self.assertRaises(RuntimeError, lj124.update_coeffs)

    # test missing coefficients
    def test_missing_AA(self):
        lj124 = azplugins.pair.lj124(r_cut=3.0, nlist = self.nl)
        self.assertRaises(RuntimeError, lj124.update_coeffs)

    # test set params
    def test_set_params(self):
        lj124 = azplugins.pair.lj124(r_cut=3.0, nlist = self.nl)
        lj124.set_params(mode="no_shift")
        lj124.set_params(mode="shift")
        lj124.set_params(mode="xplor")
        self.assertRaises(RuntimeError, lj124.set_params, mode="blah")

    # test default coefficients
    def test_default_coeff(self):
        lj124 = azplugins.pair.lj124(r_cut=3.0, nlist = self.nl)
        # (r_cut, and r_on are default)
        lj124.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
        lj124.update_coeffs()

    # test max rcut
    def test_max_rcut(self):
        lj124 = azplugins.pair.lj124(r_cut=2.5, nlist = self.nl)
        lj124.pair_coeff.set('A', 'A', sigma=1.0, epsilon=1.0)
        self.assertAlmostEqual(2.5, lj124.get_max_rcut())
        lj124.pair_coeff.set('A', 'A', r_cut = 2.0)
        self.assertAlmostEqual(2.0, lj124.get_max_rcut())

    # test specific nlist subscription
    def test_nlist_subscribe(self):
        lj124 = azplugins.pair.lj124(r_cut=2.5, nlist = self.nl)

        lj124.pair_coeff.set('A', 'A', sigma = 1.0, epsilon=1.0)
        self.nl.update_rcut()
        self.assertAlmostEqual(2.5, self.nl.r_cut.get_pair('A','A'))

        lj124.pair_coeff.set('A', 'A', r_cut = 2.0)
        self.nl.update_rcut()
        self.assertAlmostEqual(2.0, self.nl.r_cut.get_pair('A','A'))

    # test coeff list
    def test_coeff_list(self):
        lj124 = azplugins.pair.lj124(r_cut=3.0, nlist = self.nl)
        lj124.pair_coeff.set(['A', 'B'], ['A', 'C'], epsilon=1.0, sigma=1.0, r_cut=2.5, r_on=2.0)
        lj124.update_coeffs()

    # test adding types
    def test_type_add(self):
        lj124 = azplugins.pair.lj124(r_cut=3.0, nlist = self.nl)
        lj124.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
        self.s.particles.types.add('B')
        self.assertRaises(RuntimeError, lj124.update_coeffs)
        lj124.pair_coeff.set('A', 'B', epsilon=1.0, sigma=1.0)
        lj124.pair_coeff.set('B', 'B', epsilon=1.0, sigma=1.0)
        lj124.update_coeffs()

    def tearDown(self):
        del self.s, self.nl
        hoomd.context.initialize()

# test the validity of the pair potential
class potential_lj124_tests(unittest.TestCase):
    def setUp(self):
        snap = hoomd.data.make_snapshot(N=2, box=hoomd.data.boxdim(L=20),particle_types=['A'])
        if hoomd.comm.get_rank() == 0:
            snap.particles.position[0] = (0,0,0)
            snap.particles.position[1] = (1.05,0,0)
        hoomd.init.read_snapshot(snap)
        self.nl = md.nlist.cell()

    # test the calculation of force and potential
    def test_potential(self):
        lj124 = azplugins.pair.lj124(r_cut=3.0, nlist = self.nl)
        lj124.pair_coeff.set('A','A', epsilon=2.0, sigma=1.05)
        lj124.set_params(mode="no_shift")

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = hoomd.group.all())
        hoomd.run(1)
        U = 0.0
        F = -39.5897
        f0 = lj124.forces[0].force
        f1 = lj124.forces[1].force
        e0 = lj124.forces[0].energy
        e1 = lj124.forces[1].energy

        self.assertAlmostEqual(e0,0.5*U,3)
        self.assertAlmostEqual(e1,0.5*U,3)

        self.assertAlmostEqual(f0[0],F,3)
        self.assertAlmostEqual(f0[1],0)
        self.assertAlmostEqual(f0[2],0)

        self.assertAlmostEqual(f1[0],-F,3)
        self.assertAlmostEqual(f1[1],0)
        self.assertAlmostEqual(f1[2],0)

        lj124.pair_coeff.set('A','A', sigma=1.05)
        lj124.set_params(mode='shift')
        hoomd.run(1)
        U = 0.0780
        F = -39.5897
        self.assertAlmostEqual(lj124.forces[0].energy, 0.5*U, 3)
        self.assertAlmostEqual(lj124.forces[1].energy, 0.5*U, 3)
        self.assertAlmostEqual(lj124.forces[0].force[0], F, 3)
        self.assertAlmostEqual(lj124.forces[1].force[0], -F, 3)

        lj124.pair_coeff.set('A','A', sigma=0.85)
        lj124.set_params(mode='shift')
        hoomd.run(1)
        U = -1.7865
        F = 3.7974
        self.assertAlmostEqual(lj124.forces[0].energy, 0.5*U, 3)
        self.assertAlmostEqual(lj124.forces[1].energy, 0.5*U, 3)
        self.assertAlmostEqual(lj124.forces[0].force[0], F, 3)
        self.assertAlmostEqual(lj124.forces[1].force[0], -F, 3)

    # test alpha parameter in potential. if potential is handled right,
    # coefficients are processed correctly and force will also be correct
    def test_alpha(self):
        lj124 = azplugins.pair.lj124(r_cut=3.0, nlist = self.nl)
        lj124.pair_coeff.set('A','A', epsilon=2.0, sigma=1.05, alpha=0.5)
        lj124.set_params(mode="no_shift")

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = hoomd.group.all())
        hoomd.run(1)

        U = 2.5981
        self.assertAlmostEqual(lj124.forces[0].energy,0.5*U,3)
        self.assertAlmostEqual(lj124.forces[1].energy,0.5*U,3)

        lj124.pair_coeff.set('A','A', sigma=0.5)
        hoomd.run(1)
        U = -0.1329
        self.assertAlmostEqual(lj124.forces[0].energy,0.5*U,3)
        self.assertAlmostEqual(lj124.forces[1].energy,0.5*U,3)

    # test the cases where the potential should be zero
    def test_noninteract(self):
        lj124 = azplugins.pair.lj124(r_cut=1.0, nlist = self.nl)

        # outside cutoff
        lj124.pair_coeff.set('A','A', epsilon=1.0, sigma=1.0)
        lj124.set_params(mode="no_shift")

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = hoomd.group.all())
        hoomd.run(1)
        self.assertAlmostEqual(lj124.forces[0].energy, 0)
        self.assertAlmostEqual(lj124.forces[1].energy, 0)
        self.assertAlmostEqual(lj124.forces[0].force[0], 0)
        self.assertAlmostEqual(lj124.forces[1].force[0], 0)

        # inside cutoff but epsilon = 0
        lj124.pair_coeff.set('A','A', epsilon=0.0, sigma=1.0, r_cut=3.0)
        hoomd.run(1)
        self.assertAlmostEqual(lj124.forces[0].energy, 0)
        self.assertAlmostEqual(lj124.forces[1].energy, 0)
        self.assertAlmostEqual(lj124.forces[0].force[0], 0)
        self.assertAlmostEqual(lj124.forces[1].force[0], 0)

    def tearDown(self):
        del self.nl
        hoomd.context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
