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

# azplugins.pair.colloid
class pair_colloid_tests(unittest.TestCase):
    def setUp(self):
        # raw snapshot is fine, just needs to have the types
        snap = data.make_snapshot(N=100, box=data.boxdim(L=20), particle_types=['A'])
        self.s = init.read_snapshot(snap)
        self.nl = md.nlist.cell()
        context.current.sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        coll = azplugins.pair.colloid(r_cut=3.0, nlist = self.nl)
        coll.pair_coeff.set('A', 'A', epsilon=144.0, sigma=1.0, style='slv-slv', r_cut=2.5, r_on=2.0)
        coll.update_coeffs()

    # test missing style
    def test_set_missing_epsilon(self):
        coll = azplugins.pair.colloid(r_cut=3.0, nlist = self.nl)
        coll.pair_coeff.set('A', 'A', sigma=1.0, style='slv-slv')
        self.assertRaises(RuntimeError, coll.update_coeffs)

    # test missing style
    def test_set_missing_style(self):
        coll = azplugins.pair.colloid(r_cut=3.0, nlist = self.nl)
        coll.pair_coeff.set('A', 'A', epsilon=144.0, sigma=1.0)
        self.assertRaises(RuntimeError, coll.update_coeffs)

    # test style list
    def test_set_wrong_style(self):
        coll = azplugins.pair.colloid(r_cut=3.0, nlist = self.nl)
        coll.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, style='slv-slv')
        coll.update_coeffs()
        coll.pair_coeff.set('A','A', style='coll-slv')
        coll.update_coeffs()
        coll.pair_coeff.set('A','A', style='coll-coll')
        coll.update_coeffs()
        coll.pair_coeff.set('A','A', style='not-a-style')
        self.assertRaises(RuntimeError, coll.update_coeffs)

    # test missing coefficients
    def test_missing_AA(self):
        coll = azplugins.pair.colloid(r_cut=3.0, nlist = self.nl)
        self.assertRaises(RuntimeError, coll.update_coeffs)

    # test set params
    def test_set_params(self):
        coll = azplugins.pair.colloid(r_cut=3.0, nlist = self.nl)
        coll.set_params(mode="no_shift")
        coll.set_params(mode="shift")
        coll.set_params(mode="xplor")
        self.assertRaises(RuntimeError, coll.set_params, mode="blah")

    # test default coefficients
    def test_default_coeff(self):
        coll = azplugins.pair.colloid(r_cut=3.0, nlist = self.nl)
        # (sigma, r_cut, and r_on are default)
        coll.pair_coeff.set('A', 'A', epsilon=144., style='slv-slv')
        coll.update_coeffs()

    # test max rcut
    def test_max_rcut(self):
        coll = azplugins.pair.colloid(r_cut=2.5, nlist = self.nl)
        coll.pair_coeff.set('A', 'A', sigma=1.0, epsilon=144.0, style='slv-slv')
        self.assertAlmostEqual(2.5, coll.get_max_rcut())
        coll.pair_coeff.set('A', 'A', r_cut = 2.0)
        self.assertAlmostEqual(2.0, coll.get_max_rcut())

    # test specific nlist subscription
    def test_nlist_subscribe(self):
        coll = azplugins.pair.colloid(r_cut=2.5, nlist = self.nl)

        coll.pair_coeff.set('A', 'A', sigma = 1.0, epsilon=144.0, style='slv-slv')
        self.nl.update_rcut()
        self.assertAlmostEqual(2.5, self.nl.r_cut.get_pair('A','A'))

        coll.pair_coeff.set('A', 'A', r_cut = 2.0)
        self.nl.update_rcut()
        self.assertAlmostEqual(2.0, self.nl.r_cut.get_pair('A','A'))

    # test coeff list
    def test_coeff_list(self):
        coll = azplugins.pair.colloid(r_cut=3.0, nlist = self.nl)
        coll.pair_coeff.set(['A', 'B'], ['A', 'C'], epsilon=144.0, sigma=1.0, style='slv-slv', r_cut=2.5, r_on=2.0)
        coll.update_coeffs()

    # test adding types
    def test_type_add(self):
        coll = azplugins.pair.colloid(r_cut=3.0, nlist = self.nl)
        coll.pair_coeff.set('A', 'A', epsilon=144.0, sigma=1.0, style='slv-slv')
        self.s.particles.types.add('B')
        self.assertRaises(RuntimeError, coll.update_coeffs)
        coll.pair_coeff.set('A', 'B', epsilon=144.0, sigma=1.0, style='slv-slv')
        coll.pair_coeff.set('B', 'B', epsilon=144.0, sigma=1.0, style='slv-slv')
        coll.update_coeffs()

    def tearDown(self):
        del self.s, self.nl
        context.initialize()

# test the validity of the pair potential
class potential_colloid_tests(unittest.TestCase):
    def setUp(self):
        snap = data.make_snapshot(N=2, box=data.boxdim(L=100),particle_types=['A'])
        if comm.get_rank() == 0:
            snap.particles.position[0] = (0,0,0)
            snap.particles.diameter[0] = 1.5

            snap.particles.position[1] = (3.0,0,0)
            snap.particles.diameter[1] = 3.0
        init.read_snapshot(snap)

        self.coll = azplugins.pair.colloid(r_cut=6.0, nlist = md.nlist.cell())

        md.integrate.mode_standard(dt=0)
        md.integrate.nve(group = group.all())

    # test the calculation of force and potential for solvent-solvent style
    def test_slv_slv_potential(self):
        # test no shifting
        self.coll.pair_coeff.set('A','A', epsilon=100.0, sigma=2.0, style='slv-slv')
        self.coll.set_params(mode="no_shift")

        run(1)
        f0 = self.coll.forces[0].force
        f1 = self.coll.forces[1].force
        e0 = self.coll.forces[0].energy
        e1 = self.coll.forces[1].energy

        # U obtained in separate python script for LJ potential, F is central differenced
        U = -0.222455968249
        F = 0.402093561279
        self.assertAlmostEqual(e0,0.5*U,3)
        self.assertAlmostEqual(e1,0.5*U,3)

        self.assertAlmostEqual(f0[0],F,3)
        self.assertAlmostEqual(f0[1],0)
        self.assertAlmostEqual(f0[2],0)

        self.assertAlmostEqual(f1[0],-F,3)
        self.assertAlmostEqual(f1[1],0)
        self.assertAlmostEqual(f1[2],0)

        # test with shifting enabled
        self.coll.set_params(mode="shift")
        run(1)
        U -= -0.00380516787794
        self.assertAlmostEqual(self.coll.forces[0].energy,0.5*U,3)
        self.assertAlmostEqual(self.coll.forces[1].energy,0.5*U,3)
        self.assertAlmostEqual(self.coll.forces[0].force[0],F,3)
        self.assertAlmostEqual(self.coll.forces[1].force[0],-F,3)

        # set epsilon to zero and make sure this is ignored
        self.coll.set_params(mode='no_shift')
        self.coll.pair_coeff.set('A','A', epsilon=0.0)
        run(1)
        self.assertAlmostEqual(self.coll.forces[0].energy,0)
        self.assertAlmostEqual(self.coll.forces[1].energy,0)
        self.assertAlmostEqual(self.coll.forces[0].force[0],0)
        self.assertAlmostEqual(self.coll.forces[1].force[0],0)

        # shrink cutoff and ensure this case is ignored
        self.coll.pair_coeff.set('A','A', epsilon=100., r_cut=2.0)
        run(1)
        self.assertAlmostEqual(self.coll.forces[0].energy,0)
        self.assertAlmostEqual(self.coll.forces[1].energy,0)
        self.assertAlmostEqual(self.coll.forces[0].force[0],0)
        self.assertAlmostEqual(self.coll.forces[1].force[0],0)

    # test the calculation of force and potential for colloid-solvent style
    def test_coll_slv_potential(self):
        # test no shifting
        self.coll.pair_coeff.set('A','A', epsilon=100.0, sigma=1.05, style='coll-slv')
        self.coll.set_params(mode="no_shift")

        run(1)
        f0 = self.coll.forces[0].force
        f1 = self.coll.forces[1].force
        e0 = self.coll.forces[0].energy
        e1 = self.coll.forces[1].energy

        # U is obtained in a separate python script, F is obtained by central differences
        U = -0.275765225801
        F = 0.710777032561
        self.assertAlmostEqual(e0,0.5*U,3)
        self.assertAlmostEqual(e1,0.5*U,3)

        self.assertAlmostEqual(f0[0],F,3)
        self.assertAlmostEqual(f0[1],0)
        self.assertAlmostEqual(f0[2],0)

        self.assertAlmostEqual(f1[0],-F,3)
        self.assertAlmostEqual(f1[1],0)
        self.assertAlmostEqual(f1[2],0)

        # test with shifting enabled
        self.coll.set_params(mode="shift")
        run(1)
        U -= -0.00225831446085
        self.assertAlmostEqual(self.coll.forces[0].energy,0.5*U,3)
        self.assertAlmostEqual(self.coll.forces[1].energy,0.5*U,3)
        self.assertAlmostEqual(self.coll.forces[0].force[0],F,3)
        self.assertAlmostEqual(self.coll.forces[1].force[0],-F,3)

        # set epsilon to zero and make sure this is ignored
        self.coll.set_params(mode='no_shift')
        self.coll.pair_coeff.set('A','A', epsilon=0.0)
        run(1)
        self.assertAlmostEqual(self.coll.forces[0].energy,0)
        self.assertAlmostEqual(self.coll.forces[1].energy,0)
        self.assertAlmostEqual(self.coll.forces[0].force[0],0)
        self.assertAlmostEqual(self.coll.forces[1].force[0],0)

        # shrink cutoff and ensure this case is ignored
        self.coll.pair_coeff.set('A','A', epsilon=100., r_cut=2.0)
        run(1)
        self.assertAlmostEqual(self.coll.forces[0].energy,0)
        self.assertAlmostEqual(self.coll.forces[1].energy,0)
        self.assertAlmostEqual(self.coll.forces[0].force[0],0)
        self.assertAlmostEqual(self.coll.forces[1].force[0],0)

    # test the calculation of force and potential for colloid-solvent style
    def test_coll_coll_potential(self):
        # test no shifting
        self.coll.pair_coeff.set('A','A', epsilon=100.0, sigma=1.05, style='coll-coll')
        self.coll.set_params(mode="no_shift")

        run(1)
        f0 = self.coll.forces[0].force
        f1 = self.coll.forces[1].force
        e0 = self.coll.forces[0].energy
        e1 = self.coll.forces[1].energy

        # U obtained in a separate python script, F computed by central differencing of U
        U = -1.0366943672424296
        F = 1.82673534348
        self.assertAlmostEqual(e0,0.5*U,3)
        self.assertAlmostEqual(e1,0.5*U,3)

        self.assertAlmostEqual(f0[0],F,3)
        self.assertAlmostEqual(f0[1],0)
        self.assertAlmostEqual(f0[2],0)

        self.assertAlmostEqual(f1[0],-F,3)
        self.assertAlmostEqual(f1[1],0)
        self.assertAlmostEqual(f1[2],0)

        # test with shifting enabled
        self.coll.set_params(mode="shift")
        run(1)
        U -= -0.00696278336528
        self.assertAlmostEqual(self.coll.forces[0].energy,0.5*U,3)
        self.assertAlmostEqual(self.coll.forces[1].energy,0.5*U,3)
        self.assertAlmostEqual(self.coll.forces[0].force[0],F,3)
        self.assertAlmostEqual(self.coll.forces[1].force[0],-F,3)

        # set epsilon to zero and make sure this is ignored
        self.coll.set_params(mode='no_shift')
        self.coll.pair_coeff.set('A','A', epsilon=0.0)
        run(1)
        self.assertAlmostEqual(self.coll.forces[0].energy,0)
        self.assertAlmostEqual(self.coll.forces[1].energy,0)
        self.assertAlmostEqual(self.coll.forces[0].force[0],0)
        self.assertAlmostEqual(self.coll.forces[1].force[0],0)

        # shrink cutoff and ensure this case is ignored
        self.coll.pair_coeff.set('A','A', epsilon=100., r_cut=2.0)
        run(1)
        self.assertAlmostEqual(self.coll.forces[0].energy,0)
        self.assertAlmostEqual(self.coll.forces[1].energy,0)
        self.assertAlmostEqual(self.coll.forces[0].force[0],0)
        self.assertAlmostEqual(self.coll.forces[1].force[0],0)

    def tearDown(self):
        del self.coll
        context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
