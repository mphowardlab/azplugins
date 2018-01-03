# Copyright (c) 2016-2018, Panagiotopoulos Group, Princeton University
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

# azplugins.bond.fene
class bond_fene_tests(unittest.TestCase):
    def setUp(self):
        snap = data.make_snapshot(N=2, box=data.boxdim(L=20), particle_types=['A'], bond_types=['bond'])

        if comm.get_rank() == 0:
            snap.bonds.resize(1)
            snap.bonds.group[0] = [0,1]

        self.s = init.read_snapshot(snap)
        self.nl = md.nlist.cell()
        context.current.sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        fene = azplugins.bond.fene()
        fene.bond_coeff.set('bond', epsilon=1.0, sigma=1.0, k=30, r0=1.5)
        fene.update_coeffs()

    # test missing coefficients
    def test_set_missing_epsilon(self):
        fene = azplugins.bond.fene()
        fene.bond_coeff.set('bond', sigma=1.0, k=30, r0=1.5)
        self.assertRaises(RuntimeError, fene.update_coeffs)

    def test_set_missing_sigma(self):
        fene = azplugins.bond.fene()
        fene.bond_coeff.set('bond', epsilon=1.0, k=30, r0=1.5)
        self.assertRaises(RuntimeError, fene.update_coeffs)

    def test_set_missing_k(self):
        fene = azplugins.bond.fene()
        fene.bond_coeff.set('bond', epsilon=1.0, sigma=1.0, r0=1.5)
        self.assertRaises(RuntimeError, fene.update_coeffs)

    def test_set_missing_r0(self):
        fene = azplugins.bond.fene()
        fene.bond_coeff.set('bond', epsilon=1.0, sigma=1.0, k=30)
        self.assertRaises(RuntimeError, fene.update_coeffs)

     # test coefficients = 0
    def test_set_zero_epsilon(self):
        fene = azplugins.bond.fene()
        fene.bond_coeff.set('bond', epsilon=0, sigma=1.0, k=30,r0=1.5)
        self.assertRaises(ValueError, fene.update_coeffs)

    def test_set_zero_sigma(self):
        fene = azplugins.bond.fene()
        fene.bond_coeff.set('bond', epsilon=1.0, sigma=0.0, k=30, r0=1.5)
        self.assertRaises(ValueError, fene.update_coeffs)

    def test_set_zero_k(self):
        fene = azplugins.bond.fene()
        fene.bond_coeff.set('bond', epsilon=1.0, sigma=1.0, k=0, r0=1.5)
        self.assertRaises(ValueError, fene.update_coeffs)

    def test_set_zero_r0(self):
        fene = azplugins.bond.fene()
        fene.bond_coeff.set('bond', epsilon=1.0, sigma=1.0, k=30, r0=0)
        self.assertRaises(ValueError, fene.update_coeffs)

    def tearDown(self):
        del self.s
        context.initialize()

class potential_bond_fene_tests(unittest.TestCase):
    def setUp(self):
        snap = data.make_snapshot(N=2, box=data.boxdim(L=20), particle_types=['A'],bond_types = ['bond'])

        if comm.get_rank() == 0:
            snap.bonds.resize(1)
            snap.bonds.group[0] = [0,1]
            snap.particles.position[0] = (0,0,0)
            snap.particles.position[1] = (1,0,0)

        self.s = init.read_snapshot(snap)
        self.nl = md.nlist.cell()
        context.current.sorter.set_params(grid=8)

    # test the calculation of force and potential
    def test_potential(self):
        #fene = azplugins.bond.fene()
        fene = md.bond.fene()
        fene.bond_coeff.set('bond', epsilon=1.0, sigma=1.0, k=30,r0=1.5)

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = group.all())
        run(1)
        U = 20.8377999404
        F = 29.999996185302734
        f0 = fene.forces[0].force
        f1 = fene.forces[1].force
        e0 = fene.forces[0].energy
        e1 = fene.forces[1].energy

        self.assertAlmostEqual(e0,0.5*U,3)
        self.assertAlmostEqual(e1,0.5*U,3)

        self.assertAlmostEqual(f0[0],F,3)
        self.assertAlmostEqual(f0[1],0)
        self.assertAlmostEqual(f0[2],0)

        self.assertAlmostEqual(f1[0],-F,3)
        self.assertAlmostEqual(f1[1],0)
        self.assertAlmostEqual(f1[2],0)

    # test streching beyond maximal bond length of r0
    def test_potential_strech_beyond_r0(self):
        fene = azplugins.bond.fene()
        fene.bond_coeff.set('bond', epsilon=1.0, sigma=1.0, k=30,r0=0.9)
        fene.update_coeffs()
        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = group.all())
        if comm.get_num_ranks() == 1:
            self.assertRaises(RuntimeError, run, 1)

    def tearDown(self):
        del self.s
        context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
