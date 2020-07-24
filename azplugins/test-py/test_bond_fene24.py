# Copyright (c) 2018-2020, Michael P. Howard
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: astatt

import hoomd
from hoomd import md
hoomd.context.initialize()
try:
    from hoomd import azplugins
except ImportError:
    import azplugins
import unittest

# azplugins.bond.fene24
class bond_fene24_tests(unittest.TestCase):
    def setUp(self):
        snap = hoomd.data.make_snapshot(N=2,
                                  box=hoomd.data.boxdim(L=100),
                                  particle_types = ['A','B'],
                                  bond_types = ['bond_1'])

        if hoomd.comm.get_rank() == 0:
            snap.bonds.resize(1)
            snap.particles.position[0] = (0,0,0)
            snap.particles.position[1] = (0,0,1.5)
            snap.bonds.group[0] = [0,1]
            snap.bonds.typeid[0] = 0

        self.s = hoomd.init.read_snapshot(snap)

    # basic test of creation
    def test(self):
        fene24 = azplugins.bond.fene24()
        fene24.bond_coeff.set('bond_1', epsilon=1.0, sigma=1.0, lam=1.0, k=30,r0=1.5)
        fene24.update_coeffs()

    # test missing coefficients
    def test_set_missing_epsilon(self):
        fene24 = azplugins.bond.fene24()
        fene24.bond_coeff.set('bond_1', sigma=1.0, lam=1.0, k=30,r0=1.5)
        self.assertRaises(RuntimeError, fene24.update_coeffs)

    def test_set_missing_sigma(self):
        fene24 = azplugins.bond.fene24()
        fene24.bond_coeff.set('bond_1', epsilon=1.0, lam=1.0, k=30,r0=1.5)
        self.assertRaises(RuntimeError, fene24.update_coeffs)

    def test_set_missing_lambda(self):
        fene24 = azplugins.bond.fene24()
        fene24.bond_coeff.set('bond_1', epsilon=1.0,sigma=1.0, k=30,r0=1.5)
        self.assertRaises(RuntimeError, fene24.update_coeffs)

    def test_set_missing_k(self):
        fene24 = azplugins.bond.fene24()
        fene24.bond_coeff.set('bond_1', epsilon=1.0,sigma=1.0, lam=1,r0=1.5)
        self.assertRaises(RuntimeError, fene24.update_coeffs)

    def test_set_missing_r0(self):
        fene24 = azplugins.bond.fene24()
        fene24.bond_coeff.set('bond_1', epsilon=1.0,sigma=1.0, lam=1,k=30)
        self.assertRaises(RuntimeError, fene24.update_coeffs)

     # test coefficients = 0
    def test_set_zero_epsilon(self):
        fene24 = azplugins.bond.fene24()
        fene24.bond_coeff.set('bond_1', epsilon=0, sigma=1.0, lam=1.0, k=30,r0=1.5)
        self.assertRaises(ValueError, fene24.update_coeffs)

    def test_set_zero_sigma(self):
        fene24 = azplugins.bond.fene24()
        fene24.bond_coeff.set('bond_1', epsilon=1.0, sigma=0.0, lam=1.0, k=30,r0=1.5)
        self.assertRaises(ValueError, fene24.update_coeffs)

    def test_set_zero_k(self):
        fene24 = azplugins.bond.fene24()
        fene24.bond_coeff.set('bond_1', epsilon=1.0, sigma=1.0, lam=1.0, k=0,r0=1.5)
        self.assertRaises(ValueError, fene24.update_coeffs)

    def test_set_zero_r0(self):
        fene24 = azplugins.bond.fene24()
        fene24.bond_coeff.set('bond_1', epsilon=1.0, sigma=1.0, lam=1.0, k=30,r0=0)
        self.assertRaises(ValueError, fene24.update_coeffs)

    def tearDown(self):
        del self.s
        hoomd.context.initialize()

# test the validity of the bond potential
class potential_bond_fene24_tests(unittest.TestCase):
    def setUp(self):
        snap = hoomd.data.make_snapshot(N=2,
                                  box=hoomd.data.boxdim(L=100),
                                  particle_types = ['A','B'],
                                  bond_types = ['bond_1'])

        if hoomd.comm.get_rank() == 0:
            snap.bonds.resize(1)
            snap.particles.position[0] = (0,0,0)
            snap.particles.position[1] = (0.0,0,1.0)
            snap.particles.typeid[0] = 0
            snap.particles.typeid[1] = 1
            snap.bonds.group[0] = [0,1]
            snap.bonds.typeid[0] = 0
        self.s = hoomd.init.read_snapshot(snap)


    # test the calculation of force and potential at minimum
    def test_potential_value_min(self):
        fene24 = azplugins.bond.fene24()
        fene24.bond_coeff.set('bond_1', epsilon=1.0, sigma=1.0, lam=1.0, k=30,r0=1.5)
        fene24.update_coeffs()

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = hoomd.group.all())
        hoomd.run(1)
        U = 19.8377999404
        F = -42.0
        f0 = fene24.forces[0].force
        f1 = fene24.forces[1].force
        e0 = fene24.forces[0].energy
        e1 = fene24.forces[1].energy

        self.assertAlmostEqual(e0,0.5*U,3)
        self.assertAlmostEqual(e1,0.5*U,3)

        self.assertAlmostEqual(f0[0],0)
        self.assertAlmostEqual(f0[1],0)
        self.assertAlmostEqual(f0[2],F,3)

        self.assertAlmostEqual(f1[0],0)
        self.assertAlmostEqual(f1[1],0)
        self.assertAlmostEqual(f1[2],-F,3)


    # test potential in the repulsive region
    def test_potential_value_rep(self):
        fene24 = azplugins.bond.fene24()
        fene24.bond_coeff.set('bond_1', epsilon=1.0, sigma=1.01, lam=1.0, k=30,r0=1.5)

        fene24.update_coeffs()

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = hoomd.group.all())
        hoomd.run(1)

        U = 21.207765657
        F = -133.652880656

        f0 = fene24.forces[0].force
        f1 = fene24.forces[1].force
        e0 = fene24.forces[0].energy
        e1 = fene24.forces[1].energy

        self.assertAlmostEqual(e0,0.5*U,3)
        self.assertAlmostEqual(e1,0.5*U,3)

        self.assertAlmostEqual(f0[0],0)
        self.assertAlmostEqual(f0[1],0)
        self.assertAlmostEqual(f0[2],F,3)

        self.assertAlmostEqual(f1[0],0)
        self.assertAlmostEqual(f1[1],0)
        self.assertAlmostEqual(f1[2],-F,3)


    # test streching beyond maximal bond length of r0
    @unittest.skipIf(hoomd.comm.get_num_ranks() > 1, 'in mpi throwing an error does not end the simulation correctly')
    def test_potential_strech_beyond_r0(self):
        fene24 = azplugins.bond.fene24()
        fene24.bond_coeff.set('bond_1', epsilon=1.0, sigma=1.0, lam=1.0, k=30,r0=0.9)
        fene24.update_coeffs()
        md.integrate.mode_standard(dt=0.01)
        nve = md.integrate.nve(group = hoomd.group.all())
        self.assertRaises(RuntimeError,hoomd.run,1)

    def tearDown(self):
        del self.s
        hoomd.context.initialize()


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
