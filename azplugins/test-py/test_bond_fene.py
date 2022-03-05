# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2022, Auburn University
# This file is part of the azplugins project, released under the Modified BSD License.

import hoomd
from hoomd import md
hoomd.context.initialize()
try:
    from hoomd import azplugins
except ImportError:
    import azplugins
import unittest

# azplugins.bond.fene
class bond_fene_tests(unittest.TestCase):
    def setUp(self):
        snap = hoomd.data.make_snapshot(N=2, box=hoomd.data.boxdim(L=20), particle_types=['A'], bond_types=['bond'])

        if hoomd.comm.get_rank() == 0:
            snap.bonds.resize(1)
            snap.bonds.group[0] = [0,1]

        self.s = hoomd.init.read_snapshot(snap)
        self.nl = md.nlist.cell()
        hoomd.context.current.sorter.set_params(grid=8)

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
    def test_set_zero_r0(self):
        fene = azplugins.bond.fene()
        fene.bond_coeff.set('bond', epsilon=1.0, sigma=1.0, k=30, r0=0)
        self.assertRaises(ValueError, fene.update_coeffs)

    def tearDown(self):
        del self.s
        hoomd.context.initialize()

class potential_bond_fene_tests(unittest.TestCase):
    def setUp(self):
        snap = hoomd.data.make_snapshot(N=2, box=hoomd.data.boxdim(L=20), particle_types=['A'],bond_types = ['bond'])

        if hoomd.comm.get_rank() == 0:
            snap.bonds.resize(1)
            snap.bonds.group[0] = [0,1]
            snap.particles.position[0] = (0,0,0)
            snap.particles.position[1] = (1,0,0)

        self.s = hoomd.init.read_snapshot(snap)
        self.nl = md.nlist.cell()
        hoomd.context.current.sorter.set_params(grid=8)

    # test the calculation of force and potential
    def test_potential(self):
        fene = azplugins.bond.fene()
        fene.bond_coeff.set('bond', epsilon=1.0, sigma=1.0, k=30,r0=1.5)

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = hoomd.group.all())
        hoomd.run(1)
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

    # test the calculation of force and potential when delta is non-zero
    def test_potential_delta_nonzero(self):
        fene = azplugins.bond.fene()
        fene.bond_coeff.set('bond', epsilon=1.0, sigma=1.0, k=30,r0=1.5,delta=1.8)

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = hoomd.group.all())
        hoomd.run(1)
        #values of F and  U are caluclated using a calculator, by substituting
        #r0=1.5,delta=1.8,sigma=1.0,epsilon=1.0, with r=1.0
        F = 57.5403726708
        U = 12.2959912562
        f0 = fene.forces[0].force
        f1 = fene.forces[1].force
        e0 = fene.forces[0].energy
        e1 = fene.forces[1].energy

        self.assertAlmostEqual(e0,0.5*U,3)
        self.assertAlmostEqual(e1,0.5*U,3)

        self.assertAlmostEqual(f0[0],-F,3)
        self.assertAlmostEqual(f0[1],0)
        self.assertAlmostEqual(f0[2],0)

        self.assertAlmostEqual(f1[0],F,3)
        self.assertAlmostEqual(f1[1],0)
        self.assertAlmostEqual(f1[2],0)

    # test the calculation of foce and potential when sigma=0 and epsilon=0
    def test_potential_sigma_zero_epsilon_zero(self):
        fene = azplugins.bond.fene()
        fene.bond_coeff.set('bond', epsilon=0.0, sigma=0.0, k=30,r0=1.5,delta=1.8)

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = hoomd.group.all())
        hoomd.run(1)
        #values of F and  U are caluclated using a calculator, by substituting
        #r0=1.5,delta=1.8,sigma=0.0,epsilon=0.0, with r=1.0
        F = 33.540372671
        U = 11.29599912562
        f0 = fene.forces[0].force
        f1 = fene.forces[1].force
        e0 = fene.forces[0].energy
        e1 = fene.forces[1].energy

        self.assertAlmostEqual(e0,0.5*U,3)
        self.assertAlmostEqual(e1,0.5*U,3)

        self.assertAlmostEqual(f0[0],-F,3)
        self.assertAlmostEqual(f0[1],0)
        self.assertAlmostEqual(f0[2],0)

        self.assertAlmostEqual(f1[0],F,3)
        self.assertAlmostEqual(f1[1],0)
        self.assertAlmostEqual(f1[2],0)

    # test the calculation of foce and potential when sigma=nonzero and epsilon=0
    def test_potential_sigma_nonzero_epsilon_zero(self):
        fene = azplugins.bond.fene()
        fene.bond_coeff.set('bond', epsilon=0.0, sigma=1.0, k=30,r0=1.5,delta=1.8)

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = hoomd.group.all())
        hoomd.run(1)
        #values of F and  U are caluclated using a calculator, by substituting
        #r0=1.5,delta=1.8,sigma=1.0,epsilon=0.0, with r=1.0
        F = 33.540372671
        U = 11.29599912562
        f0 = fene.forces[0].force
        f1 = fene.forces[1].force
        e0 = fene.forces[0].energy
        e1 = fene.forces[1].energy

        self.assertAlmostEqual(e0,0.5*U,3)
        self.assertAlmostEqual(e1,0.5*U,3)

        self.assertAlmostEqual(f0[0],-F,3)
        self.assertAlmostEqual(f0[1],0)
        self.assertAlmostEqual(f0[2],0)

        self.assertAlmostEqual(f1[0],F,3)
        self.assertAlmostEqual(f1[1],0)
        self.assertAlmostEqual(f1[2],0)

    # test the calculation of foce and potential when sigma=0 and epsilon=nonzero
    def test_potential_sigma_zero_epsilon_nonzero(self):
        fene = azplugins.bond.fene()
        fene.bond_coeff.set('bond', epsilon=1.0, sigma=0.0, k=30,r0=1.5,delta=1.8)

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = hoomd.group.all())
        hoomd.run(1)
        #values of F and  U are caluclated using a calculator, by substituting
        #r0=1.5,delta=1.8,sigma=0.0,epsilon=1.0, with r=1.0
        F = 33.540372671
        U = 11.29599912562 #no contribution from WCA as 2^1/6 sigma < r
        f0 = fene.forces[0].force
        f1 = fene.forces[1].force
        e0 = fene.forces[0].energy
        e1 = fene.forces[1].energy

        self.assertAlmostEqual(e0,0.5*U,3)
        self.assertAlmostEqual(e1,0.5*U,3)

        self.assertAlmostEqual(f0[0],-F,3)
        self.assertAlmostEqual(f0[1],0)
        self.assertAlmostEqual(f0[2],0)

        self.assertAlmostEqual(f1[0],F,3)
        self.assertAlmostEqual(f1[1],0)
        self.assertAlmostEqual(f1[2],0)

    # test the calculation of foce and potential when k=0 and epsilon=nonzero sigma=nonzero
    def test_potential_k_zero_epsilon_nonzero_sigma_nonzero(self):
        fene = azplugins.bond.fene()
        fene.bond_coeff.set('bond', epsilon=1.0, sigma=1.0, k=0,r0=1.5,delta=1.8)

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = hoomd.group.all())
        hoomd.run(1)
        #values of F and  U are caluclated using a calculator, by substituting
        #k=0,r0=1.5,delta=1.8,sigma=1.0,epsilon=1.0, with r=1.0
        F = 24
        U = 1               #no contribution from FENE bonds as k=0
        f0 = fene.forces[0].force
        f1 = fene.forces[1].force
        e0 = fene.forces[0].energy
        e1 = fene.forces[1].energy

        self.assertAlmostEqual(e0,0.5*U,3)
        self.assertAlmostEqual(e1,0.5*U,3)

        self.assertAlmostEqual(f0[0],-F,3)
        self.assertAlmostEqual(f0[1],0)
        self.assertAlmostEqual(f0[2],0)

        self.assertAlmostEqual(f1[0],F,3)
        self.assertAlmostEqual(f1[1],0)
        self.assertAlmostEqual(f1[2],0)

    # test the calculation of foce and potential when k=0 and epsilon=zero sigma=nonzero
    def test_potential_k_zero_epsilon_zero_sigma_nonzero(self):
        fene = azplugins.bond.fene()
        fene.bond_coeff.set('bond', epsilon=0.0, sigma=1.0, k=0,r0=1.5,delta=1.8)

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = hoomd.group.all())
        hoomd.run(1)
        #values of F and  U are caluclated using a calculator, by substituting
        #r0=1.5,delta=1.8,sigma=1.0,epsilon=0.0, with r=1.0
        F = 0
        U = 0
        f0 = fene.forces[0].force
        f1 = fene.forces[1].force
        e0 = fene.forces[0].energy
        e1 = fene.forces[1].energy

        self.assertAlmostEqual(e0,0.5*U,3)
        self.assertAlmostEqual(e1,0.5*U,3)

        self.assertAlmostEqual(f0[0],-F,3)
        self.assertAlmostEqual(f0[1],0)
        self.assertAlmostEqual(f0[2],0)

        self.assertAlmostEqual(f1[0],F,3)
        self.assertAlmostEqual(f1[1],0)
        self.assertAlmostEqual(f1[2],0)

    # test the calculation of foce and potential when k=0 and epsilon=nonzero sigma=zero
    def test_potential_k_zero_epsilon_nonzero_sigma_zero(self):
        fene = azplugins.bond.fene()
        fene.bond_coeff.set('bond', epsilon=1.0, sigma=0.0, k=0,r0=1.5,delta=1.8)

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = hoomd.group.all())
        hoomd.run(1)
        #values of F and  U are caluclated using a calculator, by substituting
        #k=0,r0=1.5,delta=1.8,sigma=0.0,epsilon=1.0, with r=1.0
        F = 0
        U = 0
        f0 = fene.forces[0].force
        f1 = fene.forces[1].force
        e0 = fene.forces[0].energy
        e1 = fene.forces[1].energy

        self.assertAlmostEqual(e0,0.5*U,3)
        self.assertAlmostEqual(e1,0.5*U,3)

        self.assertAlmostEqual(f0[0],-F,3)
        self.assertAlmostEqual(f0[1],0)
        self.assertAlmostEqual(f0[2],0)

        self.assertAlmostEqual(f1[0],F,3)
        self.assertAlmostEqual(f1[1],0)
        self.assertAlmostEqual(f1[2],0)

    # test the calculation of foce and potential when k=0 and epsilon=zero sigma=zero
    def test_potential_k_zero_epsilon_nonzero_sigma_zero(self):
        fene = azplugins.bond.fene()
        fene.bond_coeff.set('bond', epsilon=0.0, sigma=0.0, k=0,r0=1.5,delta=1.8)

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = hoomd.group.all())
        hoomd.run(1)
        #values of F and  U are caluclated using a calculator, by substituting
        #k=0,r0=1.5,delta=1.8,sigma=0.0,epsilon=0.0, with r=1.0
        F = 0
        U = 0
        f0 = fene.forces[0].force
        f1 = fene.forces[1].force
        e0 = fene.forces[0].energy
        e1 = fene.forces[1].energy

        self.assertAlmostEqual(e0,0.5*U,3)
        self.assertAlmostEqual(e1,0.5*U,3)

        self.assertAlmostEqual(f0[0],-F,3)
        self.assertAlmostEqual(f0[1],0)
        self.assertAlmostEqual(f0[2],0)

        self.assertAlmostEqual(f1[0],F,3)
        self.assertAlmostEqual(f1[1],0)
        self.assertAlmostEqual(f1[2],0)

    # test streching beyond maximal bond length of r0
    def test_potential_strech_beyond_r0(self):
        fene = azplugins.bond.fene()
        fene.bond_coeff.set('bond', epsilon=1.0, sigma=1.0, k=30,r0=0.9)
        fene.update_coeffs()
        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = hoomd.group.all())
        if hoomd.comm.get_num_ranks() == 1:
            self.assertRaises(RuntimeError, hoomd.run, 1)

    def tearDown(self):
        del self.s
        hoomd.context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
