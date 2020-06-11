# Copyright (c) 2018-2020, Michael P. Howard
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

# azplugins.bond.double_well
class bond_double_well_tests(unittest.TestCase):
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
        double_well = azplugins.bond.double_well()
        double_well.bond_coeff.set('bond', V_max=1.0, a=1.0, b=0.7)
        double_well.update_coeffs()

    # test missing coefficients
    def test_set_missing_V_max(self):
        double_well = azplugins.bond.double_well()
        double_well.bond_coeff.set('bond',  a=3, b=1.5)
        self.assertRaises(RuntimeError, double_well.update_coeffs)

    def test_set_missing_a(self):
        double_well = azplugins.bond.double_well()
        double_well.bond_coeff.set('bond', V_max=1.0, b=1.5)
        self.assertRaises(RuntimeError, double_well.update_coeffs)

    def test_set_missing_b(self):
        double_well = azplugins.bond.double_well()
        double_well.bond_coeff.set('bond',  V_max=1.0, a=1.5)
        self.assertRaises(RuntimeError, double_well.update_coeffs)

    def tearDown(self):
        del self.s
        context.initialize()

# azplugins.bond.double_well
class potential_bond_double_well_tests(unittest.TestCase):
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
    def test_potential_minimum(self):

        double_well = azplugins.bond.double_well()
        # this should put the second particle in the first minimum of the potential
        double_well.bond_coeff.set('bond', a=3.0, b=0.5, V_max=1.0)

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = group.all())
        run(1)
        U = 0
        F = 0
        f0 = double_well.forces[0].force
        f1 = double_well.forces[1].force
        e0 = double_well.forces[0].energy
        e1 = double_well.forces[1].energy
        print(e1,e0)
        self.assertAlmostEqual(e0,0.5*U,3)
        self.assertAlmostEqual(e1,0.5*U,3)

        self.assertAlmostEqual(f0[0],F,3)
        self.assertAlmostEqual(f0[1],0)
        self.assertAlmostEqual(f0[2],0)

        self.assertAlmostEqual(f1[0],-F,3)
        self.assertAlmostEqual(f1[1],0)
        self.assertAlmostEqual(f1[2],0)

    # test the calculation of force and potential
    def test_potential_maximum(self):

        double_well = azplugins.bond.double_well()
        # this should put the second particle at the maxium of the potential
        double_well.bond_coeff.set('bond', a=2.0, b=2.0, V_max=5.0)

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = group.all())
        run(1)
        U = 5.0
        F = 0
        f0 = double_well.forces[0].force
        f1 = double_well.forces[1].force
        e0 = double_well.forces[0].energy
        e1 = double_well.forces[1].energy

        self.assertAlmostEqual(e0,0.5*U,3)
        self.assertAlmostEqual(e1,0.5*U,3)

        self.assertAlmostEqual(f0[0],F,3)
        self.assertAlmostEqual(f0[1],0)
        self.assertAlmostEqual(f0[2],0)

        self.assertAlmostEqual(f1[0],-F,3)
        self.assertAlmostEqual(f1[1],0)
        self.assertAlmostEqual(f1[2],0)

    # test the calculation of force and potential
    def test_potential_in_between(self):

        double_well = azplugins.bond.double_well()

        double_well.bond_coeff.set('bond', a=1.0, b=1.0, V_max=1.0)

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = group.all())
        run(1)
        U = 0.5625
        F = -1.5
        f0 = double_well.forces[0].force
        f1 = double_well.forces[1].force
        e0 = double_well.forces[0].energy
        e1 = double_well.forces[1].energy

        self.assertAlmostEqual(e0,0.5*U,3)
        self.assertAlmostEqual(e1,0.5*U,3)

        self.assertAlmostEqual(f0[0],F,3)
        self.assertAlmostEqual(f0[1],0)
        self.assertAlmostEqual(f0[2],0)

        self.assertAlmostEqual(f1[0],-F,3)
        self.assertAlmostEqual(f1[1],0)
        self.assertAlmostEqual(f1[2],0)

    def tearDown(self):
        del self.s
        context.initialize()



if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
