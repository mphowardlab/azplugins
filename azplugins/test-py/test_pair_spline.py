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

# azplugins.pair.spline
class pair_spline_tests(unittest.TestCase):
    def setUp(self):
        # raw snapshot is fine, just needs to have the types
        snap = data.make_snapshot(N=100, box=data.boxdim(L=20), particle_types=['A'])
        self.s = init.read_snapshot(snap)
        self.nl = md.nlist.cell()
        context.current.sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        spline = azplugins.pair.spline(r_cut=3.0, nlist = self.nl)
        spline.pair_coeff.set('A', 'A', amp=1.0, m=2.0, r_start=1.5)
        spline.update_coeffs()

    # test missing ron coefficient
    def test_set_missing_ron(self):
        spline = azplugins.pair.spline(r_cut=3.0, nlist = self.nl)
        spline.pair_coeff.set('A', 'A', amp=1.0, m=2.0)
        self.assertRaises(RuntimeError, spline.update_coeffs)

    # test missing a coefficient
    def test_set_missing_amp(self):
        spline = azplugins.pair.spline(r_cut=3.0, nlist = self.nl)
        spline.pair_coeff.set('A', 'A', m=3.0, r_start=1.0)
        self.assertRaises(RuntimeError, spline.update_coeffs)

    # test missing delta coefficient
    def test_set_missing_m(self):
        spline = azplugins.pair.spline(r_cut=3.0, nlist = self.nl)
        spline.pair_coeff.set('A', 'A', amp=0.3, r_start=1.0)
        self.assertRaises(RuntimeError, spline.update_coeffs)

    # test missing type coefficients
    def test_missing_AA(self):
        spline = azplugins.pair.spline(r_cut=3.0, nlist = self.nl)
        self.assertRaises(RuntimeError, spline.update_coeffs)

    # test max rcut
    def test_max_rcut(self):
        spline = azplugins.pair.spline(r_cut=2.5, nlist = self.nl)
        spline.pair_coeff.set('A', 'A', amp=1.0, m=3.0, r_start=1.0)
        self.assertAlmostEqual(2.5, spline.get_max_rcut())
        spline.pair_coeff.set('A', 'A', r_cut = 2.0)
        self.assertAlmostEqual(2.0, spline.get_max_rcut())

    # test specific nlist subscription
    def test_nlist_subscribe(self):
        spline = azplugins.pair.spline(r_cut=2.5, nlist = self.nl)

        spline.pair_coeff.set('A', 'A', amp=1.0, m=2.5, r_start=1.0)
        self.nl.update_rcut()
        self.assertAlmostEqual(2.5, self.nl.r_cut.get_pair('A','A'))

        spline.pair_coeff.set('A', 'A', r_cut = 2.0)
        self.nl.update_rcut()
        self.assertAlmostEqual(2.0, self.nl.r_cut.get_pair('A','A'))

    # test coeff list
    def test_coeff_list(self):
        spline = azplugins.pair.spline(r_cut=3.0, nlist = self.nl)
        spline.pair_coeff.set(['A', 'B'], ['A', 'C'], amp=1.0, m=3.0, r_cut=2.5, r_start=2.0)
        spline.update_coeffs()

    # test adding types
    def test_type_add(self):
        spline = azplugins.pair.spline(r_cut=3.0, nlist = self.nl)
        spline.pair_coeff.set('A', 'A', amp=1.0, m=3.0, r_start=1.0)
        self.s.particles.types.add('B')
        self.assertRaises(RuntimeError, spline.update_coeffs)
        spline.pair_coeff.set('A', 'B', amp=1.0, m=3.0, r_start=1.0)
        spline.pair_coeff.set('B', 'B', amp=1.0, m=3.0, r_start=1.0)
        spline.update_coeffs()

    def tearDown(self):
        del self.s, self.nl
        context.initialize()

# test the validity of the pair potential
class potential_spline_tests(unittest.TestCase):
    def setUp(self):
        snap = data.make_snapshot(N=2, box=data.boxdim(L=20),particle_types=['A'])
        if comm.get_rank() == 0:
            snap.particles.position[0] = (0,0,0)
            snap.particles.position[1] = (2.0,0,0)
        init.read_snapshot(snap)
        self.nl = md.nlist.cell()

    # test the calculation of force and potential
    def test_potential(self):
        spline = azplugins.pair.spline(r_cut=3.0, nlist = self.nl)
        spline.pair_coeff.set('A','A', m=2.0, amp=0.3, r_start=1.0)

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = group.all())
        run(1)
        U = 0.205005
        F = 0.211059
        f0 = spline.forces[0].force
        f1 = spline.forces[1].force
        e0 = spline.forces[0].energy
        e1 = spline.forces[1].energy

        self.assertAlmostEqual(e0,0.5*U,3)
        self.assertAlmostEqual(e1,0.5*U,3)

        self.assertAlmostEqual(f0[0],-F,3)
        self.assertAlmostEqual(f0[1],0)
        self.assertAlmostEqual(f0[2],0)

        self.assertAlmostEqual(f1[0],F,3)
        self.assertAlmostEqual(f1[1],0)
        self.assertAlmostEqual(f1[2],0)

    def test_potential_zero(self):
        spline = azplugins.pair.spline(r_cut=1.5, nlist = self.nl)
        spline.pair_coeff.set('A','A', m=2.0, amp=0.3, r_start=1.0)

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = group.all())
        run(1)
        U = 0.0
        F = 0.0
        f0 = spline.forces[0].force
        f1 = spline.forces[1].force
        e0 = spline.forces[0].energy
        e1 = spline.forces[1].energy

        self.assertAlmostEqual(e0,0.5*U,3)
        self.assertAlmostEqual(e1,0.5*U,3)
        self.assertAlmostEqual(f0[0],-F,3)
        self.assertAlmostEqual(f1[0],F,3)

    def test_potential_plateau(self):
        spline = azplugins.pair.spline(r_cut=3.0, nlist = self.nl)
        spline.pair_coeff.set('A','A', m=2.0, amp=-0.389, r_start=2.5)

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = group.all())
        run(1)
        U = -0.389
        F = 0.0
        f0 = spline.forces[0].force
        f1 = spline.forces[1].force
        e0 = spline.forces[0].energy
        e1 = spline.forces[1].energy

        self.assertAlmostEqual(e0,0.5*U,3)
        self.assertAlmostEqual(e1,0.5*U,3)
        self.assertAlmostEqual(f0[0],-F,3)
        self.assertAlmostEqual(f1[0],F,3)

    def tearDown(self):
        del self.nl
        context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
