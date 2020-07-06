# Copyright (c) 2018-2020, Michael P. Howard
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: arjunsg2

import hoomd
from hoomd import md
hoomd.context.initialize()
try:
    from hoomd import azplugins
except ImportError:
    import azplugins
import unittest

# azplugins.pair.hertz
class pair_hertz_tests(unittest.TestCase):
    def setUp(self):
        # raw snapshot is fine, just needs to have the types
        snap = hoomd.data.make_snapshot(N=100, box=hoomd.data.boxdim(L=20), particle_types=['A'])
        self.s = hoomd.init.read_snapshot(snap)
        self.nl = md.nlist.cell()
        hoomd.context.current.sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        hertz = azplugins.pair.hertz(r_cut=3.0, nlist = self.nl)
        hertz.pair_coeff.set('A', 'A', epsilon=1.0, r_cut=2.5, r_on=2.0)
        hertz.update_coeffs()

    # test missing coefficients
    def test_set_missing_epsilon(self):
        hertz = azplugins.pair.hertz(r_cut=3.0, nlist = self.nl)
        hertz.pair_coeff.set('A', 'A')
        self.assertRaises(RuntimeError, hertz.update_coeffs)

    # test missing pair
    def test_missing_AA(self):
        hertz = azplugins.pair.hertz(r_cut=3.0, nlist = self.nl)
        self.assertRaises(RuntimeError, hertz.update_coeffs)

    # test set params
    def test_set_params(self):
        hertz = azplugins.pair.hertz(r_cut=3.0, nlist = self.nl)
        hertz.set_params(mode="no_shift")
        hertz.set_params(mode="shift")
        hertz.set_params(mode="xplor")
        self.assertRaises(RuntimeError, hertz.set_params, mode="blah")

    # test default coefficients
    def test_default_coeff(self):
        hertz = azplugins.pair.hertz(r_cut=3.0, nlist = self.nl)
        # (r_cut, and r_on are default)
        hertz.pair_coeff.set('A', 'A', epsilon=1.0)
        hertz.update_coeffs()

    # test max rcut
    def test_max_rcut(self):
        hertz = azplugins.pair.hertz(r_cut=2.5, nlist = self.nl)
        hertz.pair_coeff.set('A', 'A', epsilon=1.0)
        self.assertAlmostEqual(2.5, hertz.get_max_rcut())
        hertz.pair_coeff.set('A', 'A', r_cut = 2.0)
        self.assertAlmostEqual(2.0, hertz.get_max_rcut())

    # test specific nlist subscription
    def test_nlist_subscribe(self):
        hertz = azplugins.pair.hertz(r_cut=2.5, nlist = self.nl)

        hertz.pair_coeff.set('A', 'A', epsilon=1.0)
        self.nl.update_rcut()
        self.assertAlmostEqual(2.5, self.nl.r_cut.get_pair('A','A'))

        hertz.pair_coeff.set('A', 'A', r_cut = 2.0)
        self.nl.update_rcut()
        self.assertAlmostEqual(2.0, self.nl.r_cut.get_pair('A','A'))

    # test coeff list
    def test_coeff_list(self):
        hertz = azplugins.pair.hertz(r_cut=3.0, nlist = self.nl)
        hertz.pair_coeff.set(['A', 'B'], ['A', 'C'], epsilon=1.0, r_cut=2.5, r_on=2.0)
        hertz.update_coeffs()

    # test adding types
    def test_type_add(self):
        hertz = azplugins.pair.hertz(r_cut=3.0, nlist = self.nl)
        hertz.pair_coeff.set('A', 'A', epsilon=1.0)
        self.s.particles.types.add('B')
        self.assertRaises(RuntimeError, hertz.update_coeffs)
        hertz.pair_coeff.set('A', 'B', epsilon=1.0)
        hertz.pair_coeff.set('B', 'B', epsilon=1.0)
        hertz.update_coeffs()

    def tearDown(self):
        del self.s, self.nl
        hoomd.context.initialize()

# test the validity of the pair potential
class potential_hertz_tests(unittest.TestCase):
    def setUp(self):
        snap = hoomd.data.make_snapshot(N=2, box=hoomd.data.boxdim(L=20),particle_types=['A'])
        if comm.get_rank() == 0:
            snap.particles.position[0] = (0,0,0)
            snap.particles.position[1] = (1.05,0,0)
        hoomd.init.read_snapshot(snap)
        self.nl = md.nlist.cell()

    # test the calculation of force and potential
    def test_potential(self):
        hertz = azplugins.pair.hertz(r_cut=1.5, nlist = self.nl)
        hertz.pair_coeff.set('A','A', epsilon=2.0)
        hertz.set_params(mode="no_shift")

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = group.all())
        run(1)
        U = 0.09859
        F = 0.54772
        f0 = hertz.forces[0].force
        f1 = hertz.forces[1].force
        e0 = hertz.forces[0].energy
        e1 = hertz.forces[1].energy

        self.assertAlmostEqual(e0,0.5*U,3)
        self.assertAlmostEqual(e1,0.5*U,3)

        self.assertAlmostEqual(f0[0],-F,3)
        self.assertAlmostEqual(f0[1],0)
        self.assertAlmostEqual(f0[2],0)

        self.assertAlmostEqual(f1[0],F,3)
        self.assertAlmostEqual(f1[1],0)
        self.assertAlmostEqual(f1[2],0)

        hertz = azplugins.pair.hertz(r_cut=2.05, nlist = self.nl)
        hertz.pair_coeff.set('A','A', epsilon=3.0)
        hertz.set_params(mode='shift')
        run(1)
        U = 0.498582
        F = 1.246455
        self.assertAlmostEqual(hertz.forces[0].energy, 0.5*U, 3)
        self.assertAlmostEqual(hertz.forces[1].energy, 0.5*U, 3)
        self.assertAlmostEqual(hertz.forces[0].force[0], -F, 3)
        self.assertAlmostEqual(hertz.forces[1].force[0], F, 3)

    # test the cases where the potential should be zero
    def test_noninteract(self):
        hertz = azplugins.pair.hertz(r_cut=1.0, nlist = self.nl)

        # outside cutoff
        hertz.pair_coeff.set('A','A', epsilon=1.0)
        hertz.set_params(mode="no_shift")

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = group.all())
        run(1)
        self.assertAlmostEqual(hertz.forces[0].energy, 0)
        self.assertAlmostEqual(hertz.forces[1].energy, 0)
        self.assertAlmostEqual(hertz.forces[0].force[0], 0)
        self.assertAlmostEqual(hertz.forces[1].force[0], 0)

        # inside cutoff but epsilon = 0
        hertz.pair_coeff.set('A','A', epsilon=0.0, r_cut=3.0)
        run(1)
        self.assertAlmostEqual(hertz.forces[0].energy, 0)
        self.assertAlmostEqual(hertz.forces[1].energy, 0)
        self.assertAlmostEqual(hertz.forces[0].force[0], 0)
        self.assertAlmostEqual(hertz.forces[1].force[0], 0)

    def tearDown(self):
        del self.nl
        hoomd.context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
