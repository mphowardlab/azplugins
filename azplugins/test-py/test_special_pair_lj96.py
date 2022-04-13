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

# azplugins.special_pair.lj96
class special_pair_lj96_tests(unittest.TestCase):
    def setUp(self):
        # raw snapshot is fine, just needs to have the types
        snap = hoomd.data.make_snapshot(N=100,
                                  box=hoomd.data.boxdim(L=20),
                                  particle_types=['A'],
                                  pair_types=['A-A'])
        snap.pairs.resize(50)
        for particle in range(50):
            snap.pairs.group[particle] = [particle*2, particle*2+1]
            snap.pairs.typeid[:] = 0
        self.s = hoomd.init.read_snapshot(snap)
        hoomd.context.current.sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        lj96 = azplugins.special_pair.lj96()
        lj96.pair_coeff.set('A-A', epsilon=1.0, sigma=1.0, alpha=1.0, r_cut=2.5)
        lj96.update_coeffs()

    # test missing epsilon coefficient
    def test_set_missing_epsilon(self):
        lj96 = azplugins.special_pair.lj96()
        lj96.pair_coeff.set('A-A', sigma=1.0, r_cut=2.5)
        self.assertRaises(RuntimeError, lj96.update_coeffs)

    # test missing sigma coefficient
    def test_set_missing_sigma(self):
        lj96 = azplugins.special_pair.lj96()
        lj96.pair_coeff.set('A-A', epsilon=1.0, r_cut=2.5)
        self.assertRaises(RuntimeError, lj96.update_coeffs)

    # test missing r_cut parameter
    def test_set_missing_rcut(self):
        lj96 = azplugins.special_pair.lj96()
        lj96.pair_coeff.set('A-A', epsilon=1.0, sigma=1.0)
        self.assertRaises(RuntimeError, lj96.update_coeffs)

    # test missing all coefficients
    def test_missing_AA(self):
        lj96 = azplugins.special_pair.lj96()
        self.assertRaises(RuntimeError, lj96.update_coeffs)

    # test set energy mode
    def test_set_mode(self):
        lj96 = azplugins.special_pair.lj96()
        lj96.pair_coeff.set('A-A', epsilon=1.0, sigma=1.0, r_cut=2.5, mode="blah")
        self.assertRaises(RuntimeError, lj96.update_coeffs)

    # test default coefficients
    def test_default_coeff(self):
        lj96 = azplugins.special_pair.lj96()
        lj96.pair_coeff.set('A-A', epsilon=1.0, sigma=1.0, r_cut=2.0)
        lj96.update_coeffs()

    def tearDown(self):
        del self.s
        hoomd.context.initialize()

# test the validity of the pair potential
class potential_lj96_tests(unittest.TestCase):
    def setUp(self):
        snap = hoomd.data.make_snapshot(N=2,
                                  box=hoomd.data.boxdim(L=20),
                                  particle_types=['A'],
                                  pair_types=['A-A'])
        if hoomd.comm.get_rank() == 0:
            snap.particles.position[0] = (0,0,0)
            snap.particles.position[1] = (1.05,0,0)
            snap.pairs.resize(1)
            snap.pairs.group[0] = [0,1]
            snap.pairs.typeid[0] = 0
        hoomd.init.read_snapshot(snap)

    # test the calculation of force and potential
    def test_potential(self):
        lj96 = azplugins.special_pair.lj96()
        lj96.pair_coeff.set('A-A', epsilon=2.0, sigma=1.05, r_cut=3, mode="no_shift")

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = hoomd.group.all())
        hoomd.run(1)
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

        # test energy shift mode
        lj96.pair_coeff.set('A-A', mode="shift")
        hoomd.run(1)
        U = 0.0238
        F = -38.5714
        self.assertAlmostEqual(lj96.forces[0].energy, 0.5*U, 3)
        self.assertAlmostEqual(lj96.forces[1].energy, 0.5*U, 3)
        self.assertAlmostEqual(lj96.forces[0].force[0], F, 3)
        self.assertAlmostEqual(lj96.forces[1].force[0], -F, 3)

        lj96.pair_coeff.set('A-A', sigma=0.85, mode="shift")
        hoomd.run(1)
        U = -1.7770
        F = 4.4343
        self.assertAlmostEqual(lj96.forces[0].energy, 0.5*U, 3)
        self.assertAlmostEqual(lj96.forces[1].energy, 0.5*U, 3)
        self.assertAlmostEqual(lj96.forces[0].force[0], F, 3)
        self.assertAlmostEqual(lj96.forces[1].force[0], -F, 3)

    # test alpha parameter in potential. if potential is handled right,
    # coefficients are processed correctly and force will also be correct
    def test_alpha(self):
        lj96 = azplugins.special_pair.lj96()
        lj96.pair_coeff.set('A-A', epsilon=2.0, sigma=1.05, r_cut=2.5, alpha=0.5, mode="no_shift")

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = hoomd.group.all())
        hoomd.run(1)

        U = 6.75
        F = -77.1429
        self.assertAlmostEqual(lj96.forces[0].energy,0.5*U,3)
        self.assertAlmostEqual(lj96.forces[1].energy,0.5*U,3)
        self.assertAlmostEqual(lj96.forces[0].force[0], F, 3)
        self.assertAlmostEqual(lj96.forces[1].force[0], -F, 3)

        lj96.pair_coeff.set('A-A', sigma=0.5)
        hoomd.run(1)
        U = -0.06171
        F = 0.3040
        self.assertAlmostEqual(lj96.forces[0].energy,0.5*U,3)
        self.assertAlmostEqual(lj96.forces[1].energy,0.5*U,3)
        self.assertAlmostEqual(lj96.forces[0].force[0], F, 3)
        self.assertAlmostEqual(lj96.forces[1].force[0], -F, 3)

    # test the cases where the potential should be zero
    def test_noninteract(self):
        lj96 = azplugins.special_pair.lj96()

        # outside cutoff
        lj96.pair_coeff.set('A-A', epsilon=1.0, sigma=1.0, r_cut=1.0, mode="no_shift")

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = hoomd.group.all())
        hoomd.run(1)
        self.assertAlmostEqual(lj96.forces[0].energy, 0)
        self.assertAlmostEqual(lj96.forces[1].energy, 0)
        self.assertAlmostEqual(lj96.forces[0].force[0], 0)
        self.assertAlmostEqual(lj96.forces[1].force[0], 0)

        # inside cutoff but epsilon = 0
        lj96.pair_coeff.set('A-A', epsilon=0.0, sigma=1.0, r_cut=3.0)
        hoomd.run(1)
        self.assertAlmostEqual(lj96.forces[0].energy, 0)
        self.assertAlmostEqual(lj96.forces[1].energy, 0)
        self.assertAlmostEqual(lj96.forces[0].force[0], 0)
        self.assertAlmostEqual(lj96.forces[1].force[0], 0)

    def tearDown(self):
        hoomd.context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
