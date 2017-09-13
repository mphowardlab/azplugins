# Copyright (c) 2016-2017, Panagiotopoulos Group, Princeton University
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: wes_reinhart

from hoomd import *
from hoomd import md
context.initialize()
try:
    from hoomd import azplugins
except ImportError:
    import azplugins
import unittest

# azplugins.pair.two_patch_morse
class aniso_pair_two_patch_morse_tests(unittest.TestCase):
    def setUp(self):
        # raw snapshot is fine, just needs to have the types
        snap = data.make_snapshot(N=100, box=data.boxdim(L=20), particle_types=['A'])
        if comm.get_rank() == 0:
            snap.particles.moment_inertia[:] = (0.1,0.1,0.1)
        self.s = init.read_snapshot(snap)
        self.nl = md.nlist.cell()
        context.current.sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        tpm = azplugins.pair.two_patch_morse(r_cut=1.6, nlist = self.nl)
        tpm.pair_coeff.set('A', 'A', Md=1.8341, Mr=0.0302, req=1.0043, alpha=0.40, omega=50.0,repulsion=True)
        tpm.update_coeffs()

    # test missing coefficients
    def test_set_missing_Md(self):
        tpm = azplugins.pair.two_patch_morse(r_cut=1.6, nlist = self.nl)
        tpm.pair_coeff.set('A', 'A', Mr=0.0302, req=1.0043, alpha=0.40, omega=50.0)
        self.assertRaises(RuntimeError, tpm.update_coeffs)

    # test missing coefficients
    def test_set_missing_Mr(self):
        tpm = azplugins.pair.two_patch_morse(r_cut=1.6, nlist = self.nl)
        tpm.pair_coeff.set('A', 'A', Md=1.8341, req=1.0043, alpha=0.40, omega=50.0)
        self.assertRaises(RuntimeError, tpm.update_coeffs)

    # test missing coefficients
    def test_set_missing_req(self):
        tpm = azplugins.pair.two_patch_morse(r_cut=1.6, nlist = self.nl)
        tpm.pair_coeff.set('A', 'A', Md=1.8341, Mr=0.0302, alpha=0.40, omega=50.0)
        self.assertRaises(RuntimeError, tpm.update_coeffs)

    # test missing coefficients
    def test_set_missing_alpha(self):
        tpm = azplugins.pair.two_patch_morse(r_cut=1.6, nlist = self.nl)
        tpm.pair_coeff.set('A', 'A', Md=1.8341, Mr=0.0302, req=1.0043, omega=50.0)
        self.assertRaises(RuntimeError, tpm.update_coeffs)

    # test missing coefficients
    def test_set_missing_omega(self):
        tpm = azplugins.pair.two_patch_morse(r_cut=1.6, nlist = self.nl)
        tpm.pair_coeff.set('A', 'A', Md=1.8341, Mr=0.0302, req=1.0043, alpha=0.40)
        self.assertRaises(RuntimeError, tpm.update_coeffs)

    # test missing coefficients
    def test_set_missing_AA(self):
        tpm = azplugins.pair.two_patch_morse(r_cut=1.6, nlist = self.nl)
        self.assertRaises(RuntimeError, tpm.update_coeffs)

    # test set params
    def test_set_params(self):
        tpm = azplugins.pair.two_patch_morse(r_cut=1.6, nlist = self.nl)
        tpm.set_params(mode="no_shift")
        tpm.set_params(mode="shift")
        self.assertRaises(RuntimeError, tpm.set_params, mode="blah")

    # test specific nlist subscription
    def test_nlist_subscribe(self):
        tpm = azplugins.pair.two_patch_morse(r_cut=1.6, nlist = self.nl)

        tpm.pair_coeff.set('A', 'A', Md=1.8341, Mr=0.0302, req=1.0043, alpha=0.40, omega=50.0)
        self.nl.update_rcut()
        self.assertAlmostEqual(1.6, self.nl.r_cut.get_pair('A','A'))

        tpm.pair_coeff.set('A', 'A', r_cut = 2.0)
        self.nl.update_rcut()
        self.assertAlmostEqual(2.0, self.nl.r_cut.get_pair('A','A'))

    # test coeff list
    def test_coeff_list(self):
        tpm = azplugins.pair.two_patch_morse(r_cut=1.6, nlist = self.nl)
        tpm.pair_coeff.set(['A', 'B'], ['A', 'C'], Md=1.8341, Mr=0.0302, req=1.0043, alpha=0.40, omega=50.0)
        tpm.update_coeffs()

    # test adding types
    def test_type_add(self):
        tpm = azplugins.pair.two_patch_morse(r_cut=1.6, nlist = self.nl)
        tpm.pair_coeff.set('A', 'A', Md=1.8341, Mr=0.0302, req=1.0043, alpha=0.40, omega=50.0)
        self.s.particles.types.add('B')
        self.assertRaises(RuntimeError, tpm.update_coeffs)
        tpm.pair_coeff.set('A', 'B', Md=1.8341, Mr=0.0302, req=1.0043, alpha=0.40, omega=50.0)
        tpm.pair_coeff.set('B', 'B', Md=1.8341, Mr=0.0302, req=1.0043, alpha=0.40, omega=50.0)
        tpm.update_coeffs()

    def tearDown(self):
        del self.s, self.nl
        context.initialize()

# test the validity of the pair potential
class potential_two_patch_morse_tests(unittest.TestCase):
    def setUp(self):
        snap = data.make_snapshot(N=2, box=data.boxdim(L=20),particle_types=['A'])
        if comm.get_rank() == 0:
            snap.particles.moment_inertia[:] = (0.1,0.1,0.1)
            snap.particles.position[0] = (-0.5,-0.10,-0.15)
            snap.particles.orientation[0] = (1,0,0,0)
            snap.particles.position[1] = (0.5,0.10,0.15)
            snap.particles.orientation[1] = (1,0,0,0)
        self.s = init.read_snapshot(snap)
        self.nl = md.nlist.cell()
        context.current.sorter.set_params(grid=8)

    # test the calculation of force and potential
    def test_potential(self):
        self.tpm = azplugins.pair.two_patch_morse(r_cut=1.6, nlist = self.nl)

        # test without potential shifting
        self.tpm.pair_coeff.set('A', 'A', Md=1.8341, Mr=0.0302, req=1.0043, omega=5.0, alpha=0.40)
        self.tpm.set_params(mode="no_shift")

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = group.all())
        run(1)
        U = -0.20567
        F = (11.75766, 2.46991, 3.70487)
        T = (-0.000000, -0.08879,  0.05919)
        e0 = self.tpm.forces[0].energy
        e1 = self.tpm.forces[1].energy
        f0 = self.tpm.forces[0].force
        f1 = self.tpm.forces[1].force
        t0 = self.tpm.forces[0].torque
        t1 = self.tpm.forces[1].torque

        self.assertAlmostEqual(e0,U,3)
        self.assertAlmostEqual(e1,U,3)

        self.assertAlmostEqual(f0[0],F[0],3)
        self.assertAlmostEqual(f0[1],F[1],3)
        self.assertAlmostEqual(f0[2],F[2],3)

        self.assertAlmostEqual(f1[0],-F[0],3)
        self.assertAlmostEqual(f1[1],-F[1],3)
        self.assertAlmostEqual(f1[2],-F[2],3)

        self.assertAlmostEqual(t0[0],T[0],3)
        self.assertAlmostEqual(t0[1],T[1],3)
        self.assertAlmostEqual(t0[2],T[2],3)

        self.assertAlmostEqual(t1[0],T[0],3)
        self.assertAlmostEqual(t1[1],T[1],3)
        self.assertAlmostEqual(t1[2],T[2],3)

        # test that energy shifting works
        self.tpm.pair_coeff.set('A', 'A', r_cut = 1.10)
        self.tpm.set_params(mode='shift')
        run(1)
        U = -0.14195
        e0 = self.tpm.forces[0].energy
        e1 = self.tpm.forces[1].energy
        self.assertAlmostEqual(e0,U,3)
        self.assertAlmostEqual(e1,U,3)

    # test the cases where the potential should be zero
    def test_noninteract(self):
        # particles are outside cutoff
        self.tpm = azplugins.pair.two_patch_morse(r_cut=1.0, nlist = self.nl)

        # test without potential shifting
        self.tpm.pair_coeff.set('A', 'A', Md=1.8341, Mr=0.0302, req=1.0043, omega=5.0, alpha=0.40)

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = group.all())
        run(1)
        self.assertAlmostEqual(self.tpm.forces[0].energy, 0)
        self.assertAlmostEqual(self.tpm.forces[1].energy, 0)
        self.assertAlmostEqual(self.tpm.forces[0].force[0], 0)
        self.assertAlmostEqual(self.tpm.forces[1].force[0], 0)

        # inside cutoff but Md = 0
        self.tpm.pair_coeff.set('A','A', Md=0.0, r_cut=1.6)
        run(1)
        self.assertAlmostEqual(self.tpm.forces[0].energy, 0)
        self.assertAlmostEqual(self.tpm.forces[1].energy, 0)
        self.assertAlmostEqual(self.tpm.forces[0].force[0], 0)
        self.assertAlmostEqual(self.tpm.forces[1].force[0], 0)

    # test the cases where the force should be zero
    def test_inside_minimum(self):
        snap = self.s.take_snapshot(all=True)
        if comm.get_rank() == 0:
            snap.particles.moment_inertia[:] = (0.1,0.1,0.1)
            snap.particles.position[0] = (-0.25,0,0)
            snap.particles.orientation[0] = (1,0,0,0)
            snap.particles.position[1] = (0.25,0,0)
            snap.particles.orientation[1] = (1,0,0,0)
        self.s.restore_snapshot(snap)

        # test no force
        self.tpm = azplugins.pair.two_patch_morse(r_cut=1.6, nlist = self.nl)
        self.tpm.pair_coeff.set('A', 'A', Md=1.8341, Mr=0.0302, req=1.0043, omega=100.0, alpha=0.40,repulsion=False)

        md.integrate.mode_standard(dt=0)
        nve = md.integrate.nve(group = group.all())
        run(1)
        self.assertAlmostEqual(self.tpm.forces[0].energy, 0.5 * -1.8341)
        self.assertAlmostEqual(self.tpm.forces[1].energy, 0.5 * -1.8341)
        self.assertAlmostEqual(self.tpm.forces[0].force[0], 0)
        self.assertAlmostEqual(self.tpm.forces[1].force[0], 0)

        # test force
        self.tpm = azplugins.pair.two_patch_morse(r_cut=1.6, nlist = self.nl)
        self.tpm.pair_coeff.set('A', 'A', Md=1.8341, Mr=0.0302, req=1.0043, omega=100.0, alpha=0.40,repulsion=True)

        run(1)
        self.assertTrue(self.tpm.forces[0].energy > 2.92e14)
        self.assertTrue(self.tpm.forces[1].energy > 2.92e14)
        self.assertTrue(self.tpm.forces[0].force[0] < -3.87e16)
        self.assertTrue(self.tpm.forces[1].force[0] > +3.87e16)

    def tearDown(self):
        del self.s, self.nl, self.tpm
        context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
