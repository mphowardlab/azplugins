# Copyright (c) 2018-2020, Michael P. Howard
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
import numpy

# azplugins.dpd.general
class dpd_general_tests(unittest.TestCase):
    def setUp(self):
        # raw snapshot is fine, just needs to have the types
        snap = data.make_snapshot(N=100, box=data.boxdim(L=20), particle_types=['A'])
        self.s = init.read_snapshot(snap)
        self.nl = md.nlist.cell()
        context.current.sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        dpd = azplugins.dpd.general(r_cut=1.0, nlist = self.nl, kT=1.0, seed=42)
        dpd.pair_coeff.set('A', 'A', A=25., gamma=4.5, s=0.5, r_cut=1.5)
        dpd.update_coeffs()

    # test missing A coefficient
    def test_set_missing_A(self):
        dpd = azplugins.dpd.general(r_cut=1.0, nlist = self.nl, kT=1.0, seed=42)
        dpd.pair_coeff.set('A', 'A', gamma=4.5)
        self.assertRaises(RuntimeError, dpd.update_coeffs)

    # test missing gamma coefficient
    def test_set_missing_gamma(self):
        dpd = azplugins.dpd.general(r_cut=1.0, nlist = self.nl, kT=1.0, seed=42)
        dpd.pair_coeff.set('A', 'A', A=1.0)
        self.assertRaises(RuntimeError, dpd.update_coeffs)

    # test setting exponent
    def test_set_s(self):
        dpd = azplugins.dpd.general(r_cut=1.0, nlist = self.nl, kT=1.0, seed=42)
        dpd.pair_coeff.set('A', 'A', A=1.0, gamma=4.5, s=0.5)
        dpd.update_coeffs()

    # test missing type coefficients
    def test_missing_AA(self):
        dpd = azplugins.dpd.general(r_cut=1.0, nlist = self.nl, kT=1.0, seed=42)
        self.assertRaises(RuntimeError, dpd.update_coeffs)

    # test set params
    def test_set_params(self):
        dpd = azplugins.dpd.general(r_cut=1.0, nlist = self.nl, kT=1.0, seed=42)
        dpd.set_params(kT=2.5)
        dpd.set_params(kT=variant.linear_interp([[0,1.],[100,2.]]))

    # test default coefficients
    def test_default_coeff(self):
        dpd = azplugins.dpd.general(r_cut=1.0, nlist = self.nl, kT=1.0, seed=42)
        dpd.pair_coeff.set('A', 'A', A=1.0, gamma=4.5)
        dpd.update_coeffs()

    # test coeff list
    def test_coeff_list(self):
        dpd = azplugins.dpd.general(r_cut=1.0, nlist = self.nl, kT=1.0, seed=42)
        dpd.pair_coeff.set(['A', 'B'], ['A', 'C'], A=1.0, gamma=4.5, s=1)
        dpd.update_coeffs()

    # test adding types
    def test_type_add(self):
        dpd = azplugins.dpd.general(r_cut=1.0, nlist = self.nl, kT=1.0, seed=42)
        dpd.pair_coeff.set('A', 'A', A=1.0, gamma=4.5)
        self.s.particles.types.add('B')
        self.assertRaises(RuntimeError, dpd.update_coeffs)
        dpd.pair_coeff.set('A', 'B', A=1.0, gamma=4.5)
        dpd.pair_coeff.set('B', 'B', A=1.0, gamma=4.5)
        dpd.update_coeffs()

    def tearDown(self):
        del self.s, self.nl
        context.initialize()

# test the validity of the generalized dpd potential
class dpd_general_validation_tests(unittest.TestCase):
    def setUp(self):
        if comm.get_rank() == 2:
            comm.decomposition(nx=2, ny=1, nz=1)

    # test the calculation of conservative force
    def test_conservative(self):
        snap = data.make_snapshot(N=2, box=data.boxdim(L=20),particle_types=['A'])
        if comm.get_rank() == 0:
            snap.particles.position[0] = (-0.5,0,0)
            snap.particles.position[1] = (1.5,0,0)
        s = init.read_snapshot(snap)

        nl = md.nlist.cell()
        dpd = azplugins.dpd.general(r_cut=1.0, nlist = nl, kT=0.0, seed=42)
        dpd.pair_coeff.set('A', 'A', A=2.0, gamma=4.5, s=0.5)

        # particles are initially outside the cutoff
        md.integrate.mode_standard(dt=1.e-6)
        nve = md.integrate.nve(group = group.all())
        run(1)
        U = 0.0
        F = 0.0
        f0 = dpd.forces[0].force
        f1 = dpd.forces[1].force
        e0 = dpd.forces[0].energy
        e1 = dpd.forces[1].energy

        self.assertAlmostEqual(e0,0.5*U,3)
        self.assertAlmostEqual(e1,0.5*U,3)

        self.assertAlmostEqual(f0[0],-F,3)
        self.assertAlmostEqual(f0[1],0)
        self.assertAlmostEqual(f0[2],0)

        self.assertAlmostEqual(f1[0],F,3)
        self.assertAlmostEqual(f1[1],0)
        self.assertAlmostEqual(f1[2],0)

        # move particle inside cutoff, resetting due to finite timestep being required
        snap = s.take_snapshot(all=True)
        if comm.get_rank() == 0:
            snap.particles.position[0] = (0.0,0,0)
            snap.particles.position[1] = (0.5,0,0)
            snap.particles.velocity[0] = (0,0,0)
            snap.particles.velocity[1] = (0,0,0)
        s.restore_snapshot(snap)

        # particles should now have forces on them
        run(1)
        U = 0.25
        F = 1.0
        f0 = dpd.forces[0].force
        f1 = dpd.forces[1].force
        e0 = dpd.forces[0].energy
        e1 = dpd.forces[1].energy

        self.assertAlmostEqual(e0,0.5*U,3)
        self.assertAlmostEqual(e1,0.5*U,3)

        self.assertAlmostEqual(f0[0],-F,3)
        self.assertAlmostEqual(f0[1],0)
        self.assertAlmostEqual(f0[2],0)

        self.assertAlmostEqual(f1[0],F,3)
        self.assertAlmostEqual(f1[1],0)
        self.assertAlmostEqual(f1[2],0)

    # Test that dissipative part can produce the right temperature
    #
    # This test does not validate the exponent (tricky due to randomness), but we have validated
    # in separate long running tests that the correct viscosities can be produced when s is changed.
    def test_dissipative(self):
        snap = data.make_snapshot(N=1000, box=data.boxdim(L=6),particle_types=['A'])
        snap.particles.position[:] = numpy.random.uniform(low=-1.5, high=1.5, size=(snap.particles.N, 3))
        init.read_snapshot(snap)

        nl = md.nlist.cell()
        dpd = azplugins.dpd.general(r_cut=1.0, nlist = nl, kT=1.5, seed=42)
        dpd.pair_coeff.set('A', 'A', A=0.0, gamma=4.5, s=0.5)
        md.integrate.mode_standard(dt=0.01)
        nve = md.integrate.nve(group = group.all())
        run(10)

        # record the temperature through the logger
        logger = analyze.log(filename=None, quantities=['temperature'], period=1)
        kT = []
        cb = lambda timestep : kT.append(logger.query('temperature'))
        analyze.callback(callback=cb, period=1)
        run(100)

        # average temperature should be close (within a decimal place) to the set value
        self.assertAlmostEqual(numpy.mean(kT), 1.5, 1)

    def tearDown(self):
        context.initialize()


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
