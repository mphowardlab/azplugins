# Copyright (c) 2018-2020, Michael P. Howard
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: arjunsg2
# Unit test class for azplugins.flow.sllod, modelled after
# test_flow_langevin.py

import hoomd
hoomd.context.initialize()
from hoomd import md
from hoomd import data
# try:
#     from hoomd import azplugins
# except ImportError:
#     import azplugins
import azplugins
import unittest
import numpy as np

# azplugins.flow.sllod
class flow_sllod_tests (unittest.TestCase):
    def setUp(self):
        snap = hoomd.data.make_snapshot(N=100, box=hoomd.data.boxdim(L=20), particle_types=['A'])
        self.s = hoomd.init.read_snapshot(snap)
        md.integrate.mode_standard(dt=0.005)

    # tests basic creation of the integration method
    def test(self):
        all = hoomd.group.all()
        bd = azplugins.flow.sllod(all, kT=1.2, gamma_dot=1.0)
        hoomd.run(1)
        bd.disable()

    # # test set_params
    # def test_set_params(self):
    #     all = hoomd.group.all()
    #
    #     bd = azplugins.flow.langevin(all, kT=1.2, flow=self.u, seed=1)
    #     bd.set_params(kT=1.3)
    #     bd.set_params(kT=hoomd.variant.linear_interp([(0, 4.0), (1e6, 1.0)]))
    #     bd.set_params(noiseless=True)
    #
    #     v = azplugins.flow.parabolic(U=1.0, H=4.0)
    #     bd.set_params(flow=v)
    #
    # # test set_gamma
    # def test_set_gamma(self):
    #     all = hoomd.group.all()
    #     bd = azplugins.flow.langevin(all, kT=1.2, flow=self.u, seed=1)
    #     bd.set_gamma('A', 0.5)
    #     bd.set_gamma('B', 1.0)

    def tearDown(self):
        del self.s
        hoomd.context.initialize()

# test the validity of the SLLOD algorithm implementation
class integrate_sllod_tests(unittest.TestCase):
    def setUp(self):
        snap = hoomd.data.make_snapshot(N=5, box=data.boxdim(L=20),particle_types=['A'])
        if hoomd.comm.get_rank() == 0:
            snap.particles.position[0] = (0,0,0)
            snap.particles.position[1] = (0,3.15,0)
            snap.particles.position[2] = (2,-4.25,0)
            snap.particles.position[3] = (0,0,5)
            snap.particles.position[4] = (-5,0,0)
        self.s = hoomd.init.read_snapshot(snap)
        self.nl = md.nlist.cell()
        md.integrate.mode_standard(dt=0.005)

    def test_1particle(self):
        all = hoomd.group.all()
        bd = azplugins.flow.sllod(all, kT=1.2, gamma_dot=1.5)
        hoomd.run(1)
        bd.disable()

        snap = self.s.take_snapshot()
        self.assertAlmostEqual(snap.particles.position[0][0], 0)
        self.assertAlmostEqual(snap.particles.position[1][0], 0.023625)
        self.assertAlmostEqual(snap.particles.position[2][0], 1.968125)
        self.assertAlmostEqual(snap.particles.position[3][0], 0)
        self.assertAlmostEqual(snap.particles.position[4][0], -5)

    # # test the calculation of force and potential
    # def test_potential(self):
    #     hertz = azplugins.pair.hertz(r_cut=1.5, nlist = self.nl)
    #     hertz.pair_coeff.set('A','A', epsilon=2.0)
    #     hertz.set_params(mode="no_shift")
    #
    #     md.integrate.mode_standard(dt=0)
    #     nve = md.integrate.nve(group = hoomd.group.all())
    #     hoomd.run(1)
    #     U = 0.09859
    #     F = 0.54772
    #     f0 = hertz.forces[0].force
    #     f1 = hertz.forces[1].force
    #     e0 = hertz.forces[0].energy
    #     e1 = hertz.forces[1].energy
    #
    #     self.assertAlmostEqual(e0,0.5*U,3)
    #     self.assertAlmostEqual(e1,0.5*U,3)
    #
    #     self.assertAlmostEqual(f0[0],-F,3)
    #     self.assertAlmostEqual(f0[1],0)
    #     self.assertAlmostEqual(f0[2],0)
    #
    #     self.assertAlmostEqual(f1[0],F,3)
    #     self.assertAlmostEqual(f1[1],0)
    #     self.assertAlmostEqual(f1[2],0)
    #
    #     hertz = azplugins.pair.hertz(r_cut=2.05, nlist = self.nl)
    #     hertz.pair_coeff.set('A','A', epsilon=3.0)
    #     hertz.set_params(mode='shift')
    #     hoomd.run(1)
    #     U = 0.498582
    #     F = 1.246455
    #     self.assertAlmostEqual(hertz.forces[0].energy, 0.5*U, 3)
    #     self.assertAlmostEqual(hertz.forces[1].energy, 0.5*U, 3)
    #     self.assertAlmostEqual(hertz.forces[0].force[0], -F, 3)
    #     self.assertAlmostEqual(hertz.forces[1].force[0], F, 3)
    #
    # # test the cases where the potential should be zero
    # def test_noninteract(self):
    #     hertz = azplugins.pair.hertz(r_cut=1.0, nlist = self.nl)
    #
    #     # outside cutoff
    #     hertz.pair_coeff.set('A','A', epsilon=1.0)
    #     hertz.set_params(mode="no_shift")
    #
    #     md.integrate.mode_standard(dt=0)
    #     nve = md.integrate.nve(group = hoomd.group.all())
    #     hoomd.run(1)
    #     self.assertAlmostEqual(hertz.forces[0].energy, 0)
    #     self.assertAlmostEqual(hertz.forces[1].energy, 0)
    #     self.assertAlmostEqual(hertz.forces[0].force[0], 0)
    #     self.assertAlmostEqual(hertz.forces[1].force[0], 0)
    #
    #     # inside cutoff but epsilon = 0
    #     hertz.pair_coeff.set('A','A', epsilon=0.0, r_cut=3.0)
    #     hoomd.run(1)
    #     self.assertAlmostEqual(hertz.forces[0].energy, 0)
    #     self.assertAlmostEqual(hertz.forces[1].energy, 0)
    #     self.assertAlmostEqual(hertz.forces[0].force[0], 0)
    #     self.assertAlmostEqual(hertz.forces[1].force[0], 0)

    def tearDown(self):
        del self.s, self.nl
        hoomd.context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
