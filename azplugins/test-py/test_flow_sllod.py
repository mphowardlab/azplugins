# Copyright (c) 2018-2020, Michael P. Howard
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: arjunsg2

# Unit test class for azplugins.flow.sllod,
# modelled off of test_flow_langevin.py
import hoomd
hoomd.context.initialize()
from hoomd import md
try:
    from hoomd import azplugins
except ImportError:
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
    #
    # # test adding types
    # def test_add_type(self):
    #     all = hoomd.group.all()
    #     bd = azplugins.flow.langevin(all, kT=1.2, flow=self.u, seed=1)
    #     bd.set_gamma('A', 0.5)
    #     bd.set_gamma('B', 1.0)
    #     hoomd.run(1)
    #
    #     self.s.particles.types.add('B')
    #     hoomd.run(1)

    def tearDown(self):
        del self.s
        hoomd.context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
