# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2022, Auburn University
# This file is part of the azplugins project, released under the Modified BSD License.

import hoomd
hoomd.context.initialize()
from hoomd import md
try:
    from hoomd import azplugins
except ImportError:
    import azplugins
import unittest
import numpy as np

# azplugins.flow.brownian
class flow_brownian_tests (unittest.TestCase):
    def setUp(self):
        snap = hoomd.data.make_snapshot(N=100, box=hoomd.data.boxdim(L=20), particle_types=['A'])
        self.s = hoomd.init.read_snapshot(snap)

        self.u = azplugins.flow.parabolic(U=2.0, H=10.0)

        md.integrate.mode_standard(dt=0.005)

    # tests basic creation of the integration method
    def test(self):
        all = hoomd.group.all()

        bd = azplugins.flow.brownian(all, kT=1.2, flow=self.u, seed=52)
        hoomd.run(1)
        bd.disable()

        bd = azplugins.flow.brownian(all, kT=1.2, flow=self.u, seed=1, dscale=0.5)
        hoomd.run(1)
        bd.disable()

        bd = azplugins.flow.brownian(all, kT=1.2, flow=self.u, seed=1, noiseless=True)
        hoomd.run(1)
        bd.disable()

    # test set_params
    def test_set_params(self):
        all = hoomd.group.all()

        bd = azplugins.flow.brownian(all, kT=1.2, flow=self.u, seed=1)
        bd.set_params(kT=1.3)
        bd.set_params(kT=hoomd.variant.linear_interp([(0, 4.0), (1e6, 1.0)]))
        bd.set_params(noiseless=True)

        v = azplugins.flow.parabolic(U=1.0, H=4.0)
        bd.set_params(flow=v)

    # test set_gamma
    def test_set_gamma(self):
        all = hoomd.group.all()
        bd = azplugins.flow.brownian(all, kT=1.2, flow=self.u, seed=1)
        bd.set_gamma('A', 0.5)
        bd.set_gamma('B', 1.0)

    # test adding types
    def test_add_type(self):
        all = hoomd.group.all()
        bd = azplugins.flow.brownian(all, kT=1.2, flow=self.u, seed=1)
        bd.set_gamma('A', 0.5)
        bd.set_gamma('B', 1.0)
        hoomd.run(1)

        self.s.particles.types.add('B')
        hoomd.run(1)

    # test construction fails with a bad flow type
    def test_bad_flow(self):
        all = hoomd.group.all()

        ## test known flow fields first
        # constant
        azplugins.flow.brownian(all, kT=1.2, flow=azplugins.flow.constant(U=(1,0,0)), seed=1)
        # parabolic
        azplugins.flow.brownian(all, kT=1.2, flow=self.u, seed=1)
        # quiescent
        azplugins.flow.brownian(all, kT=1.2, flow=azplugins.flow.quiescent(), seed=1)

        # passing some garbage object is an error
        with self.assertRaises(TypeError):
            bd = azplugins.flow.brownian(all, kT=1.2, flow=all, seed=1)

        # changing flow profiles is an error
        bd = azplugins.flow.brownian(all, kT=1.2, flow=self.u, seed=1)
        with self.assertRaises(TypeError):
            v = azplugins.flow.quiescent()
            bd.set_params(flow=v)

    # test that flow field is being applied properly
    def test_flow_value(self):
        all = hoomd.group.all()
        bd = azplugins.flow.brownian(all, kT=1.2, flow=self.u, seed=1, noiseless=True)
        hoomd.run(1)

        # In Brownian dynamics, the applied velocity has an instantaneous effect on the particle
        np.testing.assert_array_almost_equal(self.s.particles[0].position, [0.015,0.0,0.0])

    def tearDown(self):
        del self.s, self.u
        hoomd.context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
