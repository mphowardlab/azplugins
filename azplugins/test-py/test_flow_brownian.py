# Copyright (c) 2018-2019, Michael P. Howard
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: mphoward

from hoomd import *
context.initialize()
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
        snap = data.make_snapshot(N=100, box=data.boxdim(L=20), particle_types=['A'])
        self.s = init.read_snapshot(snap)

        self.u = azplugins.flow.parabolic(U=2.0, H=10.0)

        md.integrate.mode_standard(dt=0.005)

    # tests basic creation of the integration method
    def test(self):
        all = group.all()

        bd = azplugins.flow.brownian(all, kT=1.2, flow=self.u, seed=52)
        run(1)
        bd.disable()

        bd = azplugins.flow.brownian(all, kT=1.2, flow=self.u, seed=1, dscale=0.5)
        run(1)
        bd.disable()

        bd = azplugins.flow.brownian(all, kT=1.2, flow=self.u, seed=1, noiseless=True)
        run(1)
        bd.disable()

    # test set_params
    def test_set_params(self):
        all = group.all()

        bd = azplugins.flow.brownian(all, kT=1.2, flow=self.u, seed=1)
        bd.set_params(kT=1.3)
        bd.set_params(kT=variant.linear_interp([(0, 4.0), (1e6, 1.0)]))
        bd.set_params(noiseless=True)

        v = azplugins.flow.parabolic(U=1.0, H=4.0)
        bd.set_params(flow=v)

    # test set_gamma
    def test_set_gamma(self):
        all = group.all()
        bd = azplugins.flow.brownian(all, kT=1.2, flow=self.u, seed=1)
        bd.set_gamma('A', 0.5)
        bd.set_gamma('B', 1.0)

    # test adding types
    def test_add_type(self):
        all = group.all()
        bd = azplugins.flow.brownian(all, kT=1.2, flow=self.u, seed=1)
        bd.set_gamma('A', 0.5)
        bd.set_gamma('B', 1.0)
        run(1)

        self.s.particles.types.add('B')
        run(1)

    # test construction fails with a bad flow type
    def test_bad_flow(self):
        all = group.all()

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
        all = group.all()
        bd = azplugins.flow.brownian(all, kT=1.2, flow=self.u, seed=1, noiseless=True)
        run(1)

        # In Brownian dynamics, the applied velocity has an instantaneous effect on the particle
        np.testing.assert_array_almost_equal(self.s.particles[0].position, [0.015,0.0,0.0])

    def tearDown(self):
        del self.s, self.u
        context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
