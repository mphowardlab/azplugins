# Copyright (c) 2018-2020, Michael P. Howard
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: mphoward

import hoomd
hoomd.context.initialize()
try:
    from hoomd import azplugins
except ImportError:
    import azplugins
import unittest
import numpy as np

# azplugins.flow.parabolic
class flow_parabolic_tests(unittest.TestCase):
    def test(self):
        u = azplugins.flow.parabolic(U=2.0, H=0.5)
        vel = u([2.0,1.0,0.0])

        # (3/2)U in the middle
        np.testing.assert_array_almost_equal(vel, (3.0, 0.0, 0.0))

        # zero on the upper bound
        vel = u((-0.2,0.1,0.5))
        self.assertAlmostEqual(vel, (0.0,0.0,0.0))

        # zero on the lower bound
        vel = u((0.2,-0.1,-0.5))
        self.assertAlmostEqual(vel, (0.0,0.0,0.0))

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
