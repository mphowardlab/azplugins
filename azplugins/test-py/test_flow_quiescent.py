# Copyright (c) 2018-2019, Michael P. Howard
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: mphoward

from hoomd import *
context.initialize()
try:
    from hoomd import azplugins
except ImportError:
    import azplugins
import unittest
import numpy as np

# azplugins.flow.quiescent
class flow_quiescent_tests(unittest.TestCase):
    def test(self):
        u = azplugins.flow.quiescent()
        vel = u([0.0,1.0,2.0])

        np.testing.assert_array_almost_equal(vel, (0.0, 0.0, 0.0))

        vel = u((0.5,0.1,-0.2))
        self.assertAlmostEqual(vel, (0.0,0.0,0.0))

        vel = u((-0.5,-0.1,0.2))
        self.assertAlmostEqual(vel, (0.0,0.0,0.0))

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
