# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021, Auburn University
# This file is part of the azplugins project, released under the Modified BSD License.

import hoomd
hoomd.context.initialize()
try:
    from hoomd import azplugins
except ImportError:
    import azplugins
import unittest
import numpy as np

# azplugins.flow.constant
class flow_constant_tests(unittest.TestCase):
    def test(self):
        u = azplugins.flow.constant(U=(1,-2,3))

        # try one spot
        vel = u([-2.,1.,-4.])
        np.testing.assert_array_almost_equal(vel, (1,-2,3))

        # try a different spot
        vel = u((2.,-1.,4.))
        np.testing.assert_array_almost_equal(vel, (1,-2,3))

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
