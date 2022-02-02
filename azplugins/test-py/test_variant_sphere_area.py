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

class variant_sphere_area_tests(unittest.TestCase):
    def setUp(self):
        hoomd.init.read_snapshot(hoomd.data.make_snapshot(N=0, box=hoomd.data.boxdim(L=20.)))

    def test_create(self):
        """Test creation of sphere objects."""
        # floats
        s = azplugins.variant.sphere_area(R0=10., alpha=0.5)
        self.assertAlmostEqual(s.R0, 10.)
        self.assertAlmostEqual(s.alpha, 0.5)
        self.assertTrue(s.cpp_variant is not None)

        # ints
        azplugins.variant.sphere_area(R0=10, alpha=1)

    def test_value(self):
        """Test for value of sphere variant."""
        s = azplugins.variant.sphere_area(R0=10., alpha=0.5)

        # initial value
        R = s.cpp_variant.getValue(0)
        self.assertAlmostEqual(R, 10.)

        # evaluated in python at this timestep
        R = s.cpp_variant.getValue(1000)
        self.assertAlmostEqual(R, 7.7595917564667127, 5)

        # should now be 0, since has exceeded max time
        R = s.cpp_variant.getValue(3000)
        self.assertAlmostEqual(R, 0.0)

    def tearDown(self):
        hoomd.context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
