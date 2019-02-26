# Copyright (c) 2018-2019, Michael P. Howard
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
import numpy as np

# azplugins.analyze.rdf
class analyze_rdf_tests(unittest.TestCase):
    def setUp(self):
        snap = data.make_snapshot(N=6, box=data.boxdim(L=20), particle_types=['A','B','C'])
        snap.particles.position[:] = [[0,0,0],[2,0,0],[0,2,0],[0,0,2],[1,1,1],[0,0,0]]
        snap.particles.typeid[:] = [0,1,0,1,0,2]
        self.s = init.read_snapshot(snap)
        self.A = group.type('A')
        self.B = group.type('B')

    # basic test of creation
    def test_creation(self):
        azplugins.analyze.rdf(groups=[self.A, self.B], rcut=3.0, bin_width=0.1, period=1)
        azplugins.analyze.rdf(groups=(self.A, self.B), rcut=3.0, bin_width=0.1, period=1)
        azplugins.analyze.rdf(groups=(self.A, self.A), rcut=3.0, bin_width=0.1, period=1)

    # test bad values of groups
    def test_bad_groups(self):
        with self.assertRaises(TypeError):
            azplugins.analyze.rdf(groups=[self.A], rcut=1.0, bin_width=0.1, period=1)
        with self.assertRaises(TypeError):
            azplugins.analyze.rdf(groups=self.A, rcut=1.0, bin_width=0.1, period=1)

    # test bad values of cutoff
    def test_bad_cutoff(self):
        with self.assertRaises(ValueError):
            azplugins.analyze.rdf(groups=[self.A, self.B], rcut=-1.0, bin_width=0.1, period=1)
        with self.assertRaises(ValueError):
            azplugins.analyze.rdf(groups=[self.A, self.B], rcut=0.0, bin_width=0.1, period=1)

    # test bad values of bin width
    def test_bad_bin_width(self):
        with self.assertRaises(ValueError):
            azplugins.analyze.rdf(groups=[self.A, self.B], rcut=3.0, bin_width=-1.0, period=1)
        with self.assertRaises(ValueError):
            azplugins.analyze.rdf(groups=[self.A, self.B], rcut=3.0, bin_width=0.0, period=1)

    # test analysis
    def test_analyze_self(self):
        rdf_A = azplugins.analyze.rdf(groups=[self.A, self.A], rcut=3.0, bin_width=0.5, period=1)
        rdf_B = azplugins.analyze.rdf(groups=[self.B, self.B], rcut=3.0, bin_width=0.5, period=1)
        rdf_AB = azplugins.analyze.rdf(groups=[self.A, self.B], rcut=3.0, bin_width=0.5, period=1)

        # run a few steps and compare against VMD results
        run(3)
        bins, gr = rdf_A()
        np.testing.assert_array_almost_equal(bins, [0.25, 0.75, 1.25, 1.75, 2.25, 2.75])
        np.testing.assert_array_almost_equal(gr, [0,0,0,275.29503669949463,83.49111768755165,0], decimal=3)

        bins, gr = rdf_B()
        np.testing.assert_array_almost_equal(bins, [0.25, 0.75, 1.25, 1.75, 2.25, 2.75])
        np.testing.assert_array_almost_equal(gr, [0,0,0,0,0,167.89972018485662], decimal=3)

        bins, gr = rdf_AB()
        np.testing.assert_array_almost_equal(bins, [0.25, 0.75, 1.25, 1.75, 2.25, 2.75])
        np.testing.assert_array_almost_equal(gr, [0,0,0,137.64751834974732,83.49111768755165,55.966573394952206], decimal=3)

        # reset and everything should give zeros
        rdf_A.reset()
        bins,gr = rdf_A()
        np.testing.assert_array_almost_equal(gr, [0,0,0,0,0,0])

        rdf_B.reset()
        bins,gr = rdf_B()
        np.testing.assert_array_almost_equal(gr, [0,0,0,0,0,0])

        rdf_AB.reset()
        bins,gr = rdf_AB()
        np.testing.assert_array_almost_equal(gr, [0,0,0,0,0,0])

        # run again and make sure reset happend correctly
        run(10)
        bins, gr = rdf_A()
        np.testing.assert_array_almost_equal(bins, [0.25, 0.75, 1.25, 1.75, 2.25, 2.75])
        np.testing.assert_array_almost_equal(gr, [0,0,0,275.29503669949463,83.49111768755165,0], decimal=3)

        bins, gr = rdf_B()
        np.testing.assert_array_almost_equal(bins, [0.25, 0.75, 1.25, 1.75, 2.25, 2.75])
        np.testing.assert_array_almost_equal(gr, [0,0,0,0,0,167.89972018485662], decimal=3)

        bins, gr = rdf_AB()
        np.testing.assert_array_almost_equal(bins, [0.25, 0.75, 1.25, 1.75, 2.25, 2.75])
        np.testing.assert_array_almost_equal(gr, [0,0,0,137.64751834974732,83.49111768755165,55.966573394952206], decimal=3)

    def tearDown(self):
        del self.s, self.A, self.B
        context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
