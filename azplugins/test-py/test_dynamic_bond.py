# Copyright (c) 2018-2020, Michael P. Howard
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: astatt

import hoomd
from hoomd import md
hoomd.context.initialize()
try:
    from hoomd import azplugins
except ImportError:
    import azplugins
import unittest
import numpy as np

class update_dynamic_bond_tests_two_groups(unittest.TestCase):
    def setUp(self):
        snap = hoomd.data.make_snapshot(N=4, box=hoomd.data.boxdim(L=20),
                                        particle_types=['A','B'],
                                        bond_types=['bond'])
        if hoomd.comm.get_rank() == 0:
            snap.particles.position[:,0] = (0,0.9,1.1,2)
            snap.particles.position[:,1] = (0,0,0,0)
            snap.particles.position[:,2] = (0,0,0,0)
            snap.particles.typeid[:] = [1,0,1,0]

        self.s = hoomd.init.read_snapshot(snap)
        self.nl = hoomd.md.nlist.cell()

        self.group_1 = hoomd.group.tag_list(name="a", tags = [0,1])
        self.group_2 = hoomd.group.tag_list(name="b", tags = [2,3])
        self.u = azplugins.update.dynamic_bond(nlist=self.nl,
                                               r_cut=1.0,
                                               bond_type='bond',
                                               group_1=self.group_1,
                                               group_2=self.group_2,
                                               max_bonds_1=1,
                                               max_bonds_2=2)

    def test_set_params(self):
        self.assertEqual(self.u.cpp_updater.r_cut, 1)
        self.u.set_params(r_cut=1.5)
        self.assertEqual(self.u.cpp_updater.r_cut, 1.5)

        # check the test of box size large enough for cutoff
        with self.assertRaises(RuntimeError):
            self.u.set_params(r_cut=15.0)

        self.u.set_params(max_bonds_1=3)
        self.assertEqual(self.u.cpp_updater.max_bonds_group_1,3)
        self.u.set_params(max_bonds_2=7)
        self.assertEqual(self.u.cpp_updater.max_bonds_group_2,7)

        # check the test of bondy_type
        with self.assertRaises(RuntimeError):
            self.u.set_params(bond_type='not_existing')


    def test_form_bond(self):
        # test bond formation between particle 1-2
        # particle 0,1 are in the same group, so even if their distance is
        # below r_cut, no bond should be formed, same for goes for particle 2,3
        hoomd.md.integrate.mode_standard(dt=0.0)
        hoomd.md.integrate.nve(group=hoomd.group.all())
        hoomd.run(1)
        snap = self.s.take_snapshot(bonds=True)
        self.assertAlmostEqual(1, snap.bonds.N)
        np.testing.assert_array_almost_equal([1,2], snap.bonds.group[0])

        # after this bond 1-2 is formed, there shouldn't be a second one formed
        # e.g. a dublicate, even when the simulation continues to run
        hoomd.run(10)
        snap = self.s.take_snapshot(bonds=True)
        self.assertAlmostEqual(1, snap.bonds.N)
        np.testing.assert_array_almost_equal([1,2], snap.bonds.group[0])

    def test_no_bond_formation_outside_rcut(self):
        # put particle 1 out of range, no bonds should be formed
        self.s.particles[1].position=(1.1,5,0)
        hoomd.md.integrate.mode_standard(dt=0.0)
        hoomd.md.integrate.nve(group=hoomd.group.all())

        hoomd.run(1)
        snap = self.s.take_snapshot(bonds=True)
        self.assertAlmostEqual(0, snap.bonds.N)


    def tearDown(self):
        del self.s, self.u
        hoomd.context.initialize()


class update_dynamic_bond_tests_one_group(unittest.TestCase):
    def setUp(self):
        snap = hoomd.data.make_snapshot(N=6, box=hoomd.data.boxdim(L=20),
                                        particle_types=['A','B'],
                                        bond_types=['bond'])
        if hoomd.comm.get_rank() == 0:
            snap.particles.position[:,0] = (0,1,2,3,4,6)
            snap.particles.position[:,1] = (0,0,0,0,0,0)
            snap.particles.position[:,2] = (0,0,0,0,0,0)
            # dynamic bond operates on groups, so typeids should not matter at all
            snap.particles.typeid[:] = [0,0,1,1,0,0]

        self.s = hoomd.init.read_snapshot(snap)
        self.nl = hoomd.md.nlist.cell()

        self.group_3 = hoomd.group.tag_list(name="a", tags = [0,1,2,3,4,5])
        self.u = azplugins.update.dynamic_bond(nlist=self.nl,
                                               r_cut=1.1,
                                               bond_type='bond',
                                               group_1=self.group_3,
                                               group_2=self.group_3,
                                               max_bonds_1=2,
                                               max_bonds_2=2)

    def test_form_bond(self):
        # test bond formation. All are in the same group, so bonds should be formed
        # between 0-1, 1-2, 2-3, and 3-4 (but not 5, too far away)
        hoomd.md.integrate.mode_standard(dt=0.0)
        hoomd.md.integrate.nve(group=hoomd.group.all())
        hoomd.run(1)
        snap = self.s.take_snapshot(bonds=True)
        self.assertAlmostEqual(4, snap.bonds.N)
        np.testing.assert_array_almost_equal([0,1], snap.bonds.group[0])
        np.testing.assert_array_almost_equal([1,2], snap.bonds.group[1])
        np.testing.assert_array_almost_equal([2,3], snap.bonds.group[2])
        np.testing.assert_array_almost_equal([3,4], snap.bonds.group[3])

    def test_no_bond_formation_too_many_bonds(self):
        # same as before, we will have formed bonds 0-1-2-3-4 in the first step
        hoomd.md.integrate.mode_standard(dt=0.0)
        hoomd.md.integrate.nve(group=hoomd.group.all())
        hoomd.run(1)
        snap = self.s.take_snapshot(bonds=True)
        self.assertAlmostEqual(4, snap.bonds.N)
        np.testing.assert_array_almost_equal([0,1], snap.bonds.group[0])
        np.testing.assert_array_almost_equal([1,2], snap.bonds.group[1])
        np.testing.assert_array_almost_equal([2,3], snap.bonds.group[2])
        np.testing.assert_array_almost_equal([3,4], snap.bonds.group[3])

        # now put particle 5 in range of partice 2 and 3, but no new bonds should
        # be formed since max_bonds_1=max_bonds_2=2 and that would be the third
        # bond on those particles
        self.s.particles[5].position=(2.5,0,0)

        hoomd.run(1)
        snap = self.s.take_snapshot(bonds=True)
        self.assertAlmostEqual(4, snap.bonds.N)
        np.testing.assert_array_almost_equal([0,1], snap.bonds.group[0])
        np.testing.assert_array_almost_equal([1,2], snap.bonds.group[1])
        np.testing.assert_array_almost_equal([2,3], snap.bonds.group[2])
        np.testing.assert_array_almost_equal([3,4], snap.bonds.group[3])

    def test_bond_formation_too_many_bonds(self):
        # now put particle 5 in range of partice 2 and 3, so in principle the bonds
        # 0-1, 1-2 ,2-3, 3-4 (all length 1), 2-5 and 3-5 (length 0.5) can be formed.
        # The all_possible_bonds array  is sorted by bond distance, so bonds 2-5
        # and 3-5 are formed first. Then  in order of index 0-1, and 1-2.
        # Particle 2 now has two bonds, so 2-3 can't be formed, but 3-4 can.

        self.s.particles[5].position=(2.5,0,0)
        snap = self.s.take_snapshot(bonds=True)
        hoomd.run(1)
        snap = self.s.take_snapshot(bonds=True)
        print(snap.bonds.group)
        self.assertAlmostEqual(5, snap.bonds.N)

        np.testing.assert_array_almost_equal([2,5], snap.bonds.group[0])
        np.testing.assert_array_almost_equal([3,5], snap.bonds.group[1])
        np.testing.assert_array_almost_equal([0,1], snap.bonds.group[2])
        np.testing.assert_array_almost_equal([1,2], snap.bonds.group[3])
        np.testing.assert_array_almost_equal([3,4], snap.bonds.group[4])

    def test_group_partial_overlap(self):
        self.group_1 = hoomd.group.tag_list(name="a", tags = [0,1,2,3])
        self.group_2 = hoomd.group.tag_list(name="b", tags = [3,4,5])
        with self.assertRaises(RuntimeError):
            azplugins.update.dynamic_bond(nlist=self.nl,
                                     r_cut=1.0,
                                     bond_type='bond',
                                     group_1=self.group_1,
                                     group_2=self.group_2,
                                     max_bonds_1=1,
                                     max_bonds_2=2)

    def tearDown(self):
        del self.s, self.u
        hoomd.context.initialize()



if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
