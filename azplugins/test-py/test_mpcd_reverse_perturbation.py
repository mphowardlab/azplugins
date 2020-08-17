# Copyright (c) 2018-2020, Michael P. Howard
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: astatt

import hoomd
from hoomd import md
from hoomd import mpcd

hoomd.context.initialize()
try:
    from hoomd import azplugins
    import hoomd.azplugins.mpcd
except ImportError:
    import azplugins
    import azplugins.mpcd
import unittest

# tests for azplugins.flow.reverse_pertubation
class updater_reverse_perturbation(unittest.TestCase):
    def setUp(self):
        snapshot = hoomd.data.make_snapshot(N=1, particle_types=['A'],box=hoomd.data.boxdim(L=20))
        system1 = hoomd.init.read_snapshot(snapshot)
        snap = mpcd.data.make_snapshot(N=1)
        snap.particles.types = ['A']
        snap.particles.position[0] = (0,0,0)
        self.s = mpcd.init.read_snapshot(snap)


    # tests basic creation of the updater
    def test(self):
        field = azplugins.mpcd.reverse_perturbation(width=1.0,Nswap=1,period=100,phase=-1,target_momentum=0.5)

    # test setting parameters
    def test_parameters(self):
        field = azplugins.mpcd.reverse_perturbation(width=1.0,Nswap=1,period=100,phase=-1,target_momentum=0.5)
        field.set_params(Nswap=4)
        field.set_params(width=3)
        # cannot set a width > L/2
        with self.assertRaises(RuntimeError):
            field.set_params(width=22.0)
            hoomd.run(1)
        # cannot set a Nswap < 0
        with self.assertRaises(ValueError):
            field.set_params(Nswap=-5)
        # cannot set a period < 0
        with self.assertRaises(TypeError):
            field.set_params(period =-1)
        # cannot set a target_momentum < 0
        with self.assertRaises(ValueError):
            field.set_params(target_momentum=-1)
        # cannot set slab distance < 0
        with self.assertRaises(ValueError):
            field.set_params(H=-1)

    def tearDown(self):
        hoomd.context.initialize()

class updater_reverse_perturbation_swap(unittest.TestCase):
    def setUp(self):
        snapshot =  hoomd.data.make_snapshot(N=1, particle_types=['A'],box=hoomd.data.boxdim(L=20))
        system1 = hoomd.init.read_snapshot(snapshot)
        snap = mpcd.data.make_snapshot(N=5)
        snap.particles.types = ['A']
        snap.particles.velocity[0] = (0.1,0.0,0.0)
        snap.particles.velocity[1] = (0.8,0.0,0.0)
        snap.particles.velocity[2] = (-0.1,0.0,0.0)
        snap.particles.velocity[3] = (-0.5,0.0,0.0)
        snap.particles.velocity[4] = (-0.09,0.0,0.0)

        snap.particles.position[0] = (0.0,0.0,-5.3)
        snap.particles.position[1] = (0.0,1.0,-5.3)
        snap.particles.position[2] = (0.0,0.0,5.3)
        snap.particles.position[3] = (0.0,1.0,5.3)
        snap.particles.position[4] = (0.0,0.0,5.3)

        self.s = mpcd.init.read_snapshot(snap)

    def test_resize_Nswap(self):
        field = azplugins.mpcd.reverse_perturbation(width=0.2,Nswap=1,period=1,target_momentum=2)
        hoomd.run(1)
        self.assertAlmostEqual(field.Nswap,1)
        field = azplugins.mpcd.reverse_perturbation(width=0.2,Nswap=356,period=1,target_momentum=2)
        hoomd.run(1)
        self.assertAlmostEqual(field.Nswap,356)

    def test_set_slab_distance(self):
        field = azplugins.mpcd.reverse_perturbation(width=0.2,Nswap=1,period=1,target_momentum=2,H=3)
        hoomd.run(1)
        self.assertAlmostEqual(field.distance,3)

    def test_simple_swap(self):
        # swap velocities of particle 1/3. Don't swap 0/2 - Nswap is too small
        field = azplugins.mpcd.reverse_perturbation(width=1.0,Nswap=1,period=1,phase=0,target_momentum=0.8)
        hoomd.run(1)
        snap_out = self.s.take_snapshot()
        self.assertAlmostEqual(snap_out.particles.velocity[0][0],0.1)
        self.assertAlmostEqual(snap_out.particles.velocity[2][0],-0.1)
        self.assertAlmostEqual(snap_out.particles.velocity[1][0],-0.5)
        self.assertAlmostEqual(snap_out.particles.velocity[3][0],0.8)

    def test_swap_outside_slab(self):
        # swap no velocities because slab distance is changed - no particles are in slab at +/- 3
        field = azplugins.mpcd.reverse_perturbation(H=3,width=1.0,Nswap=1,period=1,phase=0,target_momentum=0.8)
        hoomd.run(1)
        snap_out = self.s.take_snapshot()
        self.assertAlmostEqual(snap_out.particles.velocity[0][0],0.1)
        self.assertAlmostEqual(snap_out.particles.velocity[2][0],-0.1)
        self.assertAlmostEqual(snap_out.particles.velocity[1][0],0.8)
        self.assertAlmostEqual(snap_out.particles.velocity[3][0],-0.5)

    def test_swap_changed_slab(self):
        # swap velocites of particle 1/3, shifted particle and slab positions
        snap_in = self.s.take_snapshot()
        snap_in.particles.position[1]=(0,3,-2.0)
        snap_in.particles.position[3]=(1,2,+2.0)
        snap_in.particles.velocity[1]=(0.5,0,0)
        snap_in.particles.velocity[3]=(-0.5,0,0)
        self.s.restore_snapshot(snap_in)
        field = azplugins.mpcd.reverse_perturbation(H=2,width=1.0,Nswap=1,period=1,phase=0,target_momentum=0.8)
        hoomd.run(1)
        self.assertAlmostEqual(field.distance,2)
        snap_out = self.s.take_snapshot()
        self.assertAlmostEqual(snap_out.particles.position[1][2],-2.0)
        self.assertAlmostEqual(snap_out.particles.velocity[1][0],-0.5)
        self.assertAlmostEqual(snap_out.particles.velocity[3][0],0.5)


    def test_empty_top_slab(self):
        snap_in = self.s.take_snapshot()
        snap_in.particles.position[2]=(0,3,-1.0)
        snap_in.particles.position[3]=(1,2,-1.0)
        snap_in.particles.position[4]=(2,-2,-1.0)
        self.s.restore_snapshot(snap_in)
        field = azplugins.mpcd.reverse_perturbation(width=1.0,Nswap=10,period=1,target_momentum=2)
        hoomd.run(1)
        snap_out = self.s.take_snapshot()
        self.assertAlmostEqual(snap_out.particles.velocity[0][0],0.1)
        self.assertAlmostEqual(snap_out.particles.velocity[1][0],0.8)
        self.assertAlmostEqual(snap_out.particles.velocity[2][0],-0.1)
        self.assertAlmostEqual(snap_out.particles.velocity[3][0],-0.5)

    def test_empty_bottom_slab(self):
        snap_in = self.s.take_snapshot()
        snap_in.particles.position[0]=(0,3.0,1.0)
        snap_in.particles.position[1]=(1,2.0,1.0)
        self.s.restore_snapshot(snap_in)

        field = azplugins.mpcd.reverse_perturbation(width=1.0,Nswap=10,period=1,target_momentum=2)
        hoomd.run(1)
        snap_out = self.s.take_snapshot()
        self.assertAlmostEqual(snap_out.particles.velocity[0][0],0.1)
        self.assertAlmostEqual(snap_out.particles.velocity[1][0],0.8)
        self.assertAlmostEqual(snap_out.particles.velocity[2][0],-0.1)
        self.assertAlmostEqual(snap_out.particles.velocity[3][0],-0.5)

    def test_simple_swap_outside_slab(self):
        # don't swap anything - all particles are outside of slabs
        field = azplugins.mpcd.reverse_perturbation(width=0.2,Nswap=100,period=1,target_momentum=2)
        hoomd.run(1)
        snap_out = self.s.take_snapshot()
        self.assertAlmostEqual(snap_out.particles.velocity[0][0],0.1)
        self.assertAlmostEqual(snap_out.particles.velocity[1][0],0.8)
        self.assertAlmostEqual(snap_out.particles.velocity[2][0],-0.1)
        self.assertAlmostEqual(snap_out.particles.velocity[3][0],-0.5)

    def test_simple_swap_all(self):
        # swap velocities of particle 1/3, and 0/2
        field = azplugins.mpcd.reverse_perturbation(width=1.0,Nswap=100,period=1,target_momentum=2)
        hoomd.run(1)
        snap_out = self.s.take_snapshot()
        self.assertAlmostEqual(snap_out.particles.velocity[0][0],-0.1)
        self.assertAlmostEqual(snap_out.particles.velocity[1][0],-0.5)
        self.assertAlmostEqual(snap_out.particles.velocity[2][0],0.1)
        self.assertAlmostEqual(snap_out.particles.velocity[3][0],0.8)

    def test_swap_target_momentum(self):
        # swap velocities of particle 0/2 (closer to V) and not 1/3
        field = azplugins.mpcd.reverse_perturbation(width=1.0,Nswap=1,period=1,target_momentum=0.1)
        hoomd.run(1)
        snap_out = self.s.take_snapshot()
        self.assertAlmostEqual(snap_out.particles.velocity[0][0],-0.1)
        self.assertAlmostEqual(snap_out.particles.velocity[1][0],0.8)
        self.assertAlmostEqual(snap_out.particles.velocity[2][0],0.1)
        self.assertAlmostEqual(snap_out.particles.velocity[3][0],-0.5)

    def tearDown(self):
        hoomd.context.initialize()


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
