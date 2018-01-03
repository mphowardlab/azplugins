# Copyright (c) 2016-2018, Panagiotopoulos Group, Princeton University
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: astatt

from hoomd import *
from hoomd import md
context.initialize()
try:
    from hoomd import azplugins
except ImportError:
    import azplugins
import unittest

# tests for azplugins.flow.reverse_perturbation
class updater_reverse_perturbation(unittest.TestCase):
    def setUp(self):
        snap = data.make_snapshot(N=3, box=data.boxdim(L=20), particle_types=['A'])
        snap.particles.velocity[:] = (0.1,0.1,0.1)
        self.s = init.read_snapshot(snap)

    # tests basic creation of the updater
    def test(self):
        field = azplugins.flow.reverse_perturbation(group.all(),width=1.0,Nswap=1,period=100,phase=-1,target_momentum=1)

    # test setting parameters
    def test_parameters(self):
        field = azplugins.flow.reverse_perturbation(group.all(),width=1.0,Nswap=1,period=100,phase=-1,target_momentum=1)
        field.set_params(Nswap=4)
        field.set_params(width=3)
        # cannot set a width > L/2
        with self.assertRaises(RuntimeError):
            field.set_params(width=22.0)
            run(1)
        # cannot set a Nswap < 0
        with self.assertRaises(ValueError):
            field.set_params(Nswap=-5)
         # cannot set a P < 0
        with self.assertRaises(ValueError):
            field.set_params(target_momentum=-5)
        # cannot set a period < 0
        with self.assertRaises(TypeError):
            field.set_params(period =-1)

    def test_box_change(self):
        # shrink box smaller than 2*width, which should trigger signal to check
        # box and cause a runtime error
        field = azplugins.flow.reverse_perturbation(group.all(),Nswap=1,width=3.45,period=1,target_momentum=1)
        update.box_resize(L=5.5, period=None)
        with self.assertRaises(RuntimeError):
            run(1)

    def tearDown(self):
        context.initialize()

class updater_reverse_perturbation_swap(unittest.TestCase):
    def setUp(self):
        snap = data.make_snapshot(N=5, box=data.boxdim(L=20), particle_types=['A','S'])
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

        snap.particles.typeid[4] = 1
        self.s = init.read_snapshot(snap)
        self.group0 = group.type(type='A')
        self.group1 = group.type(type='S')

    def test_resize_Nswap(self):
        field = azplugins.flow.reverse_perturbation(self.group0,width=0.2,Nswap=1,period=1,target_momentum=1)
        run(1)
        self.assertAlmostEqual(field.Nswap,1)
        field = azplugins.flow.reverse_perturbation(self.group0,width=0.2,Nswap=356,period=1,target_momentum=1)
        run(1)
        self.assertAlmostEqual(field.Nswap,356)

    def test_wrong_group(self):
        # don't swap anything - paricles belong to wrong group
        field = azplugins.flow.reverse_perturbation(self.group1,width=0.2,Nswap=100,period=1,target_momentum=1)
        run(1)
        self.assertAlmostEqual(self.s.particles[0].velocity[0],0.1)
        self.assertAlmostEqual(self.s.particles[1].velocity[0],0.8)
        self.assertAlmostEqual(self.s.particles[2].velocity[0],-0.1)
        self.assertAlmostEqual(self.s.particles[3].velocity[0],-0.5)

    def test_simple_swap(self):
        # swap velocities of particle 1/3. Don't swap 0/2 - Nswap is too small
        field = azplugins.flow.reverse_perturbation(group.all(),width=1.0,Nswap=1,period=1,phase=0,target_momentum=1)
        run(1)
        self.assertAlmostEqual(self.s.particles[0].mass,1.0)

        self.assertAlmostEqual(self.s.particles[0].velocity[0],0.1)
        self.assertAlmostEqual(self.s.particles[2].velocity[0],-0.1)
        self.assertAlmostEqual(self.s.particles[1].velocity[0],-0.5)
        self.assertAlmostEqual(self.s.particles[3].velocity[0],0.8)

    def test_group(self):
        # swap all
        field = azplugins.flow.reverse_perturbation(self.group0,width=2.0,Nswap=100,period=1,target_momentum=1)
        run(1)
        self.assertAlmostEqual(self.s.particles[1].velocity[0],-0.5)
        self.assertAlmostEqual(self.s.particles[3].velocity[0],0.8)
        self.assertAlmostEqual(self.s.particles[0].velocity[0],-0.1)
        self.assertAlmostEqual(self.s.particles[2].velocity[0],0.1)

    def test_swap_momentum_not_velocity(self):
        # swap 1/2
        self.s.particles[0].mass=1.0
        self.s.particles[1].mass=1.0
        self.s.particles[2].mass=3.0
        self.s.particles[3].mass=1.0
        self.s.particles[0].velocity =(1.0,0,0)
        self.s.particles[1].velocity =(2.0,0,0)
        self.s.particles[2].velocity =(-0.1,0,0)
        self.s.particles[3].velocity =(-0.2,0,0)
        field = azplugins.flow.reverse_perturbation(group.all(),width=2.0,Nswap=1,period=1,target_momentum=10)
        run(1)
        self.assertAlmostEqual(self.s.particles[2].mass,3.0)
        self.assertAlmostEqual(self.s.particles[0].mass,1.0)
        self.assertAlmostEqual(self.s.particles[3].velocity[0],-0.2)
        self.assertAlmostEqual(self.s.particles[0].velocity[0],1.0)

        self.assertAlmostEqual(self.s.particles[1].velocity[0],-0.3)
        self.assertAlmostEqual(self.s.particles[2].velocity[0],0.666666666)

    def test_empty_top_slab(self):
        self.s.particles[2].position=(0,3,-1.0)
        self.s.particles[3].position=(1,2,-1.0)
        self.s.particles[4].position=(2,-2,-1.0)

        field = azplugins.flow.reverse_perturbation(group.all(),width=1.0,Nswap=10,period=1,target_momentum=1)
        run(1)

        self.assertAlmostEqual(self.s.particles[0].velocity[0],0.1)
        self.assertAlmostEqual(self.s.particles[1].velocity[0],0.8)
        self.assertAlmostEqual(self.s.particles[2].velocity[0],-0.1)
        self.assertAlmostEqual(self.s.particles[3].velocity[0],-0.5)

    def test_empty_bottom_slab(self):
        self.s.particles[0].position=(0,3.0,1.0)
        self.s.particles[1].position=(1,2.0,1.0)

        field = azplugins.flow.reverse_perturbation(group.all(),width=1.0,Nswap=10,period=1,target_momentum=1)
        run(1)

        self.assertAlmostEqual(self.s.particles[0].velocity[0],0.1)
        self.assertAlmostEqual(self.s.particles[1].velocity[0],0.8)
        self.assertAlmostEqual(self.s.particles[2].velocity[0],-0.1)
        self.assertAlmostEqual(self.s.particles[3].velocity[0],-0.5)

    def test_simple_swap_outside_slab(self):
        # don't swap anything - all particles are outside of slabs
        field = azplugins.flow.reverse_perturbation(group.all(),width=0.2,Nswap=100,period=1,target_momentum=1)
        run(1)
        self.assertAlmostEqual(self.s.particles[0].velocity[0],0.1)
        self.assertAlmostEqual(self.s.particles[1].velocity[0],0.8)
        self.assertAlmostEqual(self.s.particles[2].velocity[0],-0.1)
        self.assertAlmostEqual(self.s.particles[3].velocity[0],-0.5)

    def test_simple_swap_all(self):
        # swap velocities of particles 1/3, and 0/2
        field = azplugins.flow.reverse_perturbation(group.all(),width=1.0,Nswap=100,period=1,target_momentum=1)
        run(1)
        self.assertAlmostEqual(self.s.particles[1].velocity[0],-0.5)
        self.assertAlmostEqual(self.s.particles[3].velocity[0],0.8)
        self.assertAlmostEqual(self.s.particles[0].velocity[0],-0.1)
        self.assertAlmostEqual(self.s.particles[2].velocity[0],0.1)

    def test_target_momentum_swap(self):
        # swap velocities of particles 0/2 (closer to P) and not 1/3
        field = azplugins.flow.reverse_perturbation(group.all(),width=1.0,Nswap=1,period=1,target_momentum=0.1)
        run(1)
        self.assertAlmostEqual(self.s.particles[1].velocity[0],0.8)
        self.assertAlmostEqual(self.s.particles[3].velocity[0],-0.5)
        self.assertAlmostEqual(self.s.particles[0].velocity[0],-0.1)
        self.assertAlmostEqual(self.s.particles[2].velocity[0],0.1)


    def tearDown(self):
        context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
