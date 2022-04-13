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

# azplugins.evaporate.implicit
class evaporate_implicit_tests(unittest.TestCase):
    def setUp(self):
        snap = hoomd.data.make_snapshot(N=100, box=hoomd.data.boxdim(L=20), particle_types=['A'])
        self.s = hoomd.init.read_snapshot(snap)
        self.interf = hoomd.variant.linear_interp([[0,9.],[1e6,5.]],zero=0)

    # basic test of creation
    def test_basic(self):
        evap = azplugins.evaporate.implicit(interface=self.interf)
        evap.force_coeff.set('A', k=50.0, offset=0.1, g=50.0*0.5, cutoff=0.5)
        evap.update_coeffs()

    # test missing k
    def test_set_missing_k(self):
        evap = azplugins.evaporate.implicit(interface=self.interf)
        evap.force_coeff.set('A', offset=0.1, g=50.0*0.5, cutoff=0.5)
        self.assertRaises(RuntimeError, evap.update_coeffs)

    # test missing sigma
    def test_set_missing_g(self):
        evap = azplugins.evaporate.implicit(interface=self.interf)
        evap.force_coeff.set('A', k=50.0, offset=0.1, cutoff=0.5)
        self.assertRaises(RuntimeError, evap.update_coeffs)

    # test missing cutoff
    def test_set_missing_cutoff(self):
        evap = azplugins.evaporate.implicit(interface=self.interf)
        evap.force_coeff.set('A', k=50.0, offset=0.1, g=50.0*0.5)
        self.assertRaises(RuntimeError, evap.update_coeffs)

    # test non-numeric values of cutoff
    def test_cutoff_none(self):
        evap = azplugins.evaporate.implicit(interface=self.interf)
        evap.force_coeff.set('A', k=50.0, g=50.0*0.5, cutoff=False)
        evap.update_coeffs()
        evap.force_coeff.set('A', k=50.0, g=50.0*0.5, cutoff=None)
        evap.update_coeffs()

    # test missing coefficients
    def test_missing_A(self):
        evap = azplugins.evaporate.implicit(interface=self.interf)
        self.assertRaises(RuntimeError, evap.update_coeffs)

    # test default coefficients
    def test_default_offset(self):
        evap = azplugins.evaporate.implicit(interface=self.interf)
        evap.force_coeff.set('A', k=50.0, g=50.0*0.5, cutoff=0.5)
        evap.update_coeffs()

    # test coeff list
    def test_coeff_list(self):
        evap = azplugins.evaporate.implicit(interface=self.interf)
        evap.force_coeff.set(['A', 'B'], k=50.0, g=50.0*0.5, cutoff=0.5)
        evap.update_coeffs()

    # test adding types
    def test_type_add(self):
        evap = azplugins.evaporate.implicit(interface=self.interf)
        evap.force_coeff.set('A', k=50.0, g=50.0*0.5, cutoff=0.5)
        self.s.particles.types.add('B')
        self.assertRaises(RuntimeError, evap.update_coeffs)
        evap.force_coeff.set('B', k=50.0*2**2, g=50.0*2**3/2., cutoff=1.0)
        evap.update_coeffs()

    # test using scalar for variant
    def test_scalar_interface(self):
        evap = azplugins.evaporate.implicit(interface=5.0)

    # test geometry is settable and checked
    def test_geometry(self):
        azplugins.evaporate.implicit(interface=5.0, geometry='film')
        azplugins.evaporate.implicit(interface=5.0, geometry='droplet')
        with self.assertRaises(ValueError):
            azplugins.evaporate.implicit(interface=5.0, geometry='foobar')

    def tearDown(self):
        del self.s, self.interf
        hoomd.context.initialize()

# test the validity of the potential
class evaporate_implicit_film_potential_tests(unittest.TestCase):
    def setUp(self):
        if hoomd.comm.get_num_ranks() > 1:
            hoomd.comm.decomposition(nx=2, ny=1, nz=1)

        snap = hoomd.data.make_snapshot(N=4, box=hoomd.data.boxdim(L=20),particle_types=['A','B'])
        if hoomd.comm.get_rank() == 0:
            snap.particles.position[0] = (1,1,4.6)
            snap.particles.position[1] = (-1,1,5.4)
            snap.particles.position[2] = (1,-1,5.6)
            snap.particles.position[3] = (-1,-1,6.6)
            snap.particles.typeid[:] = (0,1,0,0)
        hoomd.init.read_snapshot(snap)

        # integrator with zero timestep to compute forces
        md.integrate.mode_standard(dt=0)
        md.integrate.nve(group = hoomd.group.all())

    # test the calculation of force and potential
    def test_potential(self):
        evap = azplugins.evaporate.implicit(interface=hoomd.variant.linear_interp([[0,5.0],[1,5.0],[2,4.0],[3,4.0]]), geometry='film')
        kA = 50.0
        dB = 2.0
        kB = kA*dB**2
        evap.force_coeff.set('A', k=kA, offset=0.1, g=kA/2., cutoff=0.5)
        evap.force_coeff.set('B', k=kB, offset=-0.1, g=kB*dB/2., cutoff=dB/2.)

        # in the first run step, the interface stays at 5.0 in both verlet steps
        hoomd.run(1)
        # particle 0 is outside the interaction range
        self.assertAlmostEqual(evap.forces[0].energy, 0)
        f0 = evap.forces[0].force
        self.assertAlmostEqual(f0[0], 0)
        self.assertAlmostEqual(f0[1], 0)
        self.assertAlmostEqual(f0[2], 0)

        # particle 1 (type B) is experiencing the harmonic potential
        self.assertAlmostEqual(evap.forces[1].energy, 0.5*kB*0.5**2, 4)
        f1 = evap.forces[1].force
        self.assertAlmostEqual(f1[0], 0)
        self.assertAlmostEqual(f1[1], 0)
        self.assertAlmostEqual(f1[2], -kB*0.5, 4)

        # particle 2 (type A) is also experiencing the harmonic potential
        self.assertAlmostEqual(evap.forces[2].energy, 0.5*kA*0.5**2, 4)
        f2 = evap.forces[2].force
        self.assertAlmostEqual(f2[0], 0)
        self.assertAlmostEqual(f2[1], 0)
        self.assertAlmostEqual(f2[2], -kA*0.5, 4)

        # particle 3 (type A) is experiencing the gravitational force
        self.assertAlmostEqual(evap.forces[3].energy, 0.5*kA*0.5**2 + (kA/2.)*1.0, 4)
        f3 = evap.forces[3].force
        self.assertAlmostEqual(f3[0], 0)
        self.assertAlmostEqual(f3[1], 0)
        self.assertAlmostEqual(f3[2], -kA/2., 4)

        # disable B interactions for the next test
        evap.force_coeff.set('B', cutoff=False)
        # advance the simulation two steps so that now the interface is at 4.0
        # in both verlet steps
        hoomd.run(2)
        # particle 0 is now inside the harmonic region
        self.assertAlmostEqual(evap.forces[0].energy, 0.5*kA*0.5**2, 4)
        f0 = evap.forces[0].force
        self.assertAlmostEqual(f0[0], 0)
        self.assertAlmostEqual(f0[1], 0)
        self.assertAlmostEqual(f0[2], -kA*0.5, 4)

        # particle 1 (type B) should now be ignored by the cutoff
        self.assertAlmostEqual(evap.forces[1].energy, 0)
        f1 = evap.forces[1].force
        self.assertAlmostEqual(f1[0], 0)
        self.assertAlmostEqual(f1[1], 0)
        self.assertAlmostEqual(f1[2], 0)

        # particle 2 (type A) is also experiencing the gravitational force now
        self.assertAlmostEqual(evap.forces[2].energy, 0.5*kA*0.5**2 + (kA/2.)*1.0, 4)
        f2 = evap.forces[2].force
        self.assertAlmostEqual(f2[0], 0)
        self.assertAlmostEqual(f2[1], 0)
        self.assertAlmostEqual(f2[2], -kA*0.5, 4)

        # particle 3 (type A) is experiencing the gravitational force
        self.assertAlmostEqual(evap.forces[3].energy, 0.5*kA*0.5**2 + (kA/2.)*2.0, 4)
        f3 = evap.forces[3].force
        self.assertAlmostEqual(f3[0], 0)
        self.assertAlmostEqual(f3[1], 0)
        self.assertAlmostEqual(f3[2], -kA/2., 4)

    def test_box_outside_error(self):
        evap = azplugins.evaporate.implicit(interface=11.0, geometry='film')
        evap.force_coeff.set('A', k=0.0, g=0.0, cutoff=False)
        evap.force_coeff.set('B', k=0.0, g=0.0, cutoff=False)

        with self.assertRaises(RuntimeError):
            hoomd.run(1)

    def test_log_warning(self):
        evap = azplugins.evaporate.implicit(interface=5.0, geometry='film')
        evap.force_coeff.set('A', k=1.0, g=1.0, cutoff=1.0)
        evap.force_coeff.set('B', k=1.0, g=1.0, cutoff=1.0)

        hoomd.analyze.log(filename=None, quantities=['pressure'], period=1)
        hoomd.run(1)
        hoomd.run(1)

    def tearDown(self):
        hoomd.context.initialize()

# test the validity of the potential
class evaporate_implicit_droplet_potential_tests(unittest.TestCase):
    def setUp(self):
        if hoomd.comm.get_num_ranks() > 1:
            hoomd.comm.decomposition(nx=2, ny=1, nz=1)

        snap = hoomd.data.make_snapshot(N=4, box=hoomd.data.boxdim(L=20),particle_types=['A','B'])
        if hoomd.comm.get_rank() == 0:
            snap.particles.position[0] = (0,0,4.6)
            snap.particles.position[1] = (0,0,-5.4)
            snap.particles.position[2] = (0,5.6,0)
            snap.particles.position[3] = (6.6,0,0)
            snap.particles.typeid[:] = (0,1,0,0)
        hoomd.init.read_snapshot(snap)

        # integrator with zero timestep to compute forces
        md.integrate.mode_standard(dt=0)
        md.integrate.nve(group = hoomd.group.all())

    # test the calculation of force and potential
    def test_potential(self):
        evap = azplugins.evaporate.implicit(interface=hoomd.variant.linear_interp([[0,5.0],[1,5.0],[2,4.0],[3,4.0]]), geometry='droplet')
        kA = 50.0
        dB = 2.0
        kB = kA*dB**2
        evap.force_coeff.set('A', k=kA, offset=0.1, g=kA/2., cutoff=0.5)
        evap.force_coeff.set('B', k=kB, offset=-0.1, g=kB*dB/2., cutoff=dB/2.)

        # in the first run step, the interface stays at 5.0 in both verlet steps
        hoomd.run(1)
        # particle 0 is outside the interaction range
        self.assertAlmostEqual(evap.forces[0].energy, 0)
        f0 = evap.forces[0].force
        self.assertAlmostEqual(f0[0], 0)
        self.assertAlmostEqual(f0[1], 0)
        self.assertAlmostEqual(f0[2], 0)

        # particle 1 (type B) is experiencing the harmonic potential in +z
        self.assertAlmostEqual(evap.forces[1].energy, 0.5*kB*0.5**2, 4)
        f1 = evap.forces[1].force
        self.assertAlmostEqual(f1[0], 0)
        self.assertAlmostEqual(f1[1], 0)
        self.assertAlmostEqual(f1[2], kB*0.5, 4)

        # particle 2 (type A) is also experiencing the harmonic potential in -y
        self.assertAlmostEqual(evap.forces[2].energy, 0.5*kA*0.5**2, 4)
        f2 = evap.forces[2].force
        self.assertAlmostEqual(f2[0], 0)
        self.assertAlmostEqual(f2[1], -kA*0.5, 4)
        self.assertAlmostEqual(f2[2], 0)

        # particle 3 (type A) is experiencing the gravitational force in -x
        self.assertAlmostEqual(evap.forces[3].energy, 0.5*kA*0.5**2 + (kA/2.)*1.0, 4)
        f3 = evap.forces[3].force
        self.assertAlmostEqual(f3[0], -kA/2., 4)
        self.assertAlmostEqual(f3[1], 0)
        self.assertAlmostEqual(f3[2], 0)

        # disable B interactions for the next test
        evap.force_coeff.set('B', cutoff=False)
        # advance the simulation two steps so that now the interface is at 4.0
        # in both verlet steps
        hoomd.run(2)
        # particle 0 is now inside the harmonic region, -x
        self.assertAlmostEqual(evap.forces[0].energy, 0.5*kA*0.5**2, 4)
        f0 = evap.forces[0].force
        self.assertAlmostEqual(f0[0], 0)
        self.assertAlmostEqual(f0[1], 0)
        self.assertAlmostEqual(f0[2], -kA*0.5, 4)

        # particle 1 (type B) should now be ignored by the cutoff
        self.assertAlmostEqual(evap.forces[1].energy, 0)
        f1 = evap.forces[1].force
        self.assertAlmostEqual(f1[0], 0)
        self.assertAlmostEqual(f1[1], 0)
        self.assertAlmostEqual(f1[2], 0)

        # particle 2 (type A) is also experiencing the gravitational force now
        self.assertAlmostEqual(evap.forces[2].energy, 0.5*kA*0.5**2 + (kA/2.)*1.0, 4)
        f2 = evap.forces[2].force
        self.assertAlmostEqual(f2[0], 0)
        self.assertAlmostEqual(f2[1], -kA*0.5, 4)
        self.assertAlmostEqual(f2[2], 0)

        # particle 3 (type A) is experiencing the gravitational force
        self.assertAlmostEqual(evap.forces[3].energy, 0.5*kA*0.5**2 + (kA/2.)*2.0, 4)
        f3 = evap.forces[3].force
        self.assertAlmostEqual(f3[0], -kA/2., 4)
        self.assertAlmostEqual(f3[1], 0)
        self.assertAlmostEqual(f3[2], 0)

    def test_box_outside_error(self):
        evap = azplugins.evaporate.implicit(interface=11.0, geometry='droplet')
        evap.force_coeff.set('A', k=0.0, g=0.0, cutoff=False)
        evap.force_coeff.set('B', k=0.0, g=0.0, cutoff=False)

        with self.assertRaises(RuntimeError):
            hoomd.run(1)

    def tearDown(self):
        hoomd.context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
