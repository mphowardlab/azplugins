# Copyright (c) 2019-2020, Antonia Statt
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: astatt / Everyone is free to add additional tutorials

import numpy as np
import sys
sys.path.insert(0,'/Users/statt/programs/azplugins-0.7.0')
sys.path.insert(0,'/Users/statt/programs/hoomd-2.6.0')
import hoomd
from hoomd import md
from hoomd import data
import azplugins

L = 10
kT = 0.772
rho = 0.849
sigma = 1.0

hoomd.context.initialize()
hoomd.context.SimulationContext()

snapshot = hoomd.data.make_snapshot(N=0,box=data.boxdim(L=L),particle_types=['A'])
system = hoomd.init.read_snapshot(snapshot)

N = int(rho*system.box.get_volume())
snapshot.particles.resize(N)

l = np.arange(-L/2.,+L/2.,sigma)
positions_square_lattice = np.asarray(np.meshgrid(l, l, l)).reshape(3,-1).transpose()
np.random.shuffle(positions_square_lattice)

snapshot.particles.position[:] = positions_square_lattice[:N]
snapshot.particles.velocity[:] = np.random.normal(0, np.sqrt(kT), (N,3))
snapshot.particles.velocity[:] -= np.average(snapshot.particles.velocity,axis=0)

system.restore_snapshot(snapshot)

nl = hoomd.md.nlist.cell()
lj = hoomd.md.pair.lj(r_cut=3.0, nlist=nl)
lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=sigma)

all = hoomd.group.all()

class measure_fluid_properties:
    def __init__(self, system, binsize, L):
        self.system   = system
        self.binsize  = binsize
        self.num_bins = np.round(L/self.binsize).astype(int)
        self.H_vx     = np.zeros(self.num_bins)
        self.H_dens   = np.zeros(self.num_bins)
        self.counter  = 0
        self.range    = [-L/2.,L/2.]

    def __call__(self, timestep):
        hoomd.util.quiet_status()
        snap = self.system.take_snapshot()
        hoomd.util.unquiet_status()
        pos = snap.particles.position
        vel = snap.particles.velocity
        # pick z coordinates
        pos_z = pos[:,2]
        H_dens, edges = np.histogram(pos_z,bins = self.num_bins,range=self.range)
        H_vx, edges   = np.histogram(pos_z,weights=vel[:,0], bins = self.num_bins,range=self.range)

        self.H_dens  += H_dens
        self.H_vx    += H_vx

        if self.counter==100:
            to_save_Hvx = np.divide(self.H_vx, self.H_dens, out=np.zeros_like(self.H_vx), where=self.H_dens!=0)
            centers  =  (edges[:-1] + edges[1:])/2
            np.savetxt('tutorial_reverse_perturbation_1_LJ_vx.hist',np.c_[centers,to_save_Hvx], header="z, v_x")
            self.counter  = 0
        else:
            self.counter += 1

hoomd.md.integrate.mode_standard(dt = 0.005)
langevin = hoomd.md.integrate.langevin(group=all, kT=kT, seed=457)
hoomd.run(1e4)
langevin.disable()

o = measure_fluid_properties(system, binsize=0.1, L=L)
analyze =  hoomd.analyze.callback(o, period=1e2)
log = hoomd.analyze.log(filename="tutorial_reverse_perturbation_1_LJ.log",
                        quantities=['rp_momentum'],
                        period=1e2,overwrite=True)
hoomd.dump.gsd(filename="tutorial_reverse_perturbation_1_LJ_trajectory.gsd",
               overwrite=True, period=1e2, group=all)

nve = hoomd.md.integrate.nve(group = all)
f = azplugins.flow.reverse_perturbation(group=all,width=1,Nswap=1,period=10,target_momentum=0.5)
hoomd.run(5e5)
