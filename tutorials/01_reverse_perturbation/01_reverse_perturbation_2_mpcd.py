import numpy as np
import sys
#sys.path.insert(0,'/Users/statt/programs/azplugins-0.9.1')
sys.path.insert(0,'/Users/statt/programs/hoomd-2.6.0')
import hoomd
from hoomd import md
from hoomd import data
from hoomd import azplugins
from hoomd.azplugins import mpcd
#import azplugins

L = 10
kT = 1.0
rho_mpcd = 5
viscosity_mpcd = 3.955

hoomd.context.initialize()
hoomd.context.SimulationContext()

snapshot = hoomd.data.make_snapshot(N=0,box=data.boxdim(L=L))
system = hoomd.init.read_snapshot(snapshot)

N_mpcd = int(rho_mpcd*system.box.get_volume())
snap = hoomd.mpcd.data.make_snapshot(N=N_mpcd)

snap.particles.position[:] = np.random.uniform(-L/2.0, L/2.0, (N_mpcd,3))
snap.particles.velocity[:] = np.random.normal(0, np.sqrt(kT), (N_mpcd,3))
snap.particles.velocity[:] -= np.average(snap.particles.velocity,axis=0)

mpcd_sys = hoomd.mpcd.init.read_snapshot(snap)

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
            np.savetxt('tutorial_reverse_perturbation_2_mpcd_vx.hist',np.c_[centers,to_save_Hvx], header="z, v_x")
            self.counter  = 0
        else:
            self.counter += 1


hoomd.mpcd.integrator(dt=0.1)
mpcd_sys.sorter.set_period(25)
srd   = hoomd.mpcd.collide.srd(seed=512, period=1, angle=130., kT=kT)
bulk  = hoomd.mpcd.stream.bulk(period=1)

hoomd.run(1e5)

o = measure_fluid_properties(mpcd_sys, binsize=0.1, L=L)
analyze =  hoomd.analyze.callback(o, period=1e2)
log = hoomd.analyze.log(filename="tutorial_reverse_perturbation_2_mpcd.log",
                        quantities=['rp_momentum'],
                        period=1e2,overwrite=True)

f = azplugins.mpcd.reverse_perturbation(width=1,Nswap=1,period=1,target_momentum=0.5)

hoomd.run(1e6)

snap = mpcd_sys.take_snapshot()
pos  = snap.particles.position
vel  = snap.particles.velocity
np.save('tutorial_reverse_perturbation_2_mpcd_pos.npy',pos)
np.save('tutorial_reverse_perturbation_2_mpcd_vel.npy',vel)
