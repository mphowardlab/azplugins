// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt

/*!
 * \file TwoStepSLLODNVTFlowGPU.cu
 * \brief Declaration of SLLOD equation of motion with NVT Nosé-Hoover thermostat
 */

#include "TwoStepSLLODNVTFlowGPU.cuh"
#include <assert.h>

namespace azplugins
{
namespace gpu
{
namespace kernel
{
/*! \file TwoStepNVTGPU.cu
    \brief Defines GPU kernel code for NVT integration on the GPU. Used by TwoStepNVTGPU.
*/

//! Takes the first 1/2 step forward in the NVT integration step
/*! \param d_pos array of particle positions
    \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_image array of particle images
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param work_size Number of members in the group for this GPU
    \param box Box dimensions for periodic boundary condition handling
    \param exp_fac Velocity rescaling factor from thermostat
    \param deltaT Amount of real time to step forward in one time step
    \param shear_rate Shear rate of box deformation
    \param flipped True if the box is flipped this time step
    \param boundary_shear_velocity Shear velocity at the pbc boundary
    \param offset The offset of this GPU into the list of particles

    Take the first half step forward in the NVT integration.

    See gpu_nve_step_one_kernel() for some performance notes on how to handle the group data reads efficiently.
*/
extern "C" __global__
void sllod_nvt_step_one(Scalar4 *d_pos,
                             Scalar4 *d_vel,
                             const Scalar3 *d_accel,
                             int3 *d_image,
                             unsigned int *d_group_members,
                             unsigned int work_size,
                             BoxDim box,
                             Scalar exp_fac,
                             Scalar deltaT,
                             Scalar shear_rate,
                             bool flipped,
                             Scalar boundary_shear_velocity,
                             unsigned int offset)
    {
    // determine which particle this thread works on
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx < work_size)
        {
        unsigned int idx = d_group_members[group_idx + offset];

        // update positions to the next timestep and update velocities to the next half step
        Scalar4 postype = d_pos[idx];
        Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);

        Scalar4 velmass = d_vel[idx];
        Scalar3 vel = make_scalar3(velmass.x, velmass.y, velmass.z);
        Scalar3 accel = d_accel[idx];


        // remove flow field
        vel.x -= shear_rate*pos.y;

        // rescale velocity
        vel *= exp_fac;

        // apply sllod velocity correction
        vel.x -= Scalar(0.5)*shear_rate*vel.y*deltaT;

        // add flow field
        vel.x += shear_rate*pos.y;

        // update velocity
        vel += Scalar(0.5)*accel*deltaT;

        // update position
        pos += deltaT * vel;


        // read in the image flags
        int3 image = d_image[idx];

        // if box deformation caused a flip, wrap pos back into box
        if (flipped){
          image.x += image.y;
        //    pos.x *= -1;
        }

        // time to fix the periodic boundary conditions
        box.wrap(pos, image);

        // Periodic boundary correction to velocity:
        // if particle leaves from (+/-) y boundary it gets (-/+) velocity at boundary
        // note carefully that pair potentials dependent on differences in
        // velocities (e.g. DPD) are not yet explicitly supported.

        if ((image.y-d_image[idx].y)==1) // crossed pbc in +y, image increased by 1
        {
          vel.x -= boundary_shear_velocity;
        }
        else if ((image.y-d_image[idx].y)==-1) // crossed pbc in -y, image decreased by 1
        {
          vel.x += boundary_shear_velocity;
        }

        // write out the results
        d_pos[idx] = make_scalar4(pos.x, pos.y, pos.z, postype.w);
        d_vel[idx] = make_scalar4(vel.x, vel.y, vel.z, velmass.w);
        d_image[idx] = image;
        }
    }

//! Takes the second 1/2 step forward in the NVT integration step
/*! \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param work_size Number of members in the group for this GPU
    \param d_net_force Net force on each particle
    \param deltaT Amount of real time to step forward in one time step
    \param offset The offset of this GPU into the list of particles
*/
extern "C" __global__
void sllod_nvt_step_two(Scalar4 *d_vel,
                             Scalar3 *d_accel,
                             unsigned int *d_group_members,
                             unsigned int work_size,
                             Scalar4 *d_net_force,
                             Scalar deltaT,
                             Scalar shear_rate,
                             Scalar exp_v_fac_thermo,
                             unsigned int offset)
    {
    // determine which particle this thread works on
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx < work_size)
        {
        unsigned int idx = d_group_members[group_idx+offset];

        // read in the net force and calculate the acceleration
        Scalar4 net_force = d_net_force[idx];
        Scalar3 accel = make_scalar3(net_force.x,net_force.y,net_force.z);

        Scalar4 vel = d_vel[idx];
        Scalar3 v = make_scalar3(vel.x,vel.y,vel.z);

        Scalar mass = vel.w;
        accel = accel/mass;

        // rescale
        v *= exp_v_fac_thermo;

        // SLLOD correction to velocity: shear rate tensor dotted with velocity
        const Scalar3 v_del_u = make_scalar3(shear_rate* vel.y, 0.0, 0.0);

        // update velocity
        v += Scalar(0.5)*(accel - v_del_u)*deltaT;

        // write out data
        d_vel[idx] = make_scalar4(v.x,v.y,v.z,vel.w);

        // since we calculate the acceleration, we need to write it for the next step
        d_accel[idx] = accel;
        }
    }




} //end namespace kernel

/*! \param d_pos array of particle positions
    \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_image array of particle images
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param box Box dimensions for periodic boundary condition handling
    \param block_size Size of the block to run
    \param exp_fac Thermostat rescaling factor
    \param deltaT Amount of real time to step forward in one time step
    \param flipped true if box is flipped in this timestep
    \param boundary_shear_velocity value of the shear velocity at pbc boundary
*/
cudaError_t sllod_nvt_step_one(Scalar4 *d_pos,
                             Scalar4 *d_vel,
                             const Scalar3 *d_accel,
                             int3 *d_image,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             const BoxDim& box,
                             unsigned int block_size,
                             Scalar exp_fac,
                             Scalar deltaT,
                             Scalar shear_rate,
                             bool flipped,
                             Scalar boundary_shear_velocity,
                             const GPUPartition& gpu_partition)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)kernel::sllod_nvt_step_one);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);

    // iterate over active GPUs in reverse, to end up on first GPU when returning from this function
    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;

        // setup the grid to run the kernel
        dim3 grid( (nwork/run_block_size) + 1, 1, 1);
        dim3 threads(run_block_size, 1, 1);

        // run the kernel, starting with offset range.first
        kernel::sllod_nvt_step_one<<< grid, threads >>>(d_pos,
                             d_vel,
                             d_accel,
                             d_image,
                             d_group_members,
                             nwork,
                             box,
                             exp_fac,
                             deltaT,
                             shear_rate,
                             flipped,
                             boundary_shear_velocity,
                             range.first);
        }

    return cudaSuccess;
    }


/*! \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Net force on each particle
    \param block_size Size of the block to execute on the device
    \param deltaT Amount of real time to step forward in one time step
    \param shear_rate Box deformation shear rate
    \param exp_v_fac_thermo Exponential velocity scaling factor
*/
cudaError_t sllod_nvt_step_two(Scalar4 *d_vel,
                             Scalar3 *d_accel,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             Scalar4 *d_net_force,
                             unsigned int block_size,
                             Scalar deltaT,
                             Scalar shear_rate,
                             Scalar exp_v_fac_thermo,
                             const GPUPartition& gpu_partition)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)kernel::sllod_nvt_step_two);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);

    // iterate over active GPUs in reverse, to end up on first GPU when returning from this function
    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;

        // setup the grid to run the kernel
        dim3 grid( (nwork/run_block_size) + 1, 1, 1);
        dim3 threads(run_block_size, 1, 1);

        // run the kernel
        kernel::sllod_nvt_step_two<<< grid, threads >>>(d_vel, d_accel, d_group_members, nwork, d_net_force, deltaT, shear_rate, exp_v_fac_thermo, range.first);
        }

    return cudaSuccess;
    }

} //end namespace gpu
} //end namespace azplugins
