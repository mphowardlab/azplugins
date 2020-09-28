// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file TwoStepSLLODLangevinFlowGPU.cuh
 * \brief Definition of kernel drivers and kernels for TwoStepSLLODLangevinFlowGPU
 */

#include "TwoStepSLLODLangevinFlowGPU.cuh"


namespace azplugins
{
namespace gpu
{
namespace kernel
{
__global__ void langevin_sllod_step1(Scalar4 *d_pos,
                                    int3 *d_image,
                                    Scalar4 *d_vel,
                                    const Scalar3 *d_accel,
                                    const unsigned int *d_group,
                                    const BoxDim box,
                                    const unsigned int N,
                                    const Scalar dt)
    {
    const unsigned int grp_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (grp_idx >= N) return;

    const unsigned int idx = d_group[grp_idx];

    // position
    const Scalar4 postype = d_pos[idx];
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    unsigned int type = __scalar_as_int(postype.w);

    // velocity
    const Scalar4 velmass = d_vel[idx];
    Scalar3 vel = make_scalar3(velmass.x, velmass.y, velmass.z);
    Scalar mass = velmass.w;

    // acceleration
    const Scalar3 accel = d_accel[idx];

    // update position and wrap
    pos += (vel + Scalar(0.5) * dt * accel) * dt;
    int3 image = d_image[idx];
    box.wrap(pos,image);

    // update velocity
    vel += Scalar(0.5) * dt * accel;

    d_pos[idx] = make_scalar4(pos.x, pos.y, pos.z, __int_as_scalar(type));
    d_vel[idx] = make_scalar4(vel.x, vel.y, vel.z, mass);
    d_image[idx] = image;
    }


__global__ void langevin_sllod_step2(Scalar4 *d_vel,
                                    Scalar3 *d_accel,
                                    const Scalar4 *d_pos,
                                    const Scalar4 *d_net_force,
                                    const unsigned int *d_tag,
                                    const unsigned int *d_group,
                                    const Scalar *d_diameter,
                                    const Scalar lambda,
                                    const Scalar *d_gamma,
                                    const unsigned int ntypes,
                                    const unsigned int N,
                                    const Scalar dt,
                                    const Scalar T,
                                    const unsigned int timestep,
                                    const unsigned int seed,
                                    bool noiseless,
                                    bool use_lambda)
    {
    // optionally cache gamma into shared memory
    extern __shared__ Scalar s_gammas[];
    if (!use_lambda)
        {
        for (int cur_offset = 0; cur_offset < ntypes; cur_offset += blockDim.x)
            {
            if (cur_offset + threadIdx.x < ntypes)
                s_gammas[cur_offset + threadIdx.x] = d_gamma[cur_offset + threadIdx.x];
            }
        __syncthreads();
        }

    // one thread per particle in group
    const unsigned int grp_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (grp_idx >= N) return;
    const unsigned int idx = d_group[grp_idx];

    // get the friction coefficient
    const Scalar4 postype = d_pos[idx];
    Scalar gamma;
    if (use_lambda)
        {
        gamma = lambda*d_diameter[idx];
        }
    else
        {
        unsigned int typ = __scalar_as_int(postype.w);
        gamma = s_gammas[typ];
        }

    // get the flow field at the current position
  //  const Scalar3 flow_vel = flow_field(make_scalar3(postype.x, postype.y, postype.z));

    // compute the random force
    Scalar coeff = fast::sqrt(Scalar(6.0) * gamma * T / dt);
    if (noiseless)
        coeff = Scalar(0.0);
    hoomd::RandomGenerator rng(RNGIdentifier::TwoStepSLLODLangevinFlow, seed, d_tag[idx], timestep);
    hoomd::UniformDistribution<Scalar> uniform(-coeff, coeff);
    const Scalar3 random = make_scalar3(uniform(rng), uniform(rng), uniform(rng));

    const Scalar4 velmass = d_vel[idx];
    Scalar3 vel = make_scalar3(velmass.x, velmass.y, velmass.z);
    const Scalar mass = velmass.w;

    // total BD force
    Scalar3 bd_force = random - gamma * (vel);

    // compute the new acceleration
    const Scalar4 net_force = d_net_force[idx];
    Scalar3 accel = make_scalar3(net_force.x,net_force.y,net_force.z);
    accel += bd_force;
    const Scalar minv = Scalar(1.0) / mass;
    accel.x *= minv;
    accel.y *= minv;
    accel.z *= minv;

    // update the velocity
    vel += Scalar(0.5) * dt * accel;

    // write out update velocity and acceleration
    d_vel[idx] = make_scalar4(vel.x, vel.y, vel.z, mass);
    d_accel[idx] = accel;
    }



} // end namespace kernel

cudaError_t langevin_sllod_step1(Scalar4 *d_pos,
                                int3 *d_image,
                                Scalar4 *d_vel,
                                const Scalar3 *d_accel,
                                const unsigned int *d_group,
                                const BoxDim& box,
                                const unsigned int N,
                                const Scalar dt,
                                const unsigned int block_size)
    {
    if (N == 0) return cudaSuccess;

    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)kernel::langevin_sllod_step1);
        max_block_size = attr.maxThreadsPerBlock;
        }

    const int run_block_size = min(block_size, max_block_size);
    kernel::langevin_sllod_step1<<<N/run_block_size+1, run_block_size>>>(d_pos,
                                                                        d_image,
                                                                        d_vel,
                                                                        d_accel,
                                                                        d_group,
                                                                        box,
                                                                        N,
                                                                        dt);
    return cudaSuccess;
    }

cudaError_t langevin_sllod_step2(Scalar4 *d_vel,
                                Scalar3 *d_accel,
                                const Scalar4 *d_pos,
                                const Scalar4 *d_net_force,
                                const unsigned int *d_tag,
                                const unsigned int *d_group,
                                const Scalar *d_diameter,
                                const Scalar lambda,
                                const Scalar *d_gamma,
                                const unsigned int ntypes,
                                const unsigned int N,
                                const Scalar dt,
                                const Scalar T,
                                const unsigned int timestep,
                                const unsigned int seed,
                                bool noiseless,
                                bool use_lambda,
                                const unsigned int block_size)
    {
    if (N == 0) return cudaSuccess;

    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)kernel::langevin_sllod_step2);
        max_block_size = attr.maxThreadsPerBlock;
        }

    const int run_block_size = min(block_size, max_block_size);
    const size_t shared_bytes = sizeof(Scalar) * ntypes;

    kernel::langevin_sllod_step2<<<N/run_block_size+1, run_block_size, shared_bytes>>>(d_vel,
                                                               d_accel,
                                                               d_pos,
                                                               d_net_force,
                                                               d_tag,
                                                               d_group,
                                                               d_diameter,
                                                               lambda,
                                                               d_gamma,
                                                               ntypes,
                                                               N,
                                                               dt,
                                                               T,
                                                               timestep,
                                                               seed,
                                                               noiseless,
                                                               use_lambda);
    return cudaSuccess;
    }



} // end namespace gpu
} // end namespace azplugins
