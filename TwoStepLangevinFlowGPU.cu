// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

/*!
 * \file TwoStepLangevinFlowGPU.cu
 * \brief Definition of kernel drivers and kernels for TwoStepLangevinFlowGPU
 */

#include "TwoStepLangevinFlowGPU.cuh"
#include "FlowFields.h"

namespace azplugins
{
namespace gpu
{
namespace kernel
{
__global__ void langevin_flow_step1(Scalar4 *d_pos,
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
} // end namespace kernel

cudaError_t langevin_flow_step1(Scalar4 *d_pos,
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
        cudaFuncGetAttributes(&attr, (const void*)kernel::langevin_flow_step1);
        max_block_size = attr.maxThreadsPerBlock;
        }

    const int run_block_size = min(block_size, max_block_size);
    kernel::langevin_flow_step1<<<N/run_block_size+1, run_block_size>>>(d_pos,
                                                                        d_image,
                                                                        d_vel,
                                                                        d_accel,
                                                                        d_group,
                                                                        box,
                                                                        N,
                                                                        dt);
    return cudaSuccess;
    }

//! Explicit instantiation of ConstantFlow integrator
template cudaError_t langevin_flow_step2<azplugins::ConstantFlow>(Scalar4 *d_vel,
                                                                  Scalar3 *d_accel,
                                                                  const Scalar4 *d_pos,
                                                                  const Scalar4 *d_net_force,
                                                                  const unsigned int *d_tag,
                                                                  const unsigned int *d_group,
                                                                  const Scalar *d_diameter,
                                                                  const Scalar lambda,
                                                                  const Scalar *d_gamma,
                                                                  const unsigned int ntypes,
                                                                  const azplugins::ConstantFlow& flow_field,
                                                                  const unsigned int N,
                                                                  const Scalar dt,
                                                                  const Scalar T,
                                                                  const unsigned int timestep,
                                                                  const unsigned int seed,
                                                                  bool noiseless,
                                                                  bool use_lambda,
                                                                  const unsigned int block_size);
//! Explicit instantiation of ParabolicFlow integrator
template cudaError_t langevin_flow_step2<azplugins::ParabolicFlow>(Scalar4 *d_vel,
                                                                   Scalar3 *d_accel,
                                                                   const Scalar4 *d_pos,
                                                                   const Scalar4 *d_net_force,
                                                                   const unsigned int *d_tag,
                                                                   const unsigned int *d_group,
                                                                   const Scalar *d_diameter,
                                                                   const Scalar lambda,
                                                                   const Scalar *d_gamma,
                                                                   const unsigned int ntypes,
                                                                   const azplugins::ParabolicFlow& flow_field,
                                                                   const unsigned int N,
                                                                   const Scalar dt,
                                                                   const Scalar T,
                                                                   const unsigned int timestep,
                                                                   const unsigned int seed,
                                                                   bool noiseless,
                                                                   bool use_lambda,
                                                                   const unsigned int block_size);

//! Explicit instantiation of QuiescentFluid integrator
template cudaError_t langevin_flow_step2<azplugins::QuiescentFluid>(Scalar4 *d_vel,
                                                                    Scalar3 *d_accel,
                                                                    const Scalar4 *d_pos,
                                                                    const Scalar4 *d_net_force,
                                                                    const unsigned int *d_tag,
                                                                    const unsigned int *d_group,
                                                                    const Scalar *d_diameter,
                                                                    const Scalar lambda,
                                                                    const Scalar *d_gamma,
                                                                    const unsigned int ntypes,
                                                                    const azplugins::QuiescentFluid& flow_field,
                                                                    const unsigned int N,
                                                                    const Scalar dt,
                                                                    const Scalar T,
                                                                    const unsigned int timestep,
                                                                    const unsigned int seed,
                                                                    bool noiseless,
                                                                    bool use_lambda,
                                                                    const unsigned int block_size);

} // end namespace gpu
} // end namespace azplugins
