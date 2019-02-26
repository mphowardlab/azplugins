// Copyright (c) 2018-2019, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file TwoStepBrownianFlowGPU.cuh
 * \brief Declaration of kernel drivers for TwoStepBrownianFlowGPU
 */

#ifndef AZPLUGINS_TWO_STEP_BROWNIAN_FLOW_GPU_CUH_
#define AZPLUGINS_TWO_STEP_BROWNIAN_FLOW_GPU_CUH_

#include <cuda_runtime.h>
#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Saru.h"

namespace azplugins
{
namespace gpu
{

//! Brownian dynamics step in flow
template<class FlowField>
cudaError_t brownian_flow(Scalar4 *d_pos,
                          int3 *d_image,
                          const BoxDim& box,
                          const Scalar4 *d_net_force,
                          const unsigned int *d_tag,
                          const unsigned int *d_group,
                          const Scalar *d_diameter,
                          const Scalar lambda,
                          const Scalar *d_gamma,
                          const unsigned int ntypes,
                          const FlowField& flow_field,
                          const unsigned int N,
                          const Scalar dt,
                          const Scalar T,
                          const unsigned int timestep,
                          const unsigned int seed,
                          bool noiseless,
                          bool use_lambda,
                          const unsigned int block_size);

#ifdef NVCC
namespace kernel
{
template<class FlowField>
__global__ void brownian_flow(Scalar4 *d_pos,
                              int3 *d_image,
                              const BoxDim box,
                              const Scalar4 *d_net_force,
                              const unsigned int *d_tag,
                              const unsigned int *d_group,
                              const Scalar *d_diameter,
                              const Scalar lambda,
                              const Scalar *d_gamma,
                              const unsigned int ntypes,
                              const FlowField flow_field,
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
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    const unsigned int type = __scalar_as_int(postype.w);
    Scalar gamma;
    if (use_lambda)
        gamma = lambda*d_diameter[idx];
    else
        {
        gamma = s_gammas[type];
        }

    // get the flow velocity at the current position
    const Scalar3 flow_vel = flow_field(pos);

    // compute the random force
    Scalar coeff = fast::sqrt(Scalar(6.0)*gamma*T/dt);
    if (noiseless)
        coeff = Scalar(0.0);
    hoomd::detail::Saru s(d_tag[idx], timestep, seed);
    const Scalar3 random_force = coeff * make_scalar3(s.s<Scalar>(-1.0, 1.0),
                                                      s.s<Scalar>(-1.0, 1.0),
                                                      s.s<Scalar>(-1.0, 1.0));

    // get the conservative force
    const Scalar4 net_force = d_net_force[idx];
    Scalar3 cons_force = make_scalar3(net_force.x,net_force.y,net_force.z);

    // update position
    pos += (flow_vel + (cons_force + random_force)/gamma) * dt;
    int3 image = d_image[idx];
    box.wrap(pos, image);

    // write out the position and image
    d_pos[idx] = make_scalar4(pos.x, pos.y, pos.z, type);
    d_image[idx] = image;
    }
} // end namespace kernel

template<class FlowField>
cudaError_t brownian_flow(Scalar4 *d_pos,
                          int3 *d_image,
                          const BoxDim& box,
                          const Scalar4 *d_net_force,
                          const unsigned int *d_tag,
                          const unsigned int *d_group,
                          const Scalar *d_diameter,
                          const Scalar lambda,
                          const Scalar *d_gamma,
                          const unsigned int ntypes,
                          const FlowField& flow_field,
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
        cudaFuncGetAttributes(&attr, (const void*)kernel::brownian_flow<FlowField>);
        max_block_size = attr.maxThreadsPerBlock;
        }

    const int run_block_size = min(block_size, max_block_size);
    const size_t shared_bytes = sizeof(Scalar) * ntypes;

    kernel::brownian_flow<FlowField>
        <<<N/run_block_size+1, run_block_size, shared_bytes>>>(d_pos,
                                                               d_image,
                                                               box,
                                                               d_net_force,
                                                               d_tag,
                                                               d_group,
                                                               d_diameter,
                                                               lambda,
                                                               d_gamma,
                                                               ntypes,
                                                               flow_field,
                                                               N,
                                                               dt,
                                                               T,
                                                               timestep,
                                                               seed,
                                                               noiseless,
                                                               use_lambda);
    return cudaSuccess;
    }
#endif // NVCC

} // end namespace gpu
} // end namespace azplugins

#endif // AZPLUGINS_TWO_STEP_BROWNIAN_FLOW_GPU_CUH_
