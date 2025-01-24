// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_VELOCITY_FIELD_COMPUTE_GPU_CUH_
#define AZPLUGINS_VELOCITY_FIELD_COMPUTE_GPU_CUH_

#include <cuda_runtime.h>

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"

namespace hoomd
    {
namespace azplugins
    {
namespace gpu
    {

cudaError_t zeroVelocityFieldArrays(Scalar* d_mass, Scalar3* d_momentum, size_t num_bins);

template<class LoadOpT, class BinOpT>
cudaError_t bin_velocity_field(Scalar* d_mass,
                               Scalar3* d_momentum,
                               const LoadOpT& load_op,
                               const BinOpT& bin_op,
                               const BoxDim& global_box,
                               const unsigned int N,
                               const unsigned int block_size);

#ifdef __HIPCC__
namespace kernel
    {
template<class LoadOpT, class BinOpT>
__global__ void bin_velocity_field(Scalar* d_mass,
                                   Scalar3* d_momentum,
                                   const LoadOpT load_op,
                                   const BinOpT bin_op,
                                   const BoxDim global_box,
                                   const unsigned int N)
    {
    // one thread per particle
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    Scalar3 position, momentum;
    Scalar mass;
    load_op(position, momentum, mass, idx);
    momentum *= mass;

    // ensure particle is inside global box
    int3 img;
    global_box.wrap(position, img);

    uint3 bin = make_uint3(0, 0, 0);
    Scalar3 transformed_momentum = make_scalar3(0, 0, 0);
    const bool binned = bin_op.bin(bin, transformed_momentum, position, momentum);
    if (!binned)
        {
        return;
        }
    const auto bin_idx = bin_op.ravelBin(bin);

    atomicAdd(d_mass + bin_idx, mass);

    Scalar3* bin_momentum = d_momentum + bin_idx;
    atomicAdd(&(*bin_momentum).x, transformed_momentum.x);
    atomicAdd(&(*bin_momentum).y, transformed_momentum.y);
    atomicAdd(&(*bin_momentum).z, transformed_momentum.z);
    }

    } // namespace kernel

template<class LoadOpT, class BinOpT>
cudaError_t bin_velocity_field(Scalar* d_mass,
                               Scalar3* d_momentum,
                               const LoadOpT& load_op,
                               const BinOpT& bin_op,
                               const BoxDim& global_box,
                               const unsigned int N,
                               const unsigned int block_size)
    {
    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)kernel::bin_velocity_field<LoadOpT, BinOpT>);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(N / run_block_size + 1);

    kernel::bin_velocity_field<<<grid, run_block_size>>>(d_mass,
                                                         d_momentum,
                                                         load_op,
                                                         bin_op,
                                                         global_box,
                                                         N);

    return cudaSuccess;
    }
#endif // __HIPCC__

    } // end namespace gpu
    } // namespace azplugins
    } // end namespace hoomd
#endif // AZPLUGINS_VELOCITY_FIELD_COMPUTE_GPU_H_
