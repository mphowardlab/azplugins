// Copyright (c) 2016, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file ParticleEvaporatorGPU.cu
 * \brief Definition of kernel drivers and kernels for ParticleEvaporatorGPU
 */

#include "ParticleEvaporatorGPU.cuh"
#include "hoomd/extern/cub/cub.cuh"

namespace azplugins
{
namespace gpu
{
namespace kernel
{
__global__ void evaporate_setup_mark(unsigned char *d_select_flags,
                                     unsigned int *d_mark,
                                     const Scalar4 *d_pos,
                                     unsigned int solvent_type,
                                     Scalar z_lo,
                                     Scalar z_hi,
                                     unsigned int N)
    {
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;

    const Scalar4 postype = d_pos[idx];
    const Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    unsigned int type = __scalar_as_int(postype.w);

    // check for solvent particles in region as for an AABB
    bool inside = (type == solvent_type && !(pos.z > z_hi || pos.z < z_lo));

    // coalesce writes of all particles
    d_select_flags[idx] = (inside) ? 1 : 0;
    d_mark[idx] = idx;
    }

__global__ void evaporate_apply_picks(Scalar4 *d_pos,
                                      const unsigned int *d_picks,
                                      const unsigned int *d_mark,
                                      unsigned int evaporated_type,
                                      unsigned int N_pick)
    {
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N_pick) return;

    const unsigned int pick = d_picks[idx];
    const unsigned int pidx = d_mark[pick];

    d_pos[pidx].w = __int_as_scalar(evaporated_type);
    }
}

//! Kernel driver to change particle types in a region on the GPU
cudaError_t evaporate_setup_mark(unsigned char *d_select_flags,
                                 unsigned int *d_mark,
                                 const Scalar4 *d_pos,
                                 unsigned int solvent_type,
                                 Scalar z_lo,
                                 Scalar z_hi,
                                 unsigned int N,
                                 unsigned int block_size)
    {
    if (N == 0) return cudaSuccess;

    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)kernel::evaporate_setup_mark);
        max_block_size = attr.maxThreadsPerBlock;
        }

    const int run_block_size = min(block_size, max_block_size);
    kernel::evaporate_setup_mark<<<N/run_block_size + 1, run_block_size>>>(d_select_flags,
                                                                           d_mark,
                                                                           d_pos,
                                                                           solvent_type,
                                                                           z_lo,
                                                                           z_hi,
                                                                           N);

    return cudaSuccess;
    }

cudaError_t evaporate_select_mark(unsigned int *d_mark,
                                  unsigned int *d_num_mark,
                                  void *d_tmp_storage,
                                  size_t &tmp_storage_bytes,
                                  const unsigned char *d_select_flags,
                                  unsigned int N)
    {
    if (N == 0) return cudaSuccess;

    cub::DeviceSelect::Flagged(d_tmp_storage, tmp_storage_bytes, d_mark, d_select_flags, d_mark, d_num_mark, N);

    return cudaSuccess;
    }

cudaError_t evaporate_apply_picks(Scalar4 *d_pos,
                                  const unsigned int *d_picks,
                                  const unsigned int *d_mark,
                                  unsigned int evaporated_type,
                                  unsigned int N_pick,
                                  unsigned int block_size)
    {
    if (N_pick == 0) return cudaSuccess;

    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)kernel::evaporate_apply_picks);
        max_block_size = attr.maxThreadsPerBlock;
        }

    const int run_block_size = min(block_size, max_block_size);
    kernel::evaporate_apply_picks<<<N_pick/run_block_size+1, run_block_size>>>(d_pos,
                                                                               d_picks,
                                                                               d_mark,
                                                                               evaporated_type,
                                                                               N_pick);

    return cudaSuccess;
    }
} // end namespace gpu
} // end namespace azplugins
