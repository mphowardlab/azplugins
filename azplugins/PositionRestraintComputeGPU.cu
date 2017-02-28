// Copyright (c) 2016-2017, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: wes_reinhart

/*!
 * \file PositionRestraintComputeGPU.cu
 * \brief Defines CUDA kernels for PositionRestraintComputeGPU
 */

#include "PositionRestraintComputeGPU.cuh"

namespace azplugins
{
namespace gpu
{
namespace kernel
{
/*!
 * \param d_force Particle forces
 * \param d_member_idx Indices of group members
 * \param d_pos Particle positions
 * \param d_ref_pos Particle reference positions
 * \param d_tag Particle tags
 * \param k Field force constant
 * \param box Simulation box
 * \param N_mem Number of particles in the group
 *
 * Using one thread per particle, the potential and force of the restraining potential
 * is computed per-particle, relative to a reference position.
 *
 */
__global__ void compute_position_restraint(Scalar4 *d_force,
                                           const unsigned int *d_member_idx,
                                           const Scalar4 *d_pos,
                                           const Scalar4 *d_ref_pos,
                                           const unsigned int *d_tag,
                                           const Scalar3 k,
                                           const BoxDim box,
                                           const unsigned int N_mem)
    {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // one thread per particle
    if (idx >= N_mem)
        return;

    const unsigned int cur_p = d_member_idx[idx];
    const Scalar4 cur_pos_type = d_pos[cur_p];
    const Scalar3 cur_pos = make_scalar3(cur_pos_type.x, cur_pos_type.y, cur_pos_type.z);

    const unsigned int cur_tag = d_tag[cur_p];
    const Scalar4 cur_ref_pos_type = d_ref_pos[cur_tag];
    const Scalar3 cur_ref_pos = make_scalar3(cur_ref_pos_type.x, cur_ref_pos_type.y, cur_ref_pos_type.z);

    // compute distance between current and reference position
    Scalar3 dr = box.minImage(cur_pos - cur_ref_pos);

    // termwise squaring for energy calculation
    const Scalar3 dr2 = make_scalar3(dr.x*dr.x, dr.y*dr.y, dr.z*dr.z);

    // F = -k x, U = 0.5 kx^2
    d_force[cur_p] = make_scalar4(-k.x*dr.x,
                                  -k.y*dr.y,
                                  -k.z*dr.z,
                                  Scalar(0.5)*dot(k, dr2));
    }
} // end namespace kernel

/*!
 * \param d_force Particle forces
 * \param d_member_idx Indices of group members
 * \param d_pos Particle positions
 * \param d_ref_pos Particle reference positions
 * \param d_tag Particle tags
 * \param k Field force constant
 * \param box Simulation box
 * \param N Number of particles
 * \param N_mem Number of particles in the group
 * \param block_size Number of threads per block
 * \param compute_capability GPU compute capability
 *
 * This kernel driver is a wrapper around kernel::compute_position_restraint.
 * The forces are set to zero before calculation.
 *
 */
cudaError_t compute_position_restraint(Scalar4 *d_force,
                                       const unsigned int *d_member_idx,
                                       const Scalar4 *d_pos,
                                       const Scalar4 *d_ref_pos,
                                       const unsigned int *d_tag,
                                       const Scalar3& k,
                                       const BoxDim& box,
                                       const unsigned int N,
                                       const unsigned int N_mem,
                                       const unsigned int block_size,
                                       const unsigned int compute_capability)
    {
    // asynchronous memset in the default stream will allow other simple hosts tasks to proceed before kernel launch
    cudaError_t error;
    error = cudaMemset(d_force, 0, sizeof(Scalar4)*N);

    if (error != cudaSuccess)
        return error;

    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)kernel::compute_position_restraint);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);

    dim3 grid(N_mem / run_block_size + 1);

    kernel::compute_position_restraint<<<grid, run_block_size>>>(d_force,
                                                                 d_member_idx,
                                                                 d_pos,
                                                                 d_ref_pos,
                                                                 d_tag,
                                                                 k,
                                                                 box,
                                                                 N_mem);
    return cudaSuccess;
    }

} // end namespace gpu
} // end namespace azplugins
