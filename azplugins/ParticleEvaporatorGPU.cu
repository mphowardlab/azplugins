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
/*!
 * \param d_select_flags Flags identifying which particles to select (1 = select)
 * \param d_mark Array of particle indexes
 * \param d_pos Particle positions
 * \param solvent_type Type index of solvent particles
 * \param z_lo Lower bound of region in z
 * \param z_hi Upper bound of region in z
 * \param N Number of particles
 *
 * Using one thread per particle, all particle positions are checked. Any particles
 * of type \a solvent_type that are in the slab bounded by \a z_lo and \a z_hi
 * are flagged for evaporation in \a d_select_flags with a 1. (Others are flagged to 0.)
 * The \a d_mark array is filled up with the particle indexes so that evaporate_select_mark
 * can later select these particle indexes based on \a d_select_flags.
 */
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
    bool evap = (type == solvent_type && !(pos.z > z_hi || pos.z < z_lo));

    // coalesce writes of all particles
    d_select_flags[idx] = (evap) ? 1 : 0;
    d_mark[idx] = idx;
    }

/*!
 * \param d_pos Particle positions
 * \param d_picks Indexes of picked particles in \a d_mark
 * \param d_mark Compacted array of particle indexes marked as candidates for evaporation
 * \param evaporated_type Type index of evaporated particles
 * \param N_pick Number of picks made
 *
 * Using one thread per particle, the types of picked particles are transformed
 * to \a evaporated_type. See kernel::evaporate_setup_mark for details of how
 * particles are marked as candidates for evaporation.
 */
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

/*!
 * \param d_select_flags Flags identifying which particles to select (1 = select)
 * \param d_mark Array of particle indexes
 * \param d_pos Particle positions
 * \param solvent_type Type index of solvent particles
 * \param z_lo Lower bound of region in z
 * \param z_hi Upper bound of region in z
 * \param N Number of particles
 * \param block_size Number of threads per block
 *
 * \sa kernel::evaporate_setup_mark
 */
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

/*!
 * \param d_mark Compacted array of particle indexes marked as candidates for evaporation
 * \param d_num_mark Flag to store the total number of marked particles
 * \param d_tmp_storage Temporary storage allocated on the device, NULL on first call
 * \param tmp_storage_bytes Number of bytes necessary for temporary storage, 0 on first call
 * \param d_select_flags Flags identifying which particles to select (1 = select)
 * \param N Number of particles
 *
 * The CUB library is used to compact the particle indexes of the selected particles
 * into \a d_mark based on the flags set in \a d_select_flags. The number of marked
 * particles is also determined.
 *
 * \note This function must be called twice. On the first call, the temporary storage
 *       required is sized and stored in \a tmp_storage_bytes. Device memory must be
 *       allocated to \a d_tmp_storage, and then the function can be called again
 *       to apply the transformation.
 *
 * \note Per CUB user group, DeviceSelect is in-place safe, and so input and output
 *       do not require a double buffer.
 *
 * See kernel::evaporate_setup_mark for details of how particles are marked as candidates
 * for evaporation.
 */
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

/*!
 * \param d_pos Particle positions
 * \param d_picks Indexes of picked particles in \a d_mark
 * \param d_mark Compacted array of particle indexes marked as candidates for evaporation
 * \param evaporated_type Type index of evaporated particles
 * \param N_pick Number of picks made
 * \param block_size Number of threads per block
 *
 * \sa kernel::evaporate_apply_picks
 */
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
