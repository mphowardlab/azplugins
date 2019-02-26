// Copyright (c) 2018-2019, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file TypeUpdaterGPU.cu
 * \brief Definition of kernel drivers and kernels for TypeUpdaterGPU
 */

#include "TypeUpdaterGPU.cuh"

namespace azplugins
{
namespace gpu
{
namespace kernel
{
//! Applies the type change in region
/*!
 * \param d_pos Particle position array
 * \param inside_type Type of particles inside region
 * \param outside_type Type of particles outside region
 * \param z_lo Lower bound of region in z
 * \param z_hi Upper bound of region in z
 * \param N Number of particles
 *
 * One thread per particle is used to flip particles between \a inside_type
 * and \a outside_type based on their position in the slab bounded in *z* between
 * \a z_lo and \a z_hi.
 */
__global__ void change_types_region(Scalar4 *d_pos,
                                    unsigned int inside_type,
                                    unsigned int outside_type,
                                    Scalar z_lo,
                                    Scalar z_hi,
                                    unsigned int N)
    {
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;

    const Scalar4 postype = d_pos[idx];
    const Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    unsigned int type = __scalar_as_int(postype.w);

    // only check region if type is one that can be flipped
    if (type == inside_type || type == outside_type)
        {
        // test for overlap as for an AABB
        bool inside = !(pos.z > z_hi || pos.z < z_lo);
        if (inside)
            {
            type = inside_type;
            }
        else
            {
            type = outside_type;
            }
        }
    d_pos[idx] = make_scalar4(pos.x, pos.y, pos.z, __int_as_scalar(type));
    }
}

/*!
 * \param d_pos Particle position array
 * \param inside_type Type of particles inside region
 * \param outside_type Type of particles outside region
 * \param z_lo Lower bound of region in z
 * \param z_hi Upper bound of region in z
 * \param N Number of particles
 * \param block_size Number of threads per block
 */
cudaError_t change_types_region(Scalar4 *d_pos,
                                unsigned int inside_type,
                                unsigned int outside_type,
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
        cudaFuncGetAttributes(&attr, (const void*)kernel::change_types_region);
        max_block_size = attr.maxThreadsPerBlock;
        }

    const int run_block_size = min(block_size, max_block_size);
    kernel::change_types_region<<<N/run_block_size + 1, run_block_size>>>(d_pos,
                                                                          inside_type,
                                                                          outside_type,
                                                                          z_lo,
                                                                          z_hi,
                                                                          N);

    return cudaSuccess;
    }

} // end namespace gpu
} // end namespace azplugins
