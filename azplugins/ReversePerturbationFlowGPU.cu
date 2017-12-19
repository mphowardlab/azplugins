// Copyright (c) 2016-2017, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt

/*!
 * \file ReversePerturbationFlowGPU.cu
 * \brief Definition of kernel drivers and kernels for ReversePerturbationFlowGPU
 */

#include "ReversePerturbationFlowGPU.cuh"
#include "ReversePerturbationUtilities.h"
#include "hoomd/extern/cub/cub/cub.cuh"
#include <thrust/sort.h>
#include <thrust/device_vector.h>

namespace azplugins
{
namespace gpu
{
namespace kernel
{
/*!
 *
 * \param d_slab_pairs Empty array for holding the (signed flags,momentum) of all particles in either slab
 * \param d_pos  Array of positions of particles
 * \param d_vel  Array of velocities of particles
 * \param d_member_idx  indices of particles in the group
 * \param d_lo_pos edges of the bottom slab
 * \param d_hi_pos edges of the top slab
 * \param N  Number of particles in the group
 *
 *  The function determines in which slab a particle belongs. A particle
 *  belongs to the bottom slab, if its z-position is between the edges of
 *  d_lo_pos and its momentum is positive. The signed char \a sign evaluates to +1
 *  in this case. A particle belongs to the top slab, if the z-position is between
 *  the edges of d_hi_pos and its momentum is negative. \a sign evaluates to -1
 *  for this case. For a particle which does not belong to either slab, either because
 *  of its positon or momentum, the \a sign evaluates to 0.
 *
 *  The particle index is shifted by one to allow the particle index 0 to be treated correctly,
 *  since 0 is used as flag to indicate that the particle is in neither slab.
 *  \a d_slab_pairs holds Scalar2 with the signed flag in the x-component and momentum in
 *  the y-component for each particle.
 */
__global__ void mark_particles_in_slabs(Scalar2 *d_slab_pairs,
                                        const Scalar4 *d_pos,
                                        const Scalar4 *d_vel,
                                        const unsigned int *d_member_idx,
                                        const Scalar2 d_lo_pos,
                                        const Scalar2 d_hi_pos,
                                        const unsigned int N)
    {
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;

    const unsigned int cur_p = d_member_idx[idx];
    const Scalar4 pos = d_pos[cur_p];
    const Scalar4 vel = d_vel[cur_p];
    const Scalar z = pos.z;
    const Scalar momentum = vel.x*vel.w;

    // sort the particle into slab
    signed char sign = (d_lo_pos.x <= z && z < d_lo_pos.y && momentum > 0)
                     - (d_hi_pos.x <= z && z < d_hi_pos.y && momentum < 0);
    // idx needs to shifted by 1 so that particle 0 can be treated correctly
    int tag = int(sign)*(int(cur_p)+1);
    d_slab_pairs[idx] = make_scalar2(__int_as_scalar(tag), momentum);
    }

/*!
 * \param d_split GPU flag for split between negative and positive flags
 * \param d_type  GPU flat to record the first value in the d_slab_pairs array
 * \param d_slab_pairs Sorted Array of (signed flags,momentum) for particles in both slabs
 * \param num_threads  Number of threads = number of entries in d_slab_pairs -1
 *
 * The kernel finds the split between the sign of the flags in the sorted \a d_slab_pairs,
 * which indicates the postion of the last top slab particle and the first bottom slab particle.
 * Nslab-1 threads are used, so that they all can look for their next right
 * The bitwise XOR operation ^ returns < if the signs are different and >= if
 * they are the same.
 * There can only be either one or zero splits, so only one or zero threads can find this position,
 * so writing into m_split shouldn't cause a race condition.
 * The \a d_type is needed to treat the case of either empty bottom or top
 * slab correctly.
 *
 */
__global__ void find_split_array(unsigned int *d_split,
                                 int *d_type,
                                 const Scalar2 *d_slab_pairs,
                                 const unsigned int num_threads)
    {
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_threads) return;

    //compare sign of the index of this entry with the next one
    int pid1 = __scalar_as_int(d_slab_pairs[idx].x);
    int pid2 = __scalar_as_int(d_slab_pairs[idx+1].x);
    /* d_slab_pairs is sorted, there can be only either 1 or 0
     * threads which find this condition.
     * if their signs are different, split between top and bottom is found
     */
    if ((pid1 ^ pid2)< 0)  *d_split=idx+1;
    if (idx == 0 ) *d_type = pid1;
    }

/*!
 * \param d_layer_hi Array for (index,momentum) of particles in bottom slab
 * \param d_layer_lo  Array for (index,momentum) of  particles in top slab
 * \param d_vel Velocities of particles
 * \param d_member_idx  Indices of particles in group
 * \param num_pairs Number of swaps
 *
 * This kernel swaps the momentum of the idx-th particle in the bottom slab
 * with the momentum of the idx-th particle in the top slab. Since both arrays
 * are sorted according to their absolute momentum, the particle with the
 * highest momentum in +x-direction in the bottom slab swaps momentum with
 * the particle with the highest momentum in -x-direction in the top slab.
 * In total, num_pairs swaps are performed.
 */
__global__ void swap_momentum_pairs(const Scalar2 *d_layer_hi,
                                    const Scalar2 *d_layer_lo,
                                    Scalar4 *d_vel,
                                    const unsigned int *d_member_idx,
                                    const unsigned int num_pairs)
    {
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_pairs) return;

    // load data from the lo slab
    const Scalar2 lo = d_layer_lo[idx];
    const unsigned int lo_idx = __scalar_as_int(lo.x);
    const Scalar lo_momentum = lo.y;
    // load data from the hi slab
    const Scalar2 hi = d_layer_hi[idx];
    const unsigned int hi_idx = __scalar_as_int(hi.x);
    const Scalar hi_momentum = hi.y;

    // swap velocities
    d_vel[lo_idx].x =  hi_momentum / d_vel[lo_idx].w;
    d_vel[hi_idx].x =  lo_momentum / d_vel[hi_idx].w;
    }

/*!
 * \param d_slab_pairs Sorted Array of (signed flags,momentum) for particles in both slabs
 * \param d_layer_hi Array for (index,momentum) of top slab
 * \param d_layer_lo Array for (index,momentum) of bottom slab
 * \param num_hi_entries number of entries in the top slab ( up to m_num_swap)
 * \param Nslab total number of particles in both slabs
 * \param num_threads Number of threads, num_lo_entries + num_hi_entries
 *
 * This kernel copies the first \a num_hi_entries out of \a d_slab_pairs into
 * \a d_layer_hi and the last \a num_lo_entries into \a d_layer_lo while flipping
 * their order, so that both \a d_layer_lo and \a d_layer_hi are sorted descending
 * according to the absolute momentum.
 */
__global__ void divide_pair_array(const Scalar2 *d_slab_pairs,
                                  Scalar2 *d_layer_hi,
                                  Scalar2 *d_layer_lo,
                                  const unsigned int num_hi_entries,
                                  const unsigned int Nslab,
                                  const unsigned int num_threads)
    {
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_threads) return;

    if ( idx < num_hi_entries )  // top slab
        {
        Scalar2 current = d_slab_pairs[idx];
        int index = __scalar_as_int(current.x);
        d_layer_hi[idx] =  make_scalar2(__int_as_scalar(-index-1),current.y);
        }
    else   // bottom slab
        {
        // last element of d_slab_pairs has index Nslab-1
        Scalar2 current = d_slab_pairs[Nslab-1-(idx-num_hi_entries)];
        int index = __scalar_as_int(current.x);
        d_layer_lo[idx-num_hi_entries] =  make_scalar2(__int_as_scalar(index-1),current.y);
        }
    }
} // end namespace kernel

/*!
 * \param d_slab_pairs Empty array for holding the (signed flags,momentum) of all particles in either slab
 * \param d_pos  Array of positions of particles
 * \param d_vel  Array of velocities of particles
 * \param d_member_idx  indices of particles in the group
 * \param m_lo_pos edges of the bottom slab
 * \param m_hi_pos edges of the top slab
 * \param N  Number of particles in the group
 * \param block_size Number of threads per block
 *
 * Particles are marked according to their position in the box and their momentum
 * and the flags and momentum are written into \a d_slab_pairs.
 * See kernel::mark_particles_in_slabs for details.
 */
cudaError_t mark_particles_in_slabs(Scalar2 *d_slab_pairs,
                                    const Scalar4 *d_pos,
                                    const Scalar4 *d_vel,
                                    const unsigned int *d_member_idx,
                                    const Scalar2 d_lo_pos,
                                    const Scalar2 d_hi_pos,
                                    const unsigned int N,
                                    const unsigned int block_size)
    {
    if (N == 0) return cudaSuccess;

    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)kernel::mark_particles_in_slabs);
        max_block_size = attr.maxThreadsPerBlock;
        }
    const int run_block_size = min(block_size, max_block_size);

    kernel::mark_particles_in_slabs<<<N/run_block_size + 1, run_block_size>>>(d_slab_pairs,
                                                                              d_pos,
                                                                              d_vel,
                                                                              d_member_idx,
                                                                              d_lo_pos,
                                                                              d_hi_pos,
                                                                              N);
    return cudaSuccess;
    }


/*!
 * \param d_slab_pairs Array of (signed flags,momentum) for particles in both slabs
 * \param Nslab number of entries in \a d_slab_pairs array
 * \param p_target target momentum for momentum sorting
 *  Sort \a d_slab_pairs array with custom compare operator.
 *
 */
cudaError_t sort_pair_array(Scalar2 *d_slab_pairs,
                            const unsigned int Nslab,
                            Scalar p_target)
    {
    if (Nslab == 0) return cudaSuccess;

    // wrapper for pointer needed for thrust
    thrust::device_ptr<Scalar2> d_pairs_wrap(d_slab_pairs);
    // sort pairs according to their sign and momentum
    thrust::sort(d_pairs_wrap,d_pairs_wrap+Nslab, detail::ReversePerturbationSorter(p_target));
    return cudaSuccess;
    }

//! Functor to select elements from a Scalar2 where the x-component is nonzero.
struct NotZero
    {
    //! Selection operator
    /*!
     * \param in0  Scalar2 input
     * \returns True if \a in0.x is not zero.
     */
    __host__ __device__ __forceinline__
    bool operator()(Scalar2 &in) const
        {
        return (__scalar_as_int(in.x) != 0);
        }
    };

/*!
 * \param d_num_mark GPUFlag for saving the number of selected particles
 * \param d_tmp_storage Temporary storage allocated on the device, NULL on first call
 * \param tmp_storage_bytes Number of bytes necessary for temporary storage, 0 on first call
 * \param d_slab_pairs Arrays of (signed flags,momentum) identifying which
 *        particles are in either slab = +/- (index+1) or in no slab = 0
 * \param N Number of particles
 *
 * The CUB library is used to compact the particle indexes of the selected particles
 * into \a d_slab_pairs based on the flags set in \a d_slab_pairs. The number of selected
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
 * See kernel::mark_particles_in_slabs for details of how particles are marked to be in either slab.
 *
 */
cudaError_t select_particles_in_slabs(unsigned int *d_num_mark,
                                      void *d_tmp_storage,
                                      size_t &tmp_storage_bytes,
                                      Scalar2 *d_slab_pairs,
                                      const unsigned int N)
    {
    if (N == 0) return cudaSuccess;
    cub::DeviceSelect::If(d_tmp_storage, tmp_storage_bytes, d_slab_pairs, d_slab_pairs, d_num_mark, N,  NotZero());
    return cudaSuccess;
    }

/*!
 * \param d_slab_pairs Sorted array of (signed flags,momentum ) for particles in either slab
 * \param d_layer_hi Array for (index,momentum) of particles in bottom slab
 * \param d_layer_lo  Array for (index,momentum) of  particles in top slab
 * \param num_hi_entries Number of entries in top slab, which is m_num_hi up to m_num_slab.
 * \param num_lo_entries Number of entries in bottom slab, which is m_num_lo up to m_num_slab.
 * \param Nslab  Number of particles in either slab
 * \param num_threads  Number of spawned threads (num_hi_entries + num_lo_entries)
 * \param block_size  Number of threads per block
 *
 * This function takes the array \a d_slab_pairs, containig all particles in either slab
 * and divides the entries into the arrays \a d_layer_hi and \a d_layer_lo  depending on
 * their flag sign.
 *
 */
cudaError_t divide_pair_array(const Scalar2 *d_slab_pairs,
                             Scalar2 *d_layer_hi,
                             Scalar2 *d_layer_lo,
                             const unsigned int num_hi_entries,
                             const unsigned int Nslab,
                             const unsigned int num_threads,
                             const unsigned int block_size)
    {
    if (Nslab == 0 ) return cudaSuccess;

    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)kernel::divide_pair_array);
        max_block_size = attr.maxThreadsPerBlock;
        }

    const int run_block_size = min(block_size, max_block_size);
    kernel::divide_pair_array<<<num_threads/run_block_size+1, run_block_size>>>(d_slab_pairs,
                                                                                d_layer_hi,
                                                                                d_layer_lo,
                                                                                num_hi_entries,
                                                                                Nslab,
                                                                                num_threads);
    return cudaSuccess;
    }

/*!
 *
 * \param m_split   GPUFlag for the number of particles in the top slab
 * \param d_slab_pairs Sorted array with (signed flags,momentum) of all particles in either slab
 * \param Nslab   Number of particles in either slab
 * \param block_size  Number of threads per block
 *
 *  The function determines how many entries there are in the top slab.
 *  Nslab-1 threads are used, each one looks to its right neighbour and
 *  compares the sign of their entries, if its different, the top array
 *  ends at that position.
 *
 */
cudaError_t find_split_array(unsigned int *d_split,
                             int *d_type,
                             const Scalar2 *d_slab_pairs,
                             const unsigned int Nslab,
                             const unsigned int block_size)
    {
    if (Nslab == 0) return cudaSuccess;

    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)kernel::find_split_array);
        max_block_size = attr.maxThreadsPerBlock;
        }
    const int run_block_size = min(block_size, max_block_size);
    kernel::find_split_array<<<(Nslab-1)/run_block_size+1, run_block_size>>>(d_split,
                                                                             d_type,
                                                                             d_slab_pairs,
                                                                             Nslab-1);
    return cudaSuccess;
    }

/*!
 *
 * \param m_layer_hi Sorted array with (signed flags,momentum) of all particles in top slab
 * \param m_layer_lo Sorted array with (signed flags,momentum) of all particles in bottom slab
 * \param d_vel  Velocities of particles
 * \param d_member_idx  Indices of particles in group
 * \param num_pairs maximum number of swaps
 * \param block_size  Number of threads per block
 *
 *  The function determines how many entries there are in the top slab.
 *  Nslab-1 threads are used, each one looks to its right neighbour and
 *  compares the sign of their entries, if its different, the top array
 *  ends at that position.
 *
 */
cudaError_t swap_momentum_pairs(const Scalar2 *d_layer_hi,
                                const Scalar2 *d_layer_lo,
                                Scalar4 *d_vel,
                                const unsigned int *d_member_idx,
                                const unsigned int num_pairs,
                                const unsigned int block_size)
    {
    if (num_pairs == 0) return cudaSuccess;

    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)kernel::swap_momentum_pairs);
        max_block_size = attr.maxThreadsPerBlock;
        }
    const int run_block_size = min(block_size, max_block_size);
    kernel::swap_momentum_pairs<<<num_pairs/run_block_size+1, run_block_size>>>(d_layer_hi,
                                                                                d_layer_lo,
                                                                                d_vel,
                                                                                d_member_idx,
                                                                                num_pairs);
    return cudaSuccess;
    }

//! Transforms a Scalar2 to return the momentum from the y-component
struct GetMomentum
    {
    //! Transform operator
    /*!
     * \param in0 Scalar2 entry with (tag,momentum)
     *
     * \returns The y-component (=momentum) of a Scalar2
     */
    __host__ __device__ __forceinline__
    Scalar operator()(Scalar2 &in0) const
        {
        return in0.y;
        }
    };

/*!
 * \param m_layer_hi Sorted array with (tag,momentum) of all particles in top slab
 * \param m_layer_lo Sorted array with (tag,momentum) of all particles in bottom slab
 * \param num_pairs number of swaps
 *
 * Calculate the total momentum exchange for the current set of swaps by
 * performing two thrust::transform_reduce operations on both \a m_layer_hi and
 * \a m_layer_lo. The difference is the momentum exchange.
 */
Scalar calc_momentum_exchange(Scalar2 *d_layer_hi,
                              Scalar2 *d_layer_lo,
                              const unsigned int num_pairs)
    {

    if (num_pairs == 0) return 0.0;

    thrust::device_ptr<Scalar2> t_layer_hi = thrust::device_pointer_cast(d_layer_hi);
    thrust::device_ptr<Scalar2> t_layer_lo = thrust::device_pointer_cast(d_layer_lo);
    // transform the Scalar2 to a Scalar (y-component,momentum) and then
    // sum it up for each layer
    Scalar momentum_layer_hi = thrust::transform_reduce(t_layer_hi,
                                                        t_layer_hi + num_pairs,
                                                        GetMomentum(),
                                                        0.0,
                                                        thrust::plus<float>());
    Scalar momentum_layer_lo = thrust::transform_reduce(t_layer_lo,
                                                        t_layer_lo + num_pairs,
                                                        GetMomentum(),
                                                        0.0,
                                                        thrust::plus<float>());

    Scalar momentum_exchange = momentum_layer_lo - momentum_layer_hi;
    return momentum_exchange;
    }
} // end namespace gpu
} // end namespace azplugins
