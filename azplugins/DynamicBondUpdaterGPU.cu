// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt

/*!
 * \file DynamicBondUpdaterGPU.cu
 * \brief Definition of kernel drivers and kernels for DynamicBondUpdaterGPU
 */

#include "hoomd/HOOMDMath.h"
#include "DynamicBondUpdaterGPU.cuh"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
// todo: should azplugins have its own "extern"?

#include "hoomd/extern/neighbor/neighbor/LBVH.cuh"
#include "hoomd/extern/neighbor/neighbor/LBVHTraverser.cuh"
#include "hoomd/extern/cub/cub/cub.cuh"

#include <iostream>

namespace azplugins
{

//todo: migrate to separate file/class.
struct SortBondsGPUDistance{
  __host__ __device__ bool operator()(const Scalar3 &in0, const Scalar3 &in1)
    {
      const Scalar r_sq_1 = in0.z;
      const Scalar r_sq_2 = in1.z;
      const unsigned int tag_0 = __scalar_as_int(in0.x);
      const unsigned int tag_1 = __scalar_as_int(in1.x);
      // todo: is this necessary for the thrust::unique to filter out all dublicates or
      // would r_sq_1 < r_sq_2 be enough? What happens if two different potential
      // bonds have EXACTLY same length?
      if (r_sq_1 == r_sq_2)
          return tag_0 < tag_1;
      return r_sq_1 < r_sq_2;
    }
};

struct SortBondsGPUFirstTag{
  __host__ __device__ bool operator()(const Scalar3 &in0, const Scalar3 &in1)
    {
      const unsigned int tag_0 = __scalar_as_int(in0.x);
      const unsigned int tag_1 = __scalar_as_int(in1.x);
      return tag_0 < tag_1;
    }
};

struct SortBondsGPUSecondTag{
  __host__ __device__ bool operator()(const Scalar3 &in0, const Scalar3 &in1)
    {
      const unsigned int tag_0 = __scalar_as_int(in0.y);
      const unsigned int tag_1 = __scalar_as_int(in1.y);
      return tag_0 < tag_1;
    }
};

struct isZeroBondGPU{
  __host__ __device__ bool operator()(const Scalar3 &in0)
    {
      const unsigned int tag_0 = __scalar_as_int(in0.x);
      const unsigned int tag_1 = __scalar_as_int(in0.y);
      if ( tag_0==0 && tag_1 ==0)
      {
        return true;
      }
      else
      {
        return false;
      }
    }
};

struct CompareBondsGPU{
  __host__ __device__ bool operator()(const Scalar3 &in0, const Scalar3 &in1)
    {
        const unsigned int tag_11 = __scalar_as_int(in0.x);
        const unsigned int tag_12 = __scalar_as_int(in0.y);
        const unsigned int tag_21 = __scalar_as_int(in1.x);
        const unsigned int tag_22 = __scalar_as_int(in1.y);

        if ((tag_11==tag_21 && tag_12==tag_22) ||   // (i,j)==(i,j)
            (tag_11==tag_22 && tag_12==tag_21))     // (i,j)==(j,i)
        {
          return true;
        }
        else
        {
          return false;
        }
      }
  };

namespace gpu
{

//! Number of elements of the exisitng bond list to process in each batch
const unsigned int FILTER_BATCH_SIZE = 4;

namespace kernel
{

/*!
  This kernel is modeled after the neighbor list exclusions filtering mechanism.
*/
__global__ void filter_existing_bonds(Scalar3 *d_all_possible_bonds,
                                      const unsigned int *d_n_existing_bonds,
                                      const unsigned int *d_existing_bonds_list,
                                      const Index2D exli,
                                      const unsigned int size,
                                      const unsigned int ex_start)
  {
  // compute the bond index this thread operates on
  const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

  // quit now if this thread is processing past the end of the list of all possible bonds
  if (idx >= size)
      return;

  Scalar3 current_bond = d_all_possible_bonds[idx];
  unsigned int tag_1 = __scalar_as_int(current_bond.x);
  unsigned int tag_2 = __scalar_as_int(current_bond.y);

  if(tag_1==0 && tag_2==0)
      return;

  //const unsigned int n_neigh = d_n_neigh[idx];
  const unsigned int n_ex = d_n_existing_bonds[tag_1];

  // quit now if the ex_start flag is past the end of n_ex
  if (ex_start >= n_ex)
      return;

//  printf("in filter_existing_bonds idx %d tag_i %d tag_j %d dist %f \n",idx,tag_1,tag_2,current_bond.z);

  // count the number of existing bonds to process in this thread
  const unsigned int n_ex_process = n_ex - ex_start;

  // load the existing bond list into "local" memory - fully unrolled loops should dump this into registers
  unsigned int l_existing_bonds_list[FILTER_BATCH_SIZE];
  #pragma unroll
  for (unsigned int cur_ex_idx = 0; cur_ex_idx < FILTER_BATCH_SIZE; cur_ex_idx++)
      {
    //  printf("in filter_existing_bonds cur_ex_idx %d \n",cur_ex_idx);
      if (cur_ex_idx < n_ex_process)
          l_existing_bonds_list[cur_ex_idx] = d_existing_bonds_list[exli(tag_1, cur_ex_idx + ex_start)];
      else
          l_existing_bonds_list[cur_ex_idx] = 0xffffffff;
      //  printf("in filter_existing_bonds idx %d tag_i %d tag_j %d dist %f cur_ex_idx %d l_existing_bonds_list %d \n",idx,tag_1,tag_2,current_bond.z,cur_ex_idx,l_existing_bonds_list[cur_ex_idx]);
      }

      // test if excluded
      bool excluded = false;
      #pragma unroll
      for (unsigned int cur_ex_idx = 0; cur_ex_idx < FILTER_BATCH_SIZE; cur_ex_idx++)
          {
          if (tag_2 == l_existing_bonds_list[cur_ex_idx])
              excluded = true;
          }

      // if it is excluded, overwrite that entry with (0,0,0).
      if (excluded)
          {
            d_all_possible_bonds[idx] = make_scalar3(0,0,0.0);
          }

  }

  //! Kernel to copy the particle indexes into traversal order
  /*!
   * \param d_traverse_order List of particle indexes in traversal order.
   * \param d_indexes Original indexes of the sorted primitives.
   * \param d_primitives List of the primitives (sorted in LBVH order).
   * \param N Number of primitives.
   *
   * The primitive index for this thread is first loaded. It is then mapped back
   * to its original particle index, which is stored for subsequent traversal.
   */
  __global__ void gpu_nlist_copy_primitives_kernel(unsigned int *d_traverse_order,
                                                   const unsigned int *d_indexes,
                                                   const unsigned int *d_primitives,
                                                   const unsigned int N)
      {
      // one thread per particle
      const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
      if (idx >= N)
          return;

      const unsigned int primitive = d_primitives[idx];
      d_traverse_order[idx] = __ldg(d_indexes + primitive);
      }

  __global__ void make_sorted_index_array(  unsigned int *d_sorted_indexes,
                                          unsigned int *d_indexes_group_1,
                                          unsigned int *d_indexes_group_2,
                                          const unsigned int size_group_1,
                                          const unsigned int size_group_2)
      {
      // one thread per element in d_sorted_indexes
      const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
      if (idx >= size_group_1+size_group_2)
          return;

      if (idx<size_group_1){
          d_sorted_indexes[idx] = d_indexes_group_1[idx];
      }
      else{
          d_sorted_indexes[idx] = d_indexes_group_2[idx-size_group_1];
         }

      }

  __global__ void  nlist_copy_nlist_possible_bonds(Scalar3 *d_all_possible_bonds,
                                const Scalar4 *d_postype,
                                const unsigned int * d_tag,
                                const unsigned int * d_sorted_indexes,
                                const unsigned int * d_n_neigh,
                                const unsigned int * d_nlist,
                                const BoxDim box,
                                const unsigned int max_bonds,
                                const Scalar r_cut,
                                const unsigned int size)
      {
        // one thread per particle in group_1
        const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= size)
            return;

        // idx = group index , pidx = actual particle index
        const unsigned int pidx_i = d_sorted_indexes[idx];

        // get all information for this particle
        Scalar4 postype_i = d_postype[pidx_i];
        const unsigned int tag_i = d_tag[pidx_i];
        const unsigned int n_neigh = d_n_neigh[pidx_i];

        // loop over all neighbors of this particle
        for (unsigned int j=0; j<n_neigh;++j)
          {
              // get index of neighbor from neigh_list
              const unsigned int pidx_j = d_nlist[ pidx_i*max_bonds + j];
              Scalar4 postype_j = d_postype[pidx_j];
              const unsigned int tag_j = d_tag[pidx_j];
              //todo: shouldn't be this test be already taken care of with ParticleQueryOp refine?
              if (tag_i != tag_j){

               Scalar3 drij = make_scalar3(postype_j.x,postype_j.y,postype_j.z)
                            - make_scalar3(postype_i.x,postype_i.y,postype_i.z);

             // apply periodic boundary conditions (FLOPS: 12)
              drij = box.minImage(drij);

               Scalar dr_sq = dot(drij,drij);
               if (dr_sq<=r_cut*r_cut)
               {
                 //printf("nlist_copy_nlist_possible_bonds particle ij %d %d %d %d %f \n",tag_i,tag_j,__scalar_as_int(postype_i.w),__scalar_as_int(postype_j.w),fast::sqrt(dr_sq));
                 Scalar3 d = make_scalar3(__int_as_scalar(tag_i),__int_as_scalar(tag_j),dr_sq);
                 d_all_possible_bonds[idx + n_neigh] = d;
               }

             }

          }

      }


} //end namespace kernel


cudaError_t sort_and_remove_zeros_possible_bond_array(Scalar3 *d_all_possible_bonds,
                                             const unsigned int size,
                                            int *d_max_non_zero_bonds)
    {
    if (size == 0) return cudaSuccess;
    // wrapper for pointer needed for thrust
    thrust::device_ptr<Scalar3> d_all_possible_bonds_wrap(d_all_possible_bonds);

    // first remove all zeros
    isZeroBondGPU zero;
    thrust::device_ptr<Scalar3> last0 = thrust::remove_if(d_all_possible_bonds_wrap,d_all_possible_bonds_wrap+size, zero);
    unsigned int l0 = thrust::distance(d_all_possible_bonds_wrap, last0);

    // sort remainder by distance, should make all identical bonds consequtive
    SortBondsGPUDistance sort;
    thrust::sort(d_all_possible_bonds_wrap,d_all_possible_bonds_wrap+l0, sort);
    CompareBondsGPU comp;

    // thrust::unique only removes identical consequtive elements, so sort above is needed.
    thrust::device_ptr<Scalar3> last1 = thrust::unique(d_all_possible_bonds_wrap, d_all_possible_bonds_wrap + l0,comp);
    unsigned int l1 = thrust::distance(d_all_possible_bonds_wrap, last1);

    *d_max_non_zero_bonds=l1;

    return cudaSuccess;
    }

cudaError_t remove_zeros_possible_bond_array(Scalar3 *d_all_possible_bonds,
                                             const unsigned int size,
                                            int *d_max_non_zero_bonds)
    {
    if (size == 0) return cudaSuccess;
    // wrapper for pointer needed for thrust
    thrust::device_ptr<Scalar3> d_all_possible_bonds_wrap(d_all_possible_bonds);
    // remove all zeros
    isZeroBondGPU zero;
    thrust::device_ptr<Scalar3> last0 = thrust::remove_if(d_all_possible_bonds_wrap,d_all_possible_bonds_wrap+size, zero);
    unsigned int l0 = thrust::distance(d_all_possible_bonds_wrap, last0);
    *d_max_non_zero_bonds=l0;

    return cudaSuccess;
    }


cudaError_t filter_existing_bonds(Scalar3 *d_all_possible_bonds,
                             unsigned int *d_n_existing_bonds,
                             const unsigned int *d_existing_bonds_list,
                             const Index2D& exli,
                             const unsigned int size,
                             const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)kernel::filter_existing_bonds);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);

    // determine parameters for kernel launch
    int n_blocks = size/run_block_size + 1;

    // split the processing of the full exclusion list up into a number of batches
    unsigned int n_batches = (unsigned int)ceil(double(exli.getH())/double(FILTER_BATCH_SIZE));
    unsigned int ex_start = 0;
    for (unsigned int batch = 0; batch < n_batches; batch++)
        {
        kernel::filter_existing_bonds<<<n_blocks, run_block_size>>>(d_all_possible_bonds,
                                                              d_n_existing_bonds,
                                                              d_existing_bonds_list,
                                                              exli,
                                                              size,
                                                              ex_start);

        ex_start += FILTER_BATCH_SIZE;
        }

    return cudaSuccess;
    }

cudaError_t make_sorted_index_array( unsigned int *d_sorted_indexes,
                                 unsigned int *d_indexes_group_1,
                                 unsigned int *d_indexes_group_2,
                                 const unsigned int size_group_1,
                                 const unsigned int size_group_2,
                                 const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)kernel::make_sorted_index_array);
        max_block_size = attr.maxThreadsPerBlock;
        }

    const unsigned int run_block_size = min(block_size,max_block_size);
    const unsigned int num_blocks = ((size_group_1+size_group_2) + run_block_size - 1)/run_block_size;

    kernel::make_sorted_index_array<<<num_blocks, run_block_size>>>(d_sorted_indexes,
                                                                d_indexes_group_1,
                                                                d_indexes_group_2,
                                                                size_group_1,
                                                                size_group_2);

    return cudaSuccess;
    }


/*!
 * \param d_traverse_order List of particle indexes in traversal order.
 * \param d_indexes Original indexes of the sorted primitives.
 * \param d_primitives List of the primitives (sorted in LBVH order).
 * \param N Number of primitives.
 * \param block_size Number of CUDA threads per block.
 *
 * \sa gpu_nlist_copy_primitives_kernel
 */
cudaError_t gpu_nlist_copy_primitives(unsigned int *d_traverse_order,
                                      const unsigned int *d_indexes,
                                      const unsigned int *d_primitives,
                                      const unsigned int N,
                                      const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)kernel::gpu_nlist_copy_primitives_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    int run_block_size = min(block_size,max_block_size);
    kernel::gpu_nlist_copy_primitives_kernel<<<N/run_block_size + 1, run_block_size>>>(d_traverse_order,
                                                                               d_indexes,
                                                                               d_primitives,
                                                                               N);
    return cudaSuccess;
    }


cudaError_t nlist_copy_nlist_possible_bonds(Scalar3 *d_all_possible_bonds,
                          const Scalar4 *d_postype,
                          const unsigned int *d_tag,
                          const unsigned int *d_sorted_indexes,
                          const unsigned int *d_n_neigh,
                          const unsigned int *d_nlist,
                          const BoxDim box,
                          const unsigned int max_bonds,
                          const Scalar r_cut,
                          const unsigned int size,
                          const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
    {
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void *)kernel::nlist_copy_nlist_possible_bonds);
    max_block_size = attr.maxThreadsPerBlock;
    }

    int run_block_size = min(block_size,max_block_size);
    kernel::nlist_copy_nlist_possible_bonds<<<size/run_block_size + 1, run_block_size>>>(d_all_possible_bonds,
                                                                         d_postype,
                                                                         d_tag,
                                                                         d_sorted_indexes,
                                                                         d_n_neigh,
                                                                         d_nlist,
                                                                         box,
                                                                         max_bonds,
                                                                         r_cut,
                                                                         size);
    return cudaSuccess;
    }

} // end namespace gpu
} // end namespace azplugins


// explicit templates for neighbor::LBVH with PointMapInsertOp
template void neighbor::gpu::lbvh_gen_codes(unsigned int *, unsigned int *, const azplugins::gpu::PointMapInsertOp&,
const Scalar3, const Scalar3, const unsigned int, const unsigned int, cudaStream_t);
template void neighbor::gpu::lbvh_bubble_aabbs(const neighbor::gpu::LBVHData, const azplugins::gpu::PointMapInsertOp&,
unsigned int *, const unsigned int, const unsigned int, cudaStream_t);
template void neighbor::gpu::lbvh_one_primitive(const neighbor::gpu::LBVHData, const azplugins::gpu::PointMapInsertOp&, cudaStream_t);
template void neighbor::gpu::lbvh_traverse_ropes(azplugins::gpu::NeighborListOp&, const neighbor::gpu::LBVHCompressedData&,
const azplugins::gpu::ParticleQueryOp&, const Scalar3 *, unsigned int, unsigned int, cudaStream_t);
