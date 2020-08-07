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


namespace azplugins
{

//todo: migrate to separate file/class.
//this sorts according distance, then first tag, then second tag
struct SortBondsGPU{
  __host__ __device__ bool operator()(const Scalar3 &i, const Scalar3 &j)
    {
      const Scalar r_sq_1 = i.z;
      const Scalar r_sq_2 = j.z;
      if (r_sq_1==r_sq_2)
      {
        const unsigned int tag_11 = __scalar_as_int(i.x);
        const unsigned int tag_21 = __scalar_as_int(j.x);
        if (tag_11==tag_21)
        {
        const unsigned int tag_12 = __scalar_as_int(i.y);
        const unsigned int tag_22 = __scalar_as_int(j.y);
        return tag_22>tag_12;
        }
        else
        {
          return tag_21>tag_11;
        }
      }
      else
      {
        return r_sq_2>r_sq_1;
      }
    }

};

// returns true if given possible bond is zero, e.g. (0,0,0.0)
// possible bonds are ordered, such that tag_a < tag_b in (tag_a,tag_b,rsq)
// meaning we only need to check tag_b == 0
struct isZeroBondGPU{
  __host__ __device__ bool operator()(const Scalar3 &i)
    {
      const unsigned int tag_1 = __scalar_as_int(i.y);
      return !(bool)tag_1;
    }
};

struct CompareBondsGPU{
  __host__ __device__ bool operator()(const Scalar3 &i, const Scalar3 &j)
    {
      const unsigned int tag_11 = __scalar_as_int(i.x);
      const unsigned int tag_12 = __scalar_as_int(i.y);
      const unsigned int tag_21 = __scalar_as_int(j.x);
      const unsigned int tag_22 = __scalar_as_int(j.y);

      if ((tag_11==tag_21 && tag_12==tag_22))   // should work because pairs are ordered
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

  // count the number of existing bonds to process in this thread
  const unsigned int n_ex_process = n_ex - ex_start;

  // load the existing bond list into "local" memory - fully unrolled loops should dump this into registers
  unsigned int l_existing_bonds_list[FILTER_BATCH_SIZE];
  #pragma unroll
  for (unsigned int cur_ex_idx = 0; cur_ex_idx < FILTER_BATCH_SIZE; cur_ex_idx++)
      {
      if (cur_ex_idx < n_ex_process)
          l_existing_bonds_list[cur_ex_idx] = d_existing_bonds_list[exli(tag_1, cur_ex_idx + ex_start)];
      else
          l_existing_bonds_list[cur_ex_idx] = 0xffffffff;
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
        unsigned int n_curr_bond = 0;
        const Scalar r_cutsq = r_cut*r_cut;

        // get all information for this particle
        Scalar4 postype_i = d_postype[pidx_i];
        const unsigned int tag_i = d_tag[pidx_i];
        const unsigned int n_neigh = d_n_neigh[pidx_i];

        // loop over all neighbors of this particle
        for (unsigned int j=0; j<n_neigh;++j)
          {
              // get index of neighbor from neigh_list
              const unsigned int pidx_j = d_nlist[pidx_i*max_bonds + j];
              Scalar4 postype_j = d_postype[pidx_j];
              const unsigned int tag_j = d_tag[pidx_j];

              Scalar3 drij = make_scalar3(postype_j.x,postype_j.y,postype_j.z)
                           - make_scalar3(postype_i.x,postype_i.y,postype_i.z);

             // apply periodic boundary conditions (FLOPS: 12)
              drij = box.minImage(drij);

              // same as on the cpu, just not during the tree traversal
               Scalar dr_sq = dot(drij,drij);

               if (dr_sq < r_cutsq)
                   {
                   if (n_curr_bond < max_bonds)
                      {
                      // sort the two tags in this possible bond pair
                      const unsigned int tag_a = tag_j>tag_i ? tag_i : tag_j;
                      const unsigned int tag_b = tag_j>tag_i ? tag_j : tag_i;
                      Scalar3 d = make_scalar3(__int_as_scalar(tag_a),__int_as_scalar(tag_b),dr_sq);
                      d_all_possible_bonds[idx*max_bonds + n_curr_bond] = d;
                      }
                    ++n_curr_bond;
                  }

          }

      }


} //end namespace kernel

/*
profiling results: sort - find_if - unique 34.5 %
                  remove_if -> sort -> unique 14.6%
sort is slow. can we use Radix_sort instead? need keys. cantor pairing function?

*/
cudaError_t sort_and_remove_zeros_possible_bond_array_1(Scalar3 *d_all_possible_bonds,
                                             const unsigned int size,
                                            int *d_max_non_zero_bonds)
    {
    if (size == 0) return cudaSuccess;
    // wrapper for pointer needed for thrust
    thrust::device_ptr<Scalar3> d_all_possible_bonds_wrap(d_all_possible_bonds);

    // first remove all zeros - makes sort after faster
    isZeroBondGPU zero;
    thrust::device_ptr<Scalar3> last0 = thrust::remove_if(d_all_possible_bonds_wrap,d_all_possible_bonds_wrap+size, zero);
    unsigned int l0 = thrust::distance(d_all_possible_bonds_wrap, last0);

    *d_max_non_zero_bonds=l0;

    return cudaSuccess;
    }

    cudaError_t sort_and_remove_zeros_possible_bond_array_2(Scalar3 *d_all_possible_bonds,
                                                 const unsigned int size,
                                                int *d_max_non_zero_bonds)
        {
        if (size == 0) return cudaSuccess;
        // wrapper for pointer needed for thrust
        thrust::device_ptr<Scalar3> d_all_possible_bonds_wrap(d_all_possible_bonds);

        // sort remainder by distance, should make all identical bonds consequtive
        SortBondsGPU sort;
        thrust::sort(thrust::device,d_all_possible_bonds_wrap,d_all_possible_bonds_wrap+size, sort);

        *d_max_non_zero_bonds=size;

        return cudaSuccess;
        }


cudaError_t sort_and_remove_zeros_possible_bond_array_3(Scalar3 *d_all_possible_bonds,
                                                 const unsigned int size,
                                                int *d_max_non_zero_bonds)
      {
      if (size == 0) return cudaSuccess;
      // wrapper for pointer needed for thrust
      thrust::device_ptr<Scalar3> d_all_possible_bonds_wrap(d_all_possible_bonds);


      CompareBondsGPU comp;
      // thrust::unique only removes identical consequtive elements, so sort above is needed.
      thrust::device_ptr<Scalar3> last1 = thrust::unique(d_all_possible_bonds_wrap, d_all_possible_bonds_wrap + size,comp);
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
