// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt

/*!
 * \file DynamicBondUpdaterGPU.cu
 * \brief Definition of kernel drivers and kernels for DynamicBondUpdaterGPU
 */

#include "DynamicBondUpdaterGPU.cuh"
#include <thrust/sort.h>
#include <thrust/device_vector.h>


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
      if (r_sq_1 ==r_sq_2)
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

/*! \param d_n_neigh Number of neighbors for each particle (read/write)
  \param d_nlist Neighbor list for each particle (read/write)
  \param nli Indexer for indexing into d_nlist
  \param d_n_ex Number of exclusions for each particle
  \param d_ex_list List of exclusions for each particle
  \param exli Indexer for indexing into d_ex_list
  \param N Number of particles
  \param ex_start Start filtering the nlist from exclusion number \a ex_start

  gpu_nlist_filter_kernel() processes the neighbor list \a d_nlist and removes any entries that are excluded. To allow
  for an arbitrary large number of exclusions, these are processed in batch sizes of FILTER_BATCH_SIZE. The kernel
  must be called multiple times in order to fully remove all exclusions from the nlist.

  \note The driver gpu_nlist_filter properly makes as many calls as are necessary, it only needs to be called once.

  \b Implementation

  One thread is run for each particle. Exclusions \a ex_start, \a ex_start + 1, ... are loaded in for that particle
  (or the thread returns if there are no exclusions past that point). The thread then loops over the neighbor list,
  comparing each entry to the list of exclusions. If the entry is not excluded, it is written back out. \a d_n_neigh
  is updated to reflect the current number of particles in the list at the end of the kernel call.
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
  unsigned int new_n_neigh = 0;

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

      // add this entry back to the list if it is not excluded
      if (excluded)
          {
            d_all_possible_bonds[idx] = make_scalar3(0,0,0.0);
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

    // thrust::unique only removes identical consequtive elements.
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


} // end namespace gpu
} // end namespace azplugins
