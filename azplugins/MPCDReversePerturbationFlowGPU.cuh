// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

/*!
 * \file MPCDReversePerturbationFlowGPU.cuh
 * \brief Declaration of kernel drivers for MPCDReversePerturbationFlowGPU
 */

#ifndef AZPLUGINS_MPCD_REVERSE_PERTURBATION_FLOW_GPU_CUH_
#define AZPLUGINS_MPCD_REVERSE_PERTURBATION_FLOW_GPU_CUH_

#include <cuda_runtime.h>
#include "hoomd/HOOMDMath.h"

namespace azplugins
{
namespace gpu
{
//! Kernel driver for marking particles in slabs
cudaError_t mpcd_mark_particles_in_slabs(Scalar2 *d_slab_pairs,
                                         const Scalar4 *d_pos,
                                         const Scalar4 *d_vel,
                                         const Scalar mass,
                                         const Scalar2 slab_lo,
                                         const Scalar2 slab_hi,
                                         const unsigned int N,
                                         const unsigned int block_size);

//! Kernel driver for sorting the array
cudaError_t mpcd_sort_pair_array(Scalar2 *d_slab_pairs,
                                 const unsigned int Nslab,
                                 const Scalar p_target);

//! Kernel driver for selecting the particles in either slab
cudaError_t mpcd_select_particles_in_slabs(unsigned int *d_num_mark,
                                           void *d_tmp_storage,
                                           size_t &tmp_storage_bytes,
                                           Scalar2 *d_slab_flags,
                                           const unsigned int N);

//! Kernel driver for swapping momentum
cudaError_t mpcd_swap_momentum_pairs(const Scalar2 *d_layer_hi,
                                     const Scalar2 *d_layer_lo,
                                     Scalar4 *d_vel,
                                     Scalar mass,
                                     const unsigned int num_pairs,
                                     const unsigned int block_size);

//! Kernel driver for finding the split between top and bottom entries
cudaError_t mpcd_find_split_array(unsigned int *d_split,
                                  int *d_type,
                                  const Scalar2 *d_slab_pairs,
                                  const unsigned int Nslab,
                                  const unsigned int block_size);

//! Kernel driver for deviding top and bottom slabs into seperate arrays
cudaError_t mpcd_divide_pair_array(const Scalar2 *d_slab_pairs,
                                   Scalar2 *d_layer_hi,
                                   Scalar2 *d_layer_lo,
                                   const unsigned int num_hi_entries,
                                   const unsigned int Nslab,
                                   const unsigned int num_threads,
                                   const unsigned int block_size);

//! Kernel driver to calculate the momentum exchange
Scalar mpcd_calc_momentum_exchange(Scalar2 *d_layer_hi,
                                   Scalar2 *d_layer_lo,
                                   const unsigned int num_pairs);
}
}
#endif // AZPLUGINS_REVERSE_PERTURBATION_FLOW_GPU_CUH_
