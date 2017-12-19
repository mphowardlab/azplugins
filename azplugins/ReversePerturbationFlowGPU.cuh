// Copyright (c) 2016-2017, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt

/*!
 * \file ReversePerturbationFlowGPU.cuh
 * \brief Declaration of kernel drivers for ReversePerturbationFlowGPU
 */

#ifndef AZPLUGINS_REVERSE_PERTURBATION_FLOW_GPU_CUH_
#define AZPLUGINS_REVERSE_PERTURBATION_FLOW_GPU_CUH_

#include <cuda_runtime.h>
#include "hoomd/HOOMDMath.h"

namespace azplugins
{
namespace gpu
{
//! Kernel driver for marking particles in slabs
cudaError_t mark_particles_in_slabs(Scalar2 *d_slab_pairs,
                                    const Scalar4 *d_pos,
                                    const Scalar4 *d_vel,
                                    const unsigned int *d_member_idx,
                                    const Scalar2 slab_lo,
                                    const Scalar2 slab_hi,
                                    const unsigned int N,
                                    const unsigned int block_size);

//! Kernel driver for sorting the array
cudaError_t sort_pair_array(Scalar2 *d_slab_pairs,
                            const unsigned int Nslab,
                            Scalar p_target);

//! Kernel driver for selecting the particles in either slab
cudaError_t select_particles_in_slabs(unsigned int *d_num_mark,
                                      void *d_tmp_storage,
                                      size_t &tmp_storage_bytes,
                                      Scalar2 *d_slab_flags,
                                      const unsigned int N);

//! Kernel driver for swapping momentum
cudaError_t swap_momentum_pairs(const Scalar2 *d_layer_hi,
                                const Scalar2 *d_layer_lo,
                                Scalar4 *d_vel,
                                const unsigned int *d_member_idx,
                                const unsigned int num_pairs,
                                const unsigned int block_size);

//! Kernel driver for finding the split between top and bottom entries
cudaError_t find_split_array(unsigned int *d_split,
                             int *d_type,
                             const Scalar2 *d_slab_pairs,
                             const unsigned int Nslab,
                             const unsigned int block_size);

//! Kernel driver for deviding top and bottom slabs into seperate arrays
cudaError_t divide_pair_array(const Scalar2 *d_slab_pairs,
                              Scalar2 *d_layer_hi,
                              Scalar2 *d_layer_lo,
                              const unsigned int num_hi_entries,
                              const unsigned int Nslab,
                              const unsigned int num_threads,
                              const unsigned int block_size);

//! Kernel driver to calculate the momentum exchange
Scalar calc_momentum_exchange(Scalar2 *d_layer_hi,
                              Scalar2 *d_layer_lo,
                              const unsigned int num_pairs);
}
}
#endif // AZPLUGINS_REVERSE_PERTUBATION_FLOW_GPU_CUH_
