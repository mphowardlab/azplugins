// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file ParticleEvaporatorGPU.cuh
 * \brief Declaration of kernel drivers for ParticleEvaporatorGPU
 */

#ifndef AZPLUGINS_PARTICLE_EVAPORATOR_GPU_CUH_
#define AZPLUGINS_PARTICLE_EVAPORATOR_GPU_CUH_

#include "hoomd/HOOMDMath.h"
#include <cuda_runtime.h>

namespace azplugins
    {
namespace gpu
    {
//! Kernel driver to change particle types in a region on the GPU
cudaError_t evaporate_setup_mark(unsigned char* d_select_flags,
                                 unsigned int* d_mark,
                                 const Scalar4* d_pos,
                                 unsigned int solvent_type,
                                 Scalar z_lo,
                                 Scalar z_hi,
                                 unsigned int N,
                                 unsigned int block_size);

//! Drives CUB device selection routines for marked particles
cudaError_t evaporate_select_mark(unsigned int* d_mark,
                                  unsigned int* d_num_mark,
                                  void* d_tmp_storage,
                                  size_t& tmp_storage_bytes,
                                  const unsigned char* d_select_flags,
                                  unsigned int N);

//! Updates particles types according to picks made on cpu
cudaError_t evaporate_apply_picks(Scalar4* d_pos,
                                  const unsigned int* d_picks,
                                  const unsigned int* d_mark,
                                  unsigned int evaporated_type,
                                  unsigned int N_pick,
                                  unsigned int block_size);
    } // namespace gpu
    } // namespace azplugins

#endif // AZPLUGINS_PARTICLE_EVAPORATOR_GPU_CUH_
