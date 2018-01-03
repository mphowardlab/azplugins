// Copyright (c) 2016-2018, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file ParticleEvaporatorGPU.cuh
 * \brief Declaration of kernel drivers for ParticleEvaporatorGPU
 */

#ifndef AZPLUGINS_PARTICLE_EVAPORATOR_GPU_CUH_
#define AZPLUGINS_PARTICLE_EVAPORATOR_GPU_CUH_

#include <cuda_runtime.h>
#include "hoomd/HOOMDMath.h"

namespace azplugins
{
namespace gpu
{
//! Kernel driver to change particle types in a region on the GPU
cudaError_t evaporate_setup_mark(unsigned char *d_select_flags,
                                 unsigned int *d_mark,
                                 const Scalar4 *d_pos,
                                 unsigned int solvent_type,
                                 Scalar z_lo,
                                 Scalar z_hi,
                                 unsigned int N,
                                 unsigned int block_size);

//! Drives CUB device selection routines for marked particles
cudaError_t evaporate_select_mark(unsigned int *d_mark,
                                  unsigned int *d_num_mark,
                                  void *d_tmp_storage,
                                  size_t &tmp_storage_bytes,
                                  const unsigned char *d_select_flags,
                                  unsigned int N);

//! Updates particles types according to picks made on cpu
cudaError_t evaporate_apply_picks(Scalar4 *d_pos,
                                  const unsigned int *d_picks,
                                  const unsigned int *d_mark,
                                  unsigned int evaporated_type,
                                  unsigned int N_pick,
                                  unsigned int block_size);
}
}

#endif // AZPLUGINS_PARTICLE_EVAPORATOR_GPU_CUH_
