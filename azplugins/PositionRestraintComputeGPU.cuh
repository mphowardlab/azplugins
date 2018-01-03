// Copyright (c) 2016-2018, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: wes_reinhart

/*!
 * \file PositionRestraintComputeGPU.cuh
 * \brief Declares CUDA kernels for PositionRestraintComputeGPU
 */

#ifndef AZPLUGINS_POSITION_RESTRAINT_COMPUTE_GPU_CUH_
#define AZPLUGINS_POSITION_RESTRAINT_COMPUTE_GPU_CUH_

#include <cuda_runtime.h>
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/ParticleData.cuh"

namespace azplugins
{
namespace gpu
{
//! Kernel driver to compute position restraints on the GPU
cudaError_t compute_position_restraint(Scalar4 *d_force,
                                       const unsigned int *d_member_idx,
                                       const Scalar4 *d_pos,
                                       const Scalar4 *d_ref_pos,
                                       const unsigned int *d_tag,
                                       const Scalar3& k,
                                       const BoxDim& box,
                                       const unsigned int N,
                                       const unsigned int N_mem,
                                       const unsigned int block_size,
                                       const unsigned int compute_capability);
} // end namespace gpu
} // end namespace azplugins

#endif // AZPLUGINS_POSITION_RESTRAINT_COMPUTE_GPU_CUH_
