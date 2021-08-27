// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

/*!
 * \file OrientationRestraintComputeGPU.cuh
 * \brief Declares CUDA kernels for OrientationRestraintComputeGPU
 */

#ifndef AZPLUGINS_ORIENTATION_RESTRAINT_COMPUTE_GPU_CUH_
#define AZPLUGINS_ORIENTATION_RESTRAINT_COMPUTE_GPU_CUH_

#include <cuda_runtime.h>
#include "hoomd/md/QuaternionMath.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/ParticleData.cuh"

namespace azplugins
{
namespace gpu
{

//! Kernel driver to compute orientation restraints on the GPU
cudaError_t compute_orientation_restraint(Scalar4 *d_force,
                                          Scalar4 *d_torque,
                                          const unsigned int *d_member_idx,
                                          const Scalar4 *d_orient,
                                          const Scalar4 *d_ref_orient,
                                          const unsigned int *d_tag,
                                          const Scalar& k,
                                          const BoxDim& box,
                                          const unsigned int N,
                                          const unsigned int N_mem,
                                          const unsigned int block_size,
                                          const unsigned int compute_capability);
} // end namespace gpu
} // end namespace azplugins

#endif // AZPLUGINS_ORIENTATION_RESTRAINT_COMPUTE_GPU_CUH_
