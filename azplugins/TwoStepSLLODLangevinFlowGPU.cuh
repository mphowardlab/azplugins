// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file TwoStepSLLODLangevinFlowGPU.cuh
 * \brief Declaration of kernel drivers for TwoStepSLLODLangevinFlowGPU
 */

#ifndef AZPLUGINS_TWO_STEP_SLLOD_LANGEVIN_GPU_CUH_
#define AZPLUGINS_TWO_STEP_SLLOD_LANGEVIN_GPU_CUH_

#include <cuda_runtime.h>
#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/RandomNumbers.h"
#include "RNGIdentifiers.h"

namespace azplugins
{
namespace gpu
{

//! Step one of the langevin dynamics algorithm (NVE step)
cudaError_t langevin_sllod_step1(Scalar4 *d_pos,
                                int3 *d_image,
                                Scalar4 *d_vel,
                                const Scalar3 *d_accel,
                                const unsigned int *d_group,
                                const BoxDim& box,
                                const unsigned int N,
                                const Scalar dt,
                                const unsigned int block_size);

//! Step two of the langevin dynamics step (drag and velocity update)
cudaError_t langevin_sllod_step2(Scalar4 *d_vel,
                                Scalar3 *d_accel,
                                const Scalar4 *d_pos,
                                const Scalar4 *d_net_force,
                                const unsigned int *d_tag,
                                const unsigned int *d_group,
                                const Scalar *d_diameter,
                                const Scalar lambda,
                                const Scalar *d_gamma,
                                const unsigned int ntypes,
                                const unsigned int N,
                                const Scalar dt,
                                const Scalar T,
                                const unsigned int timestep,
                                const unsigned int seed,
                                bool noiseless,
                                bool use_lambda,
                                const unsigned int block_size);

} // end namespace gpu
} // end namespace azplugins

#endif // AZPLUGINS_TWO_STEP_SLLOD_LANGEVIN_FLOW_GPU_CUH_
