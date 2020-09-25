// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt

/*!
 * \file TwoStepSLLODNVTFlowGPU.cuh
 * \brief Declaration of SLLOD equation of motion with NVT Nosé-Hoover thermostat
 */

#ifndef AZPLUGINS_SLLOD_NVT_FLOW_GPU_CUH_
#define AZPLUGINS_SLLOD_NVT_FLOW_GPU_CUH_

#include <cuda_runtime.h>
#include "hoomd/ParticleData.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/GPUPartition.cuh"

namespace azplugins
{
namespace gpu
{

//! Kernel driver for the first part of the NVT update called by TwoStepNVTGPU
cudaError_t sllod_nvt_step_one(Scalar4 *d_pos,
                             Scalar4 *d_vel,
                             const Scalar3 *d_accel,
                             int3 *d_image,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             const BoxDim& box,
                             unsigned int block_size,
                             Scalar exp_fac,
                             Scalar deltaT,
                             Scalar shear_rate,
                             bool flipped,
                             Scalar m_boundary_shear_velocity,
                             const GPUPartition& gpu_partition
                             );

//! Kernel driver for the second part of the NVT update called by NVTUpdaterGPU
cudaError_t sllod_nvt_step_two(Scalar4 *d_vel,
                             Scalar3 *d_accel,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             Scalar4 *d_net_force,
                             unsigned int block_size,
                             Scalar deltaT,
                             Scalar shear_rate,
                             Scalar exp_v_fac_thermo,
                             const GPUPartition& gpu_partition);


} // end namespace gpu
} // end namespace azplugins

#endif //AZPLUGINS_SLLOD_NVT_FLOW_GPU_CUH_
