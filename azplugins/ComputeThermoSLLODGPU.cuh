// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#ifndef _AZPLUGINS_COMPUTE_THERMO_SLLOD_GPU_CUH_
#define _AZPLUGINS_COMPUTE_THERMO_SLLOD_GPU_CUH_

#include <cuda_runtime.h>

#include "hoomd/ParticleData.cuh"
#include "hoomd/ComputeThermoTypes.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/GPUPartition.cuh"
#include "hoomd/ComputeThermoGPU.cuh"

/*! \file ComputeThermoGPU.cuh
    \brief Kernel driver function declarations for ComputeThermoGPU
    */

namespace azplugins
{
namespace gpu
{

//! Computes the partial sums of thermodynamic properties for ComputeThermo
cudaError_t compute_thermo_partial(Scalar *d_properties,
                               Scalar4 *d_vel,
                               unsigned int *d_body,
                               unsigned int *d_tag,
                               unsigned int *d_group_members,
                               unsigned int group_size,
                               const BoxDim& box,
                               const compute_thermo_args& args,
                               bool compute_pressure_tensor,
                               bool compute_rotational_energy,
                               const GPUPartition& gpu_partition
                               );

//! Computes the final sums of thermodynamic properties for ComputeThermo
cudaError_t compute_thermo_final(Scalar *d_properties,
                               Scalar4 *d_vel,
                               unsigned int *d_body,
                               unsigned int *d_tag,
                               unsigned int *d_group_members,
                               unsigned int group_size,
                               const BoxDim& box,
                               const compute_thermo_args& args,
                               bool compute_pressure_tensor,
                               bool compute_rotational_energy
                               );

cudaError_t add_flow_field(Scalar4 *d_vel,
                           Scalar4 *d_pos,
                           unsigned int *d_group_members,
                           unsigned int group_size,
                           unsigned int block_size,
                           Scalar shear_rate,
                           const GPUPartition& gpu_partition);

cudaError_t remove_flow_field(Scalar4 *d_vel,
                           Scalar4 *d_pos,
                           unsigned int *d_group_members,
                           unsigned int group_size,
                           unsigned int block_size,
                           Scalar shear_rate,
                           const GPUPartition& gpu_partition);

} // end namespace gpu
} // end namespace azplugins

#endif
