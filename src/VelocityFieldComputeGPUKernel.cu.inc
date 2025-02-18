// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

// See CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off
#include "ParticleDataLoader.h"
#include "VelocityFieldComputeGPU.cuh"
#include "@_geometry@BinningOperation.h"

#define BINNING_OPERATION @_geometry@BinningOperation
// clang-format on

namespace hoomd
    {
namespace azplugins
    {
namespace gpu
    {

template __attribute__((visibility("default"))) cudaError_t
bin_velocity_field<detail::LoadParticleGroupPositionVelocityMass, BINNING_OPERATION>(
    Scalar* d_mass,
    Scalar3* d_momentum,
    const detail::LoadParticleGroupPositionVelocityMass& load_op,
    const BINNING_OPERATION& bin_op,
    const BoxDim& global_box,
    const unsigned int N,
    const unsigned int block_size);

template __attribute__((visibility("default"))) cudaError_t
bin_velocity_field<detail::LoadMPCDParticlePositionVelocityMass, BINNING_OPERATION>(
    Scalar* d_mass,
    Scalar3* d_momentum,
    const detail::LoadMPCDParticlePositionVelocityMass& load_op,
    const BINNING_OPERATION& bin_op,
    const BoxDim& global_box,
    const unsigned int N,
    const unsigned int block_size);

    } // end namespace gpu
    } // namespace azplugins
    } // end namespace hoomd
