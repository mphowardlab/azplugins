// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_VELOCITY_COMPUTE_GPU_CUH_
#define AZPLUGINS_VELOCITY_COMPUTE_GPU_CUH_

#include <cuda_runtime.h>

#include "ParticleDataLoader.h"

#include "hoomd/HOOMDMath.h"

namespace hoomd
    {
namespace azplugins
    {
namespace gpu
    {

//! Specialization to HOOMD group
void sum_momentum_and_mass(Scalar3& momentum,
                           Scalar& mass,
                           const detail::LoadHOOMDGroupVelocityMass& load_op,
                           unsigned int N);

//! Specialization to MPCD group
void sum_momentum_and_mass(Scalar3& momentum,
                           Scalar& mass,
                           const detail::LoadMPCDVelocityMass& load_op,
                           unsigned int N);

    } // end namespace gpu
    } // namespace azplugins
    } // end namespace hoomd

#endif // AZPLUGINS_VELOCITY_COMPUTE_GPU_CUH_
