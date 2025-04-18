// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file TypeUpdaterGPU.cuh
 * \brief Declaration of kernel drivers for TypeUpdaterGPU
 */

#ifndef AZPLUGINS_TYPE_UPDATER_GPU_CUH_
#define AZPLUGINS_TYPE_UPDATER_GPU_CUH_

#include "hoomd/HOOMDMath.h"
#include <cuda_runtime.h>

namespace azplugins
    {
namespace gpu
    {
//! Kernel driver to change particle types in a region on the GPU
cudaError_t change_types_region(Scalar4* d_pos,
                                unsigned int inside_type,
                                unsigned int outside_type,
                                Scalar z_lo,
                                Scalar z_hi,
                                unsigned int N,
                                unsigned int block_size);
    } // namespace gpu
    } // namespace azplugins

#endif // AZPLUGINS_TYPE_UPDATER_GPU_CUH_
