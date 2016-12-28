// Copyright (c) 2016, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file TypeUpdaterGPU.cuh
 * \brief Declaration of kernel drivers for TypeUpdaterGPU
 */

#ifndef AZPLUGINS_TYPE_UPDATER_GPU_CUH_
#define AZPLUGINS_TYPE_UPDATER_GPU_CUH_

#include <cuda_runtime.h>
#include "hoomd/HOOMDMath.h"

namespace azplugins
{
namespace gpu
{
//! Kernel driver to change particle types in a region on the GPU
cudaError_t change_types_region(Scalar4 *d_pos,
                                unsigned int inside_type,
                                unsigned int outside_type,
                                Scalar z_lo,
                                Scalar z_hi,
                                unsigned int N,
                                unsigned int block_size);
}
}

#endif // AZPLUGINS_TYPE_UPDATER_GPU_CUH_
