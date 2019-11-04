// Copyright (c) 2018-2019, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

#ifndef AZPLUGINS_PLANE_RESTRAINT_COMPUTE_GPU_CUH_
#define AZPLUGINS_PLANE_RESTRAINT_COMPUTE_GPU_CUH_

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"

#include <cuda_runtime.h>

namespace azplugins
{
namespace gpu
{
//! Kernel driver to compute plane restraints on the GPU
cudaError_t compute_plane_restraint(Scalar4* forces,
                                    Scalar* virials,
                                    const unsigned int* group,
                                    const Scalar4* positions,
                                    const int3* images,
                                    const BoxDim box,
                                    Scalar3 o,
                                    Scalar3 n,
                                    Scalar k,
                                    unsigned int N,
                                    unsigned int virial_pitch,
                                    unsigned int block_size);
} // end namespace gpu
} // end namespace azplugins

#endif // AZPLUGINS_PLANE_RESTRAINT_COMPUTE_GPU_CUH_
