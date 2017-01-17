// Copyright (c) 2016, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file ParticleEvaporatorGPU.cuh
 * \brief Declaration of kernel drivers for ParticleEvaporatorGPU
 */

#ifndef AZPLUGINS_IMPLICIT_EVAPORATOR_GPU_CUH_
#define AZPLUGINS_IMPLICIT_EVAPORATOR_GPU_CUH_

#include <cuda_runtime.h>
#include "hoomd/HOOMDMath.h"

namespace azplugins
{
namespace gpu
{

//! Kernel driver to build unsorted mpcd cell list
cudaError_t compute_implicit_evap_force(Scalar4 *d_force,
                                        Scalar *d_virial,
                                        const Scalar4 *d_pos,
                                        const Scalar4 *d_params,
                                        const Scalar interf_origin,
                                        const unsigned int N,
                                        const unsigned int ntypes,
                                        const unsigned int block_size,
                                        const unsigned int compute_capability);
} // end namespace gpu
} // end namespace azplugins

#endif // AZPLUGINS_IMPLICIT_EVAPORATOR_GPU_CUH_
