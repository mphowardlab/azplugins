// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

/*!
 * \file ImplicitDropletEvaporatorGPU.cuh
 * \brief Declaration of kernel drivers for ImplicitDropletEvaporatorGPU
 */

#ifndef AZPLUGINS_IMPLICIT_DROPLET_EVAPORATOR_GPU_CUH_
#define AZPLUGINS_IMPLICIT_DROPLET_EVAPORATOR_GPU_CUH_

#include <cuda_runtime.h>
#include "hoomd/HOOMDMath.h"

namespace azplugins
{
namespace gpu
{

//! Kernel driver to evaluate ImplicitDropletEvaporatorGPU force
cudaError_t compute_implicit_evap_droplet_force(Scalar4 *d_force,
                                                Scalar *d_virial,
                                                const Scalar4 *d_pos,
                                                const Scalar4 *d_params,
                                                const Scalar interf_origin,
                                                const unsigned int N,
                                                const unsigned int ntypes,
                                                const unsigned int block_size);

} // end namespace gpu
} // end namespace azplugins

#endif // AZPLUGINS_IMPLICIT_DROPLET_EVAPORATOR_GPU_CUH_
