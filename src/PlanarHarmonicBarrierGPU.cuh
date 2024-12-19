// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file PlanarHarmonicBarrierGPU.cuh
 * \brief Declaration of kernel drivers for PlanarHarmonicBarrierGPU
 */

#ifndef AZPLUGINS_PLANAR_HARMONIC_BARRIER_GPU_CUH_
#define AZPLUGINS_PLANAR_HARMONIC_BARRIER_GPU_CUH_

#include "hoomd/HOOMDMath.h"
#include <cuda_runtime.h>

namespace hoomd
    {
namespace azplugins
    {
namespace gpu
    {

//! Kernel driver to evaluate PlanarHarmonicBarrierGPU force
cudaError_t compute_implicit_evap_force(Scalar4* d_force,
                                        Scalar* d_virial,
                                        const Scalar4* d_pos,
                                        const Scalar4* d_params,
                                        const Scalar interf_origin,
                                        const unsigned int N,
                                        const unsigned int ntypes,
                                        const unsigned int block_size);

    } // end namespace gpu
    } // end namespace azplugins
    } // end namespace hoomd

#endif // AZPLUGINS_PLANAR_HARMONIC_BARRIER_GPU_CUH_
