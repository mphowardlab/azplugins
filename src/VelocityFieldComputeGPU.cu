// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#include "VelocityFieldComputeGPU.cuh"

namespace hoomd
    {
namespace azplugins
    {
namespace gpu
    {

cudaError_t zeroVelocityFieldArrays(Scalar* d_mass, Scalar3* d_momentum, size_t num_bins)
    {
    cudaError_t result = cudaMemset(d_mass, 0, sizeof(Scalar) * num_bins);
    if (result != cudaSuccess)
        {
        return result;
        }

    result = cudaMemset(d_momentum, 0, sizeof(Scalar3) * num_bins);
    if (result != cudaSuccess)
        {
        return result;
        }

    return cudaSuccess;
    }

    } // end namespace gpu
    } // end namespace azplugins
    } // end namespace hoomd
