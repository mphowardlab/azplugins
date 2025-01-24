// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_VELOCITY_COMPUTE_GPU_H_
#define AZPLUGINS_VELOCITY_COMPUTE_GPU_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "VelocityCompute.h"

namespace hoomd
    {
namespace azplugins
    {
//! Compute the center-of-mass velocity of a group of particles on the GPU
class PYBIND11_EXPORT VelocityComputeGPU : public VelocityCompute
    {
    public:
    using VelocityCompute::VelocityCompute;

    protected:
    void sumMomentumAndMass(Scalar3& momentum, Scalar& mass) override;
    };
    } // end namespace azplugins
    } // end namespace hoomd

#endif // AZPLUGINS_VELOCITY_COMPUTE_GPU_H_
