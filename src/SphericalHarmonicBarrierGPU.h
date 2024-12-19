// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file SphericalHarmonicBarrierGPU.h
 * \brief Declaration of SphericalHarmonicBarrierGPU
 */

#ifndef AZPLUGINS_SPHERICAL_HARMONIC_BARRIER_GPU_H_
#define AZPLUGINS_SPHERICAL_HARMONIC_BARRIER_GPU_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "HarmonicBarrierGPU.h"

namespace hoomd
    {

namespace azplugins
    {

//! Moving harmonic potential in a spherical (droplet) geometry (on the GPU)
class PYBIND11_EXPORT SphericalHarmonicBarrierGPU : public HarmonicBarrierGPU
    {
    public:
    //! Constructor
    SphericalHarmonicBarrierGPU(std::shared_ptr<SystemDefinition> sysdef,
                                std::shared_ptr<Variant> interf);

    //! Destructor
    virtual ~SphericalHarmonicBarrierGPU();

    protected:
    //! Implements the force calculation
    virtual void computeForces(uint64_t timestep);
    };

    } // end namespace azplugins

    } // end namespace hoomd

#endif // AZPLUGINS_SPHERICAL_HARMONIC_BARRIER_GPU_H_
