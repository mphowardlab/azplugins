// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file SphericalMovingHarmonicPotentialGPU.h
 * \brief Declaration of SphericalMovingHarmonicPotentialGPU
 */

#ifndef AZPLUGINS_SPHERICAL_MOVING_HARMONIC_POTENTIAL_GPU_H_
#define AZPLUGINS_SPHERICAL_MOVING_HARMONIC_POTENTIAL_GPU_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "MovingHarmonicPotentialGPU.h"

namespace hoomd
    {

namespace azplugins
    {

//! Moving harmonic potential in a spherical (droplet) geometry (on the GPU)
class PYBIND11_EXPORT SphericalMovingHarmonicPotentialGPU : public MovingHarmonicPotentialGPU
    {
    public:
    //! Constructor
    SphericalMovingHarmonicPotentialGPU(std::shared_ptr<SystemDefinition> sysdef,
                                        std::shared_ptr<Variant> interf);

    //! Destructor
    virtual ~SphericalMovingHarmonicPotentialGPU();

    protected:
    //! Implements the force calculation
    virtual void computeForces(unsigned int timestep);
    };

    } // end namespace azplugins

    } // end namespace hoomd

#endif // AZPLUGINS_SPHERICAL_MOVING_HARMONIC_POTENTIAL_GPU_H_
