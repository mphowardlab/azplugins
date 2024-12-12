// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file PlanarMovingHarmonicBarrierGPU.h
 * \brief Declaration of PlanarMovingHarmonicBarrierGPU
 */

#ifndef AZPLUGINS_PLANAR_MOVING_HARMONIC_BARRIER_GPU_H_
#define AZPLUGINS_PLANAR_MOVING_HARMONIC_BARRIER_GPU_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "MovingHarmonicPotentialGPU.h"

namespace hoomd
    {

namespace azplugins
    {

//! Moving Harmonic Potential in a planar (thin film) geometry (on the GPU)
class PYBIND11_EXPORT PlanarMovingHarmonicBarrierGPU : public MovingHarmonicPotentialGPU
    {
    public:
    //! Constructor
    PlanarMovingHarmonicBarrierGPU(std::shared_ptr<SystemDefinition> sysdef,
                                   std::shared_ptr<Variant> interf);

    //! Destructor
    virtual ~PlanarMovingHarmonicBarrierGPU();

    protected:
    //! Implements the force calculation
    virtual void computeForces(unsigned int timestep);
    };

    } // end namespace azplugins

    } // end namespace hoomd

#endif // AZPLUGINS_PLANAR_MOVING_HARMONIC_BARRIER_GPU_H_
