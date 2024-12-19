// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file SphericalHarmonicBarrier.h
 * \brief Declaration of SphericalHarmonicBarrier
 */

#ifndef AZPLUGINS_SPHERICAL_HARMONIC_BARRIER_H_
#define AZPLUGINS_SPHERICAL_HARMONIC_BARRIER_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "HarmonicBarrier.h"

#include <pybind11/pybind11.h>

namespace hoomd
    {

namespace azplugins
    {

//! Moving Harmonic Potential in a spherical (droplet) geometry
/*
 * The interface normal is that of a sphere, and its origin is (0,0,0).
 */
class PYBIND11_EXPORT SphericalHarmonicBarrier : public HarmonicBarrier
    {
    public:
    //! Constructor
    SphericalHarmonicBarrier(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<Variant> interf);

    virtual ~SphericalHarmonicBarrier();

    protected:
    //! Implements the force calculation
    virtual void computeForces(uint64_t timestep);
    };

    } // end namespace azplugins

    } // end namespace hoomd
#endif // AZPLUGINS_SPHERICAL_HARMONIC_BARRIER_H_
