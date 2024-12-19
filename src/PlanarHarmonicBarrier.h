// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file PlanarHarmonicBarrier.h
 * \brief Declaration of PlanarHarmonicBarrier
 */

#ifndef AZPLUGINS_PLANAR_HARMONIC_BARRIER_H_
#define AZPLUGINS_PLANAR_HARMONIC_BARRIER_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "HarmonicBarrier.h"

#include <pybind11/pybind11.h>

namespace hoomd
    {

namespace azplugins
    {

//! Moving Harmonic Potential in a planar (thin film) geometry
/*!
 * The interface normal is defined along +z going from the liquid into the vapor phase,
 * and the origin is z = 0. This effectively models a drying thin film.
 */
class PYBIND11_EXPORT PlanarHarmonicBarrier : public HarmonicBarrier
    {
    public:
    //! Constructor
    PlanarHarmonicBarrier(std::shared_ptr<SystemDefinition> sysdef,
                          std::shared_ptr<Variant> interf);

    virtual ~PlanarHarmonicBarrier();

    protected:
    //! Implements the force calculation
    virtual void computeForces(uint64_t timestep);
    };

    } // end namespace azplugins

    } // end namespace hoomd

#endif // AZPLUGINS_PLANAR_HARMONIC_BARRIER_H_
