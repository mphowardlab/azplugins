// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file MovingHarmonicPotentialGPU.h
 * \brief Declaration of MovingHarmonicPotentialGPU
 */

#ifndef AZPLUGINS_MOVING_HARMONIC_POTENTIAL_GPU_H_
#define AZPLUGINS_MOVING_HARMONIC_POTENTIAL_GPU_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "MovingHarmonicPotential.h"

#include "hoomd/Autotuner.h"

namespace hoomd
    {

namespace azplugins
    {

//! Moving Harmonic potential on the GPU
/*!
 * This class does not implement any force evaluation on its own, as the geometry should be
 * implemented by deriving classes. It exists as a thin layer between MovingHarmonicPotential
 * to remove some boilerplate of setting up the autotuners.
 */
class PYBIND11_EXPORT MovingHarmonicPotentialGPU : public MovingHarmonicPotential
    {
    public:
    //! Constructor
    MovingHarmonicPotentialGPU(std::shared_ptr<SystemDefinition> sysdef,
                               std::shared_ptr<Variant> interf);

    //! Set autotuner parameters
    /*!
     * \param enable Enable/disable autotuning
     * \param period period (approximate) in time steps when returning occurs
     */
    virtual void setAutotunerParams(bool enable, unsigned int period)
        {
        MovingHarmonicPotential::setAutotunerParams(enable, period);
        m_tuner->setPeriod(period);
        m_tuner->setEnabled(enable);
        }

    protected:
    std::unique_ptr<Autotuner> m_tuner; //!< Autotuner for block size
    };

    } // end namespace azplugins

    } // end namespace hoomd

#endif // AZPLUGINS_MOVING_HARMONIC_POTENTIAL_GPU_H_
