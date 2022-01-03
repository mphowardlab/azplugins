// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt


/*!
 * \file SinusoidalChannelFillerGPU.h
 * \brief Definition of virtual particle filler for azplugins::detail::SinusoidalChannel on the GPU.
 */

#ifndef AZPLUGINS_SINE_CHANNEL_GEOMETRY_FILLER_GPU_H_
#define AZPLUGINS_SINE_CHANNEL_GEOMETRY_FILLER_GPU_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "SinusoidalChannelFiller.h"
#include "hoomd/Autotuner.h"
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"


namespace azplugins
{

//! Adds virtual particles to the MPCD particle data for SinusoidalExpansionConstriction using the GPU
class PYBIND11_EXPORT SinusoidalChannelFillerGPU : public SinusoidalChannelFiller
    {
    public:
        //! Constructor
        SinusoidalChannelFillerGPU(std::shared_ptr<mpcd::SystemData> sysdata,
                                   Scalar density,
                                   unsigned int type,
                                   std::shared_ptr<::Variant> T,
                                   unsigned int seed,
                                   std::shared_ptr<const detail::SinusoidalChannel> geom);

        //! Set autotuner parameters
        /*!
         * \param enable Enable/disable autotuning
         * \param period period (approximate) in time steps when returning occurs
         */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            SinusoidalChannelFiller::setAutotunerParams(enable, period);

            m_tuner->setEnabled(enable); m_tuner->setPeriod(period);
            }

    protected:
        //! Draw particles within the fill volume on the GPU
        virtual void drawParticles(unsigned int timestep);

    private:
        std::unique_ptr<::Autotuner> m_tuner;   //!< Autotuner for drawing particles
    };

namespace detail
{
//! Export SinusoidalChannelFillerGPU to python
void export_SinusoidalChannelFillerGPU(pybind11::module& m);
} // end namespace detail
} // end namespace azplugins
#endif // AZPLUGINS_SINE_CHANNEL_GEOMETRY_FILLER_GPU_H_
