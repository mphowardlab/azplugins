// Copyright (c) 2016, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file TypeUpdaterGPU.h
 * \brief Declaration of TypeUpdaterGPU
 */

#ifndef AZPLUGINS_TYPE_UPDATER_GPU_H_
#define AZPLUGINS_TYPE_UPDATER_GPU_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "TypeUpdater.h"
#include "hoomd/Autotuner.h"

namespace azplugins
{

class TypeUpdaterGPU : public TypeUpdater
    {
    public:
        //! Simple constructor
        TypeUpdaterGPU(std::shared_ptr<SystemDefinition> sysdef);

        //! Constructor with parameters
        TypeUpdaterGPU(std::shared_ptr<SystemDefinition> sysdef,
                       unsigned int inside_type,
                       unsigned int outside_type,
                       Scalar z_lo,
                       Scalar z_hi);

        //! Destructor
        virtual ~TypeUpdaterGPU() {};

        //! Set autotuner parameters
        /*!
         * \param enable Enable / disable autotuning
         * \param period period (approximate) in time steps when retuning occurs
         */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            TypeUpdater::setAutotunerParams(enable, period);
            m_tuner->setPeriod(period);
            m_tuner->setEnabled(enable);
            }

    protected:
        //! Changes the particle types according to an update rule on the GPU
        virtual void changeTypes(unsigned int timestep);

        std::unique_ptr<Autotuner> m_tuner; //!< Tuner for changing types
    };

namespace detail
{
void export_TypeUpdaterGPU(pybind11::module& m);
} // end namespace detail

} // end namespace azplugins

#endif // AZPLUGINS_TYPE_UPDATER_GPU_H_
