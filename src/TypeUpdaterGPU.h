// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

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

//! Particle type updater on the GPU
/*!
 * See TypeUpdater for details. This class inherits and minimally implements
 * the CPU methods from TypeUpdater on the GPU.
 */
class PYBIND11_EXPORT TypeUpdaterGPU : public TypeUpdater
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
    virtual ~TypeUpdaterGPU() { };

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

    private:
    std::unique_ptr<Autotuner> m_tuner; //!< Tuner for changing types
    };

namespace detail
    {
//! Export TypeUpdaterGPU to python
void export_TypeUpdaterGPU(pybind11::module& m);
    } // end namespace detail

    } // end namespace azplugins

#endif // AZPLUGINS_TYPE_UPDATER_GPU_H_
