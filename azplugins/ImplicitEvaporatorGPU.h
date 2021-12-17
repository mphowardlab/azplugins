// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

/*!
 * \file ImplicitEvaporatorGPU.h
 * \brief Declaration of ImplicitEvaporatorGPU
 */

#ifndef AZPLUGINS_IMPLICIT_EVAPORATOR_GPU_H_
#define AZPLUGINS_IMPLICIT_EVAPORATOR_GPU_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "ImplicitEvaporator.h"

#include "hoomd/Autotuner.h"

namespace azplugins
{

//! Implicit solvent evaporator on the GPU
/*!
 * This class does not implement any force evaluation on its own, as the geometry should be
 * implemented by deriving classes. It exists as a thin layer between ImplicitEvaporator
 * to remove some boilerplate of setting up the autotuners.
 */
class PYBIND11_EXPORT ImplicitEvaporatorGPU : public ImplicitEvaporator
    {
    public:
        //! Constructor
        ImplicitEvaporatorGPU(std::shared_ptr<SystemDefinition> sysdef,
                              std::shared_ptr<Variant> interf);

        //! Set autotuner parameters
        /*!
         * \param enable Enable/disable autotuning
         * \param period period (approximate) in time steps when returning occurs
         */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            ImplicitEvaporator::setAutotunerParams(enable, period);
            m_tuner->setPeriod(period);
            m_tuner->setEnabled(enable);
            }

    protected:
        std::unique_ptr<Autotuner> m_tuner;   //!< Autotuner for block size
    };

namespace detail
{
//! Exports the ImplicitEvaporatorGPU to python
void export_ImplicitEvaporatorGPU(pybind11::module& m);
} // end namespace detail

} // end namespace azplugins

#endif // AZPLUGINS_IMPLICIT_EVAPORATOR_GPU_H_
