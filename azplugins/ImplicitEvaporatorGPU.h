// Copyright (c) 2016-2018, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

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
 * \warning The virial is not computed for this external potential, and a warning
 *          will be raised the first time it is requested.
 */
class ImplicitEvaporatorGPU : public ImplicitEvaporator
    {
    public:
        //! Constructor
        ImplicitEvaporatorGPU(std::shared_ptr<SystemDefinition> sysdef,
                              std::shared_ptr<Variant> interf);

        //! Destructor
        virtual ~ImplicitEvaporatorGPU() {};

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
        //! Implements the force calculation
        virtual void computeForces(unsigned int timestep);

    private:
        std::unique_ptr<Autotuner> m_tuner;   //!< Autotuner for block size
    };

namespace detail
{
//! Exports the ImplicitEvaporatorGPU to python
void export_ImplicitEvaporatorGPU(pybind11::module& m);
} // end namespace detail

} // end namespace azplugins

#endif // AZPLUGINS_IMPLICIT_EVAPORATOR_GPU_H_
