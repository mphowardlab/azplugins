// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file TwoStepSLLODLangevinFlowGPU.h
 * \brief Declaration of TwoStepSLLODLangevinFlowGPU
 */

#ifndef AZPLUGINS_TWO_STEP_SLLOD_LANGEVIN_FLOW_GPU_H_
#define AZPLUGINS_TWO_STEP_SLLOD_LANGEVIN_FLOW_GPU_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "TwoStepSLLODLangevinFlow.h"
#include "TwoStepSLLODLangevinFlowGPU.cuh"
#include "hoomd/Autotuner.h"

namespace azplugins
{

class PYBIND11_EXPORT TwoStepSLLODLangevinFlowGPU : public TwoStepSLLODLangevinFlow
    {
    public:
        //! Constructor
        TwoStepSLLODLangevinFlowGPU(std::shared_ptr<SystemDefinition> sysdef,
                               std::shared_ptr<ParticleGroup> group,
                               std::shared_ptr<Variant> T,
                               Scalar shear_rate,
                               unsigned int seed,
                               bool use_lambda,
                               Scalar lambda,
                               bool noiseless,
                               const std::string& suffix);

        //! Destructor
        virtual ~TwoStepSLLODLangevinFlowGPU() {};

        //! Performs the second step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

        //! Set autotuner parameters
        /*!
         * \param enable Enable/disable autotuning
         * \param period period (approximate) in time steps when returning occurs
         */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            TwoStepSLLODLangevinFlow::setAutotunerParams(enable, period);
            this->m_tuner_1->setPeriod(period); this->m_tuner_1->setEnabled(enable);
            this->m_tuner_2->setPeriod(period); this->m_tuner_2->setEnabled(enable);
            }

    private:
        std::unique_ptr<Autotuner> m_tuner_1;   //!< Kernel tuner for step 1
        std::unique_ptr<Autotuner> m_tuner_2;   //!< Kernel tuner for step 2
    };

namespace detail
{
//! Export TwoStepSLLODLangevinFlowGPU to python
void export_TwoStepSLLODLangevinFlowGPU(pybind11::module& m);

} // end namespace detail
}
#endif // AZPLUGINS_TWO_STEP_LANGEVIN_FLOW_GPU_H_
