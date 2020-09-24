// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt

/*!
 * \file TwoStepSLLODNVTFlowGPU.h
 * \brief Declaration of SLLOD equation of motion with NVT Nosé-Hoover thermostat
 */

#ifndef AZPLUGINS_SLLOD_NVT_FLOW_GPU_H_
#define AZPLUGINS_SLLOD_NVT_FLOW_GPU_H_


#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "TwoStepSLLODNVTFlow.h"
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include "hoomd/Autotuner.h"

namespace azplugins
{
//! Integrates part of the system forward in two steps in the NVT ensemble on the GPU
/*! Implements Nose-Hoover NVT integration through the IntegrationMethodTwoStep interface, runs on the GPU

    In order to compute efficiently and limit the number of kernel launches integrateStepOne() performs a first
    pass reduction on the sum of m*v^2 and stores the partial reductions. A second kernel is then launched to reduce
    those to a final \a sum2K, which is a scalar but stored in a GPUArray for convenience.

    \ingroup updaters
*/
class PYBIND11_EXPORT TwoStepSLLODNVTFlowGPU : public TwoStepSLLODNVTFlow
    {
    public:
        //! Constructs the integration method and associates it with the system
        TwoStepSLLODNVTFlowGPU(std::shared_ptr<SystemDefinition> sysdef,
                      std::shared_ptr<ParticleGroup> group,
                      std::shared_ptr<ComputeThermo> thermo,
                      Scalar tau,
                      std::shared_ptr<Variant> T,
                      Scalar shear_rate,
                      const std::string& suffix = std::string(""));
        virtual ~TwoStepSLLODNVTFlowGPU() {};

        //! Performs the first step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            TwoStepSLLODNVTFlow::setAutotunerParams(enable, period);
            m_tuner_one->setPeriod(period);
            m_tuner_one->setEnabled(enable);
            m_tuner_two->setPeriod(period);
            m_tuner_two->setEnabled(enable);
            m_tuner_rm_flowfield->setPeriod(period);
            m_tuner_rm_flowfield->setEnabled(enable);
            m_tuner_add_flowfield->setPeriod(period);
            m_tuner_add_flowfield->setEnabled(enable);
            }

    protected:
        std::unique_ptr<Autotuner> m_tuner_one; //!< Autotuner for block size (step one kernel)
        std::unique_ptr<Autotuner> m_tuner_two; //!< Autotuner for block size (step two kernel)

        std::unique_ptr<Autotuner> m_tuner_rm_flowfield; //!< Autotuner for removing flow field from velocities
        std::unique_ptr<Autotuner> m_tuner_add_flowfield; //!< Autotuner for adding flow field to velocities

        virtual void addFlowField();
        virtual void removeFlowField();
    };

namespace detail
{
//! Exports the TwoStepSLLODNVTFlowGPU class to python
void export_TwoStepSLLODNVTFlowGPU(pybind11::module& m);

} // end namespace detail
} // end namespace azplugins

#endif // #ifndef AZPLUGINS_SLLOD_NVT_FLOW_GPU_H_
