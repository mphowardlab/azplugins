// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file TwoStepSLLODLangevinFlow.h
 * \brief Declaration of TwoStepSLLODLangevinFlow
 */

#ifndef AZPLUGINS_TWO_STEP_SLLOD_LANGEVIN_H_
#define AZPLUGINS_TWO_STEP_SLLOD_LANGEVIN_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "hoomd/md/TwoStepLangevinBase.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

#include "RNGIdentifiers.h"

namespace azplugins
{

//! Integrates part of the system forward in two steps with Langevin dynamics under flow
/*!
 * \note Only translational motion is supported by this integrator.
 */

class PYBIND11_EXPORT TwoStepSLLODLangevinFlow : public TwoStepLangevinBase
    {
    public:
        //! Constructor
        TwoStepSLLODLangevinFlow(std::shared_ptr<SystemDefinition> sysdef,
                            std::shared_ptr<ParticleGroup> group,
                            std::shared_ptr<Variant> T,
                            unsigned int seed,
                            bool use_lambda,
                            Scalar lambda,
                            bool noiseless)
        : TwoStepLangevinBase(sysdef, group, T, seed, use_lambda, lambda),m_noiseless(noiseless)
            {
            m_exec_conf->msg->notice(5) << "Constructing TwoStepSLLODLangevinFlow" << std::endl;
            if (m_sysdef->getNDimensions() < 3)
                {
                m_exec_conf->msg->error() << "flow.sllod_langevin is only supported in 3D" << std::endl;
                throw std::runtime_error("Langevin dynamics in flow is only supported in 3D");
                }
            }

        //! Destructor
        virtual ~TwoStepSLLODLangevinFlow()
            {
            m_exec_conf->msg->notice(5) << "Destroying TwoStepSLLODLangevinFlow" << std::endl;
            }

        //! Performs the second step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);


        //! Get the flag for if noise is applied to the motion
        bool getNoiseless() const
            {
            return m_noiseless;
            }

        //! Set the flag to apply noise to the motion
        /*!
         * \param noiseless If true, do not apply a random diffusive force
         */
        void setNoiseless(bool noiseless)
            {
            m_noiseless = noiseless;
            }

    protected:
        bool m_noiseless;                           //!< If set true, there will be no random noise
    };


namespace detail
{
//! Export TwoStepSLLODLangevinFlow to python
void export_TwoStepSLLODLangevinFlow(pybind11::module& m);

} // end namespace detail
} // end namespace azplugins

#endif // AZPLUGINS_TWO_STEP_LANGEVIN_FLOW_H_
