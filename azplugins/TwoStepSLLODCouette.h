// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: arjunsg2

/*!
 * \file TwoStepSLLODCouette.h
 * \brief Declaration of TwoStepSLLODCouette
 */

#ifndef AZPLUGINS_TWO_STEP_SLLOD_COUETTE_H_
#define AZPLUGINS_TWO_STEP_SLLOD_COUETTE_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "hoomd/md/IntegrationMethodTwoStep.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"
#include "RNGIdentifiers.h"
#include "hoomd/Variant.h"
#include <vector>

namespace azplugins
{

/*! Integrates part of the system forward in two steps with SLLOD equations
    of motion under homogenous linear shear (Couette) flow
 */

class PYBIND11_EXPORT TwoStepSLLODCouette : public IntegrationMethodTwoStep
    {
    public:
        //! Constructor
        TwoStepSLLODCouette(std::shared_ptr<SystemDefinition> sysdef,
                            std::shared_ptr<ParticleGroup> group,
                            std::shared_ptr<Variant> T,
                            unsigned int seed,
                            bool use_lambda,
                            Scalar lambda,
                            Scalar gamma_dot,
                            bool noiseless);

        //! Destructor
        virtual ~TwoStepSLLODCouette()
            {
            m_exec_conf->msg->notice(5) << "Destroying TwoStepSLLODCouette" << std::endl;
            }

        void setT(std::shared_ptr<Variant> T)
            {
            m_T = T;
            }

        void setGamma(unsigned int typ, Scalar gamma)
            {
            // check for user errors
            if (m_use_lambda)
                {
                m_exec_conf->msg->error() << "Trying to set gamma when it is set to be the diameter! " << typ << std::endl;
                throw std::runtime_error("Error setting params in TwoStepSLLODCouette");
                }
            if (typ >= m_pdata->getNTypes())
                {
                m_exec_conf->msg->error() << "Trying to set gamma for a non existent type! " << typ << std::endl;
                throw std::runtime_error("Error setting params in TwoStepSLLODCouette");
                }

            ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::readwrite);
            h_gamma.data[typ] = gamma;
            }

        void set_gamma_dot(Scalar gamma_dot)
            {
            m_gamma_dot = gamma_dot;
            }

        void setNoiseless(bool noiseless)
            {
            m_noiseless = noiseless;
            }

        //! Performs the second step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

    protected:
        std::shared_ptr<Variant> m_T;
        unsigned int m_seed;
        bool m_use_lambda;
        Scalar m_lambda;
        Scalar m_gamma_dot;
        bool m_noiseless;

        GlobalVector<Scalar> m_gamma;
    };


namespace detail
{

//! Export TwoStepSLLODCouette to python
void export_TwoStepSLLODCouette(pybind11::module& m);

} // end namespace detail
} // end namespace azplugins

#endif // AZPLUGINS_TWO_STEP_SLLOD_COUETTE_H_
