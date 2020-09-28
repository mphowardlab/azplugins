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
                            Scalar shear_rate,
                            unsigned int seed,
                            bool use_lambda,
                            Scalar lambda,
                            bool noiseless,
                            const std::string& suffix);

        //! Destructor
        virtual ~TwoStepSLLODLangevinFlow();

        //! Performs the second step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

        bool deformGlobalBox();
        
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

        //! Turn on or off Tally
        /*! \param tally if true, tallies energy exchange from the thermal reservoir */
        void setTally(bool tally)
            {
            m_tally= tally;
            }

        //! Returns a list of log quantities this integrator calculates
        virtual std::vector< std::string > getProvidedLogQuantities();

        //! Returns logged values
        Scalar getLogValue(const std::string& quantity, unsigned int timestep, bool &my_quantity_flag);

    protected:
        Scalar m_shear_rate;
        Scalar m_reservoir_energy;         //!< The energy of the reservoir the system is coupled to.
        Scalar m_extra_energy_overdeltaT;  //!< An energy packet that isn't added until the next time step
        bool m_tally;                      //!< If true, changes to the energy of the reservoir are calculated
        std::string m_log_name;            //!< Name of the reservoir quantity that we log
        bool m_noiseless;                  //!< If set true, there will be no random noise
    };


namespace detail
{
//! Export TwoStepSLLODLangevinFlow to python
void export_TwoStepSLLODLangevinFlow(pybind11::module& m);

} // end namespace detail
} // end namespace azplugins

#endif // AZPLUGINS_TWO_STEP_LANGEVIN_FLOW_H_
