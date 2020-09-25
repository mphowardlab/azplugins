// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt

/*!
 * \file TwoStepSLLODNVTFlow.h
 * \brief Declaration of SLLOD equation of motion with NVT Nosé-Hoover thermostat
 */

#ifndef AZPLUGINS_SLLOD_NVT_FLOW_H_
#define AZPLUGINS_SLLOD_NVT_FLOW_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include "hoomd/md/IntegrationMethodTwoStep.h"
#include "hoomd/Variant.h"
#include "ComputeThermoSLLOD.h"
#include "RNGIdentifiers.h"

namespace azplugins
{
//! Integrates part of the system forward in two steps in the NVT ensemble
/*! Implements Martyna-Tobias-Klein (MTK) NVT integration through the IntegrationMethodTwoStep interface

    Integrator variables mapping:
     - [0] -> xi
     - [1] -> eta

    The instantaneous temperature of the system is computed with the provided ComputeThermo. Correct dynamics require
    that the thermo computes the temperature of the assigned group and with D*N-D degrees of freedom. TwoStepSLLODNVTFlow does
    not check for these conditions.

    For the update equations of motion, see Refs. \cite{Martyna1994,Martyna1996}

    \ingroup updaters
*/
class PYBIND11_EXPORT TwoStepSLLODNVTFlow : public IntegrationMethodTwoStep
    {
    public:
        //! Constructs the integration method and associates it with the system
        TwoStepSLLODNVTFlow(std::shared_ptr<SystemDefinition> sysdef,
                   std::shared_ptr<ParticleGroup> group,
                   std::shared_ptr<ComputeThermoSLLOD> thermo,
                   Scalar tau,
                   std::shared_ptr<Variant> T,
                   Scalar shear_rate,
                   const std::string& suffix = std::string(""));
        virtual ~TwoStepSLLODNVTFlow();

        //! Update the temperature
        /*! \param T New temperature to set
        */
        virtual void setT(std::shared_ptr<Variant> T)
            {
            m_T = T;
            }

        //! Update the shear rate and velocity at boundary
        /*! \param shear_rate New shear rate to set
        */
        virtual void setShearRate(Scalar shear_rate)
            {
            m_shear_rate = shear_rate;
            BoxDim global_box = m_pdata->getGlobalBox();
            const Scalar3 global_hi = global_box.getHi();
            m_boundary_shear_velocity = global_hi.y * m_shear_rate*2;

            }

        //! Update the tau value
        /*! \param tau New time constant to set
        */
        virtual void setTau(Scalar tau)
            {
            m_tau = tau;
            }

        //! Set the value of xi (for unit tests)
        void setXi(Scalar new_xi)
            {
            IntegratorVariables v = getIntegratorVariables();
            Scalar& xi = v.variable[0];
            xi = new_xi;
            setIntegratorVariables(v);
            }

        //! Returns a list of log quantities this integrator calculates
        virtual std::vector< std::string > getProvidedLogQuantities();

        //! Returns logged values
        Scalar getLogValue(const std::string& quantity, unsigned int timestep, bool &my_quantity_flag);

        //! Performs the first step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

        //! Get needed pdata flags
        /*! in anisotropic mode, we need the rotational kinetic energy
        */
        virtual PDataFlags getRequestedPDataFlags()
            {
            PDataFlags flags;
            return flags;
            }

        //! Initialize integrator variables
        virtual void initializeIntegratorVariables()
            {
            IntegratorVariables v = getIntegratorVariables();
            v.type = "sllod_nvt";
            v.variable.clear();
            v.variable.resize(4);
            v.variable[0] = Scalar(0.0);
            v.variable[1] = Scalar(0.0);
            v.variable[2] = Scalar(0.0);
            v.variable[3] = Scalar(0.0);
            setIntegratorVariables(v);
            }

        //! Randomize the thermostat variable
        virtual void randomizeVelocities(unsigned int timestep);

    protected:
        std::shared_ptr<ComputeThermoSLLOD> m_thermo;    //!< compute for thermodynamic quantities

        Scalar m_tau;                     //!< tau value for Nose-Hoover
        std::shared_ptr<Variant> m_T;     //!< Temperature set point
        std::string m_log_name;           //!< Name of the reservoir quantity that we log

        Scalar m_exp_thermo_fac;          //!< Thermostat rescaling factor

        Scalar m_shear_rate;              //!< shear rate for SLLOD box deformation
        Scalar m_boundary_shear_velocity; //!< value of the velocity at the box boundary = Ly*shear_rate

        //! advance the thermostat
        /*!\param timestep The time step
         * \param broadcast True if we should broadcast the integrator variables via MPI
         */
        void advanceThermostat(unsigned int timestep, bool broadcast=true);

        bool deformGlobalBox();
    };

namespace detail
{
//! Exports the TwoStepSLLODNVTFlow class to python
void export_TwoStepSLLODNVTFlow(pybind11::module& m);

} // end namespace detail
} // end namespace azplugins

#endif // #ifndef AZPLUGINS_SLLOD_NVT_FLOW_H_
