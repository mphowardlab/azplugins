// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2022, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

/*!
 * \file TwoStepLangevinFlow.h
 * \brief Declaration of TwoStepLangevinFlow
 */

#ifndef AZPLUGINS_TWO_STEP_LANGEVIN_FLOW_H_
#define AZPLUGINS_TWO_STEP_LANGEVIN_FLOW_H_

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
template<class FlowField>
class PYBIND11_EXPORT TwoStepLangevinFlow : public TwoStepLangevinBase
    {
    public:
        //! Constructor
        TwoStepLangevinFlow(std::shared_ptr<SystemDefinition> sysdef,
                            std::shared_ptr<ParticleGroup> group,
                            std::shared_ptr<Variant> T,
                            std::shared_ptr<FlowField> flow_field,
                            unsigned int seed,
                            bool use_lambda,
                            Scalar lambda,
                            bool noiseless)
        : TwoStepLangevinBase(sysdef, group, T, seed, use_lambda, lambda), m_flow_field(flow_field), m_noiseless(noiseless)
            {
            m_exec_conf->msg->notice(5) << "Constructing TwoStepLangevinFlow" << std::endl;
            if (m_sysdef->getNDimensions() < 3)
                {
                m_exec_conf->msg->error() << "flow.langevin is only supported in 3D" << std::endl;
                throw std::runtime_error("Langevin dynamics in flow is only supported in 3D");
                }
            }

        //! Destructor
        virtual ~TwoStepLangevinFlow()
            {
            m_exec_conf->msg->notice(5) << "Destroying TwoStepLangevinFlow" << std::endl;
            }

        //! Performs the second step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

        //! Get the flow field
        std::shared_ptr<FlowField> getFlowField() const
            {
            return m_flow_field;
            }

        //! Set the flow field
        /*!
         * \param flow_field New flow field to apply
         */
        void setFlowField(std::shared_ptr<FlowField> flow_field)
            {
            m_flow_field = flow_field;
            }

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
        std::shared_ptr<FlowField> m_flow_field;    //!< Flow field functor
        bool m_noiseless;                           //!< If set true, there will be no random noise
    };

template<class FlowField>
void TwoStepLangevinFlow<FlowField>::integrateStepOne(unsigned int timestep)
    {
    if (m_aniso)
        {
        m_exec_conf->msg->error() << "azplugins.integrate: anisotropic particles are not supported with langevin flow integrators." << std::endl;
        throw std::runtime_error("Anisotropic integration not supported with langevin flow");
        }
    if (m_prof) m_prof->push("Langevin step 1");

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::read);

    const BoxDim& box = m_pdata->getBox();

    // perform the first half step of velocity verlet
    unsigned int group_size = m_group->getNumMembers();
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int idx = m_group->getMemberIndex(group_idx);

        const Scalar4 postype = h_pos.data[idx];
        Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
        unsigned int type = __scalar_as_int(postype.w);

        // velocity
        const Scalar4 velmass = h_vel.data[idx];
        Scalar3 vel = make_scalar3(velmass.x, velmass.y, velmass.z);
        Scalar mass = velmass.w;

        // acceleration
        const Scalar3 accel = h_accel.data[idx];

        // update position and wrap
        pos += (vel + Scalar(0.5) * m_deltaT * accel) * m_deltaT;
        box.wrap(pos,h_image.data[idx]);

        // update velocity
        vel += Scalar(0.5) * m_deltaT * accel;

        h_pos.data[idx] = make_scalar4(pos.x, pos.y, pos.z, __int_as_scalar(type));
        h_vel.data[idx] = make_scalar4(vel.x, vel.y, vel.z, mass);
        }

    if (m_prof) m_prof->pop();
    }

template<class FlowField>
void TwoStepLangevinFlow<FlowField>::integrateStepTwo(unsigned int timestep)
    {
    if (m_aniso)
        {
        m_exec_conf->msg->error() << "azplugins.integrate: anisotropic particles are not supported with langevin flow integrators." << std::endl;
        throw std::runtime_error("Anisotropic integration not supported with langevin flow");
        }
    if (m_prof) m_prof->push("Langevin step 2");

    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_net_force(m_pdata->getNetForce(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::read);

    const Scalar currentTemp = m_T->getValue(timestep);
    const FlowField& flow_field = *m_flow_field;

    // second step of velocity verlet while modifying accelerations
    unsigned int group_size = m_group->getNumMembers();
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int idx = m_group->getMemberIndex(group_idx);

        // get the friction coefficient
        const Scalar4 postype = h_pos.data[idx];
        Scalar gamma;
        if (m_use_lambda)
            gamma = m_lambda*h_diameter.data[idx];
        else
            {
            unsigned int type = __scalar_as_int(postype.w);
            gamma = h_gamma.data[type];
            }

        // get the flow velocity at the current position
        const Scalar3 flow_vel = flow_field(make_scalar3(postype.x, postype.y, postype.z));

        // compute the random force
        Scalar coeff = fast::sqrt(Scalar(6.0)*gamma*currentTemp/m_deltaT);
        if (m_noiseless)
            coeff = Scalar(0.0);
        hoomd::RandomGenerator rng(azplugins::RNGIdentifier::TwoStepLangevinFlow, m_seed, h_tag.data[idx], timestep);
        hoomd::UniformDistribution<Scalar> uniform(-coeff, coeff);
        const Scalar3 random = make_scalar3(uniform(rng), uniform(rng), uniform(rng));

        const Scalar4 velmass = h_vel.data[idx];
        Scalar3 vel = make_scalar3(velmass.x, velmass.y, velmass.z);
        const Scalar mass = velmass.w;

        // total BD force
        Scalar3 bd_force = random - gamma * (vel - flow_vel);

        // compute the new acceleration
        const Scalar4 net_force = h_net_force.data[idx];
        Scalar3 accel = make_scalar3(net_force.x,net_force.y,net_force.z);
        accel += bd_force;
        const Scalar minv = Scalar(1.0) / mass;
        accel.x *= minv;
        accel.y *= minv;
        accel.z *= minv;

        // update the velocity
        vel += Scalar(0.5) * m_deltaT * accel;

        // write out update velocity and acceleration
        h_vel.data[idx] = make_scalar4(vel.x, vel.y, vel.z, mass);
        h_accel.data[idx] = accel;
        }

    if (m_prof) m_prof->pop();
    }

namespace detail
{
//! Export TwoStepLangevinFlow to python
template<class FlowField>
void export_TwoStepLangevinFlow(pybind11::module& m, const std::string& name)
    {
    namespace py = pybind11;
    typedef TwoStepLangevinFlow<FlowField> LangevinFlow;

    py::class_<LangevinFlow, std::shared_ptr<LangevinFlow> >(m, name.c_str(), py::base<TwoStepLangevinBase>())
        .def(py::init<std::shared_ptr<SystemDefinition>,
                      std::shared_ptr<ParticleGroup>,
                      std::shared_ptr<Variant>,
                      std::shared_ptr<FlowField>,
                      unsigned int,
                      bool,
                      Scalar,
                      bool>())
        .def("setFlowField", &LangevinFlow::setFlowField)
        .def("setNoiseless", &LangevinFlow::setNoiseless);
    }
} // end namespace detail
} // end namespace azplugins

#endif // AZPLUGINS_TWO_STEP_LANGEVIN_FLOW_H_
