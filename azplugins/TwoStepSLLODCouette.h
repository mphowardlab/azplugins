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
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

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
                            Scalar n_gamma_dot)
        : IntegrationMethodTwoStep(sysdef, group), gamma_dot(n_gamma_dot)
            {
            m_exec_conf->msg->notice(5) << "Constructing TwoStepSLLODCouette" << std::endl;
            if (m_sysdef->getNDimensions() < 2)
                {
                m_exec_conf->msg->error() << "flow.couette is only supported in > 2D" << std::endl;
                throw std::runtime_error("Couette flow is only supported in > 2D");
                }
            }

        //! Destructor
        virtual ~TwoStepSLLODCouette()
            {
            m_exec_conf->msg->notice(5) << "Destroying TwoStepSLLODCouette" << std::endl;
            }

        //! Performs the second step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

    protected:
        Scalar gamma_dot;
    };

void TwoStepSLLODCouette::integrateStepOne(unsigned int timestep)
    {
    if (m_aniso)
        {
        m_exec_conf->msg->error() << "azplugins.integrate: anisotropic particles are not supported with couette flow integrators." << std::endl;
        throw std::runtime_error("Anisotropic integration not supported with couette flow");
        }
    if (m_prof) m_prof->push("SLLOD-Couette step 1");

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

        // shear rate tensor dotted with position
        const Scalar3 r_del_u = make_scalar3(gamma_dot * postype.y, 0.0, 0.0);

        // shear rate tensor dotted with velocity
        const Scalar3 v_del_u = make_scalar3(gamma_dot * velmass.y, 0.0, 0.0);

        // update position
        pos += (vel + r_del_u + Scalar(0.5) * m_deltaT * accel) * m_deltaT;

        // update velocity
        vel += Scalar(0.5) * m_deltaT * (accel - v_del_u);

        // Deform box
        BoxDim newBox = m_pdata->getGlobalBox();
        Scalar3 x = newBox.getLatticeVector(0);
        Scalar3 y = newBox.getLatticeVector(1);
        Scalar3 z = newBox.getLatticeVector(2);
        Scalar xy = newBox.getTiltFactorXY();
        Scalar yz = newBox.getTiltFactorYZ();
        Scalar xz = newBox.getTiltFactorXZ();
        const Scalar boundary_shear = y.y * gamma_dot;
        xy += gamma_dot * m_deltaT;
        if xy > 1:
            xy = -1;
        newBox.setTiltFactors(xy, xz, yz);
        m_pdata->setGlobalBox(newBox);

        // if particle leaves from (+/-) y boundary it gets (-/+) shear_rate
        // note carefully that pair potentials dependent on dv (e.g. DPD)
        // not yet explicitly supported due to minimum image convention
        if pos.y > y.y/2 + yz*pos.z:
            vel.x -= boundary_shear;
        if pos.y < -y.y/2 + yz*pos.z:
            vel.x += boundary_shear;

        // Wrap back into box
        box.wrap(pos,h_image.data[idx]);

        h_pos.data[idx] = make_scalar4(pos.x, pos.y, pos.z, __int_as_scalar(type));
        h_vel.data[idx] = make_scalar4(vel.x, vel.y, vel.z, mass);
        }

    if (m_prof) m_prof->pop();
    }

void TwoStepSLLODCouette::integrateStepTwo(unsigned int timestep)
    {
    if (m_prof) m_prof->push("SLLOD-Couette step 2");

    unsigned int group_size = m_group->getNumMembers();

    const GlobalArray< Scalar4 >& net_force = m_pdata->getNetForce();
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);

    // v(t+deltaT) = v(t+deltaT/2) + 1/2 * deltaT * (a(t+deltaT) - v(t+deltaT/2)*del_u)
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);

        // first, calculate acceleration from the net force
        Scalar minv = Scalar(1.0) / h_vel.data[j].w;
        h_accel.data[j].x = h_net_force.data[j].x*minv;
        h_accel.data[j].y = h_net_force.data[j].y*minv;
        h_accel.data[j].z = h_net_force.data[j].z*minv;

        // then, update the velocity
        h_vel.data[j].x += Scalar(1.0/2.0)*(h_accel.data[j].x - h_vel.data[j].y * gamma_dot)*m_deltaT;
        h_vel.data[j].y += Scalar(1.0/2.0)*h_accel.data[j].y*m_deltaT;
        h_vel.data[j].z += Scalar(1.0/2.0)*h_accel.data[j].z*m_deltaT;
        }

    if (m_prof) m_prof->pop();
    }

namespace detail
{
//! Export TwoStepSLLODCouette to python
void export_TwoStepSLLODCouette(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<TwoStepSLLODCouette, std::shared_ptr<TwoStepSLLODCouette> >(m, "TwoStepSLLODCouette", py::base<IntegrationMethodTwoStep>())
        .def(py::init<std::shared_ptr<SystemDefinition>,
                      std::shared_ptr<ParticleGroup>,
                      Scalar>());
    }
} // end namespace detail
} // end namespace azplugins

#endif // AZPLUGINS_TWO_STEP_SLLOD_COUETTE_H_
