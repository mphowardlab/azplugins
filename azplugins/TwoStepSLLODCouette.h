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
                            bool noiseless)
        : IntegrationMethodTwoStep(sysdef, group), m_T(T), m_seed(seed),
            m_use_lambda(use_lambda), m_lambda(lambda), m_gamma_dot(gamma_dot), m_noiseless(noiseless)
            {
            m_exec_conf->msg->notice(5) << "Constructing TwoStepSLLODCouette" << std::endl;

            if (use_lambda)
                m_exec_conf->msg->notice(2) << "flow.sllod is determining gamma from particle diameters" << std::endl;
            else
                m_exec_conf->msg->notice(2) << "flow.sllod is using specified gamma values" << std::endl;

            // In case of MPI run, every rank should be initialized with the same seed.
            // For simplicity we broadcast the seed of rank 0 to all ranks.

            #ifdef ENABLE_MPI
            if( this->m_pdata->getDomainDecomposition() )
                bcast(m_seed,0,this->m_exec_conf->getMPICommunicator());
            #endif

            // Hash the User's Seed to make it less likely to be a low positive integer
            m_seed = m_seed*0x12345677 + 0x12345 ; m_seed^=(m_seed>>16); m_seed*= 0x45679;

            // allocate memory for the per-type gamma storage and initialize them to 1.0
            GlobalVector<Scalar> gamma(m_pdata->getNTypes(), m_exec_conf);
            m_gamma.swap(gamma);
            TAG_ALLOCATION(m_gamma);

            ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::overwrite);
            for (unsigned int i = 0; i < m_gamma.size(); i++)
                h_gamma.data[i] = Scalar(1.0);

            #ifdef ENABLE_CUDA
            if (m_exec_conf->isCUDAEnabled() && m_exec_conf->allConcurrentManagedAccess())
                {
                cudaMemAdvise(m_gamma.get(), sizeof(Scalar)*m_gamma.getNumElements(), cudaMemAdviseSetReadMostly, 0);
                }
            #endif
            }

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

void TwoStepSLLODCouette::integrateStepOne(unsigned int timestep)
    {
    if (m_aniso)
        {
        m_exec_conf->msg->error() << "azplugins.flow: anisotropic particles are not supported with couette flow integrators." << std::endl;
        throw std::runtime_error("Anisotropic integration not supported with couette flow");
        }
    if (m_prof) m_prof->push("SLLOD-Couette step 1");

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::read);

    // Deform box
    BoxDim newBox = m_pdata->getGlobalBox();
    Scalar3 y = newBox.getLatticeVector(1);
    Scalar xy = newBox.getTiltFactorXY();
    Scalar yz = newBox.getTiltFactorYZ();
    Scalar xz = newBox.getTiltFactorXZ();
    const Scalar boundary_shear = y.y * m_gamma_dot;
    xy += m_gamma_dot * m_deltaT;
    bool flipped = false;
    if (xy > 1){
        xy = -1;
        flipped = true;
    }
    newBox.setTiltFactors(xy, xz, yz);
    m_pdata->setGlobalBox(newBox);
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
        const Scalar3 r_del_u = make_scalar3(m_gamma_dot * postype.y, 0.0, 0.0);

        // shear rate tensor dotted with velocity
        const Scalar3 v_del_u = make_scalar3(m_gamma_dot * velmass.y, 0.0, 0.0);

        // update position
        pos += (vel + r_del_u + Scalar(0.5) * m_deltaT * accel) * m_deltaT;

        // update velocity
        vel += Scalar(0.5) * m_deltaT * (accel - v_del_u);

        // if particle leaves from (+/-) y boundary it gets (-/+) shear_rate
        // note carefully that pair potentials dependent on dv (e.g. DPD)
        // not yet explicitly supported due to minimum image convention
        if (pos.y > y.y/2 + yz*pos.z){
            vel.x -= boundary_shear;
        }
        if (pos.y < -y.y/2 + yz*pos.z){
            vel.x += boundary_shear;
        }

        // Wrap back into box
        if (flipped){
            pos.x *= -1;
        }
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
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    const Scalar currentTemp = m_T->getValue(timestep);

    // v(t+deltaT) = v(t+deltaT/2) + 1/2 * deltaT * (a(t+deltaT) - v(t+deltaT/2)*del_u)
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        unsigned int ptag = h_tag.data[j];

        Scalar gamma;
        if (m_use_lambda)
            gamma = m_lambda*h_diameter.data[j];
        else
            {
            unsigned int type = __scalar_as_int(h_pos.data[j].w);
            gamma = h_gamma.data[type];
            }

        hoomd::RandomGenerator rng(azplugins::RNGIdentifier::TwoStepLangevinFlow, m_seed, ptag, timestep);
        hoomd::UniformDistribution<Scalar> uniform(Scalar(-1), Scalar(1));
        Scalar rx = uniform(rng);
        Scalar ry = uniform(rng);
        Scalar rz = uniform(rng);
        Scalar coeff = fast::sqrt(Scalar(6.0)*gamma*currentTemp/m_deltaT);
        if (m_noiseless)
            coeff = Scalar(0.0);
        Scalar bd_fx = rx*coeff - gamma*(h_vel.data[j].x - h_pos.data[j].y * m_gamma_dot);
        Scalar bd_fy = ry*coeff - gamma*h_vel.data[j].y;
        Scalar bd_fz = rz*coeff - gamma*h_vel.data[j].z;

        // first, calculate acceleration from the net force
        Scalar minv = Scalar(1.0) / h_vel.data[j].w;
        h_accel.data[j].x = (h_net_force.data[j].x + bd_fx)*minv;
        h_accel.data[j].y = (h_net_force.data[j].y + bd_fy)*minv;
        h_accel.data[j].z = (h_net_force.data[j].z + bd_fz)*minv;

        // then, update the velocity
        h_vel.data[j].x += Scalar(1.0/2.0)*(h_accel.data[j].x - h_vel.data[j].y * m_gamma_dot)*m_deltaT;
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
                      std::shared_ptr<Variant>,
                      unsigned int,
                      bool,
                      Scalar,
                      Scalar,
                      bool>())
        .def("setT", &TwoStepSLLODCouette::setT)
        .def("set_gamma_dot", &TwoStepSLLODCouette::set_gamma_dot)
        .def("setNoiseless", &TwoStepSLLODCouette::setNoiseless)
        .def("setGamma", &TwoStepSLLODCouette::setGamma);
    }
} // end namespace detail
} // end namespace azplugins

#endif // AZPLUGINS_TWO_STEP_SLLOD_COUETTE_H_
