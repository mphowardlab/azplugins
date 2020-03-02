// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: wes_reinhart

/*!
 * \file OrientationRestraintCompute.cc
 * \brief Definition of OrientationRestraintCompute
 */

#include "OrientationRestraintCompute.h"

#include "hoomd/md/QuaternionMath.h"
#include "hoomd/HOOMDMath.h"

namespace azplugins
{
/*!
 * \param sysdef SystemDefinition containing the ParticleData to compute torques on
 * \param group A group of particles
 */
OrientationRestraintCompute::OrientationRestraintCompute(std::shared_ptr<SystemDefinition> sysdef,
                                                         std::shared_ptr<ParticleGroup> group)
        : ForceCompute(sysdef), m_group(group)
    {
    m_exec_conf->msg->notice(5) << "Constructing OrientationRestraintCompute" << std::endl;;

    GPUArray<Scalar4> ref_orient(m_pdata->getN(), m_exec_conf);
    m_ref_orient.swap(ref_orient);

    setForceConstant(Scalar(0.0));

    // MPI is not supported (communication between ranks not implemented)
    #ifdef ENABLE_MPI
    if(m_exec_conf->getNRanks() > 1)
        {
        m_exec_conf->msg->error() << "restrain.orientation: MPI is not supported" << std::endl;
        throw std::runtime_error("restrain.orientation: MPI is not supported");
        }
    #endif

    // single precision is dangerous for rotational dynamics
    // (the energy and torque may not be conservative)
    #ifdef SINGLE_PRECISION
    m_exec_conf->msg->warning() << "restrain.orientation: energy and torques may not be conservative in SINGLE precision" << std::endl;
    #endif

    setInitialOrientations();

    m_logname_list.push_back("restraint_orientation_energy");
    }

OrientationRestraintCompute::~OrientationRestraintCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying OrientationRestraintCompute" << std::endl;;
    }

void OrientationRestraintCompute::setInitialOrientations()
    {
    assert(m_ref_orient.getNumElements() == m_pdata->getN());

    ArrayHandle<Scalar4> h_orient(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_ref_orient(m_ref_orient, access_location::host, access_mode::overwrite);

    memcpy(h_ref_orient.data, h_orient.data, sizeof(Scalar4) * m_pdata->getN());
    }

/*!
 * \param tag Index of particle to set reference for
 * \param orient Quaternion to set as the reference orientation
 */
void OrientationRestraintCompute::setOrientation(unsigned int tag, Scalar4 &orient)
    {
    ArrayHandle<Scalar4> h_ref_orient(m_ref_orient, access_location::host, access_mode::overwrite);
    h_ref_orient.data[tag] = orient;
    }

std::vector<std::string> OrientationRestraintCompute::getProvidedLogQuantities()
    {
    return m_logname_list;
    }

Scalar OrientationRestraintCompute::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    compute(timestep);
    if (quantity == m_logname_list[0])
        {
        return calcEnergySum();
        }
    else
        {
        m_exec_conf->msg->error() << "restrain.orientation: " << quantity
                                  << " is not a valid log quantity for OrientationRestraintCompute" << std::endl;
        throw std::runtime_error("Error getting log value");
        }
    }

/*!
 * \param timestep Current timestep
 */
void OrientationRestraintCompute::computeForces(unsigned int timestep)
    {
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> h_torque(m_torque, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);
    // zero the torques and virial
    memset((void*)h_force.data, 0, sizeof(Scalar4) * m_force.getNumElements());
    memset((void*)h_torque.data, 0, sizeof(Scalar4) * m_torque.getNumElements());
    memset((void*)h_virial.data, 0, sizeof(Scalar) * m_virial.getNumElements());

    ArrayHandle<Scalar4> h_ref_orient(m_ref_orient, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orient(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_member_idx(m_group->getIndexArray(), access_location::host, access_mode::read);

    for (unsigned int cur_idx = 0; cur_idx < m_group->getNumMembers(); ++cur_idx)
        {
        const unsigned int cur_p = h_member_idx.data[cur_idx];
        const Scalar4 cur_orient = h_orient.data[cur_p];

        const unsigned int cur_tag = h_tag.data[cur_p];
        const Scalar4 cur_ref_orient = h_ref_orient.data[cur_tag];

        // convert patch vector in the body frame of each particle to space frame
        vec3<Scalar> n_i = rotate(quat<Scalar>(cur_orient), vec3<Scalar>(1.0, 0, 0));
        vec3<Scalar> n_ref = rotate(quat<Scalar>(cur_ref_orient), vec3<Scalar>(1.0, 0, 0));

        // compute angle between current and initial orientation
        Scalar orient_dot = dot(n_i,n_ref);

        // compute energy
        // U = k * sin(theta)^2
        //   = k * [ 1 - cos(theta)^2 ]
        //   = k * [ 1 - (n_i \dot n_ref)^2 ]
        Scalar energy = m_k * ( Scalar(1.0) - orient_dot * orient_dot );

        // compute torque
        // T = -dU/d(n_i \dot n_ref) * (n_i x n_ref)
        //   = -k * [ 1 - 2 (n_i \dot n_ref) ] * (n_i x n_ref)
        // const Scalar dUddot = ( Scalar(1.0) - Scalar(2.0) * orient_dot );
        const Scalar dUddot = Scalar(-2.0) * m_k * orient_dot;
        vec3<Scalar> torque_dir = cross(n_i,n_ref);
        const Scalar3 torque = vec_to_scalar3(Scalar(-1.0) * dUddot * torque_dir );

        h_torque.data[cur_p] = make_scalar4(torque.x,
            torque.y,
            torque.z,
            0.0);

        h_force.data[cur_p].w = energy;
        }
    }

namespace detail
{
/*!
 * \param m Python module to export to
 */
void export_OrientationRestraintCompute(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_< OrientationRestraintCompute, std::shared_ptr<OrientationRestraintCompute> >
        (m, "OrientationRestraintCompute", py::base<ForceCompute>() )
        .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup> >())
        .def("setForceConstant", &OrientationRestraintCompute::setForceConstant)
        .def("setOrientation", &OrientationRestraintCompute::setOrientation);
    }
} // end namespace detail
} // end namespace azplugins
