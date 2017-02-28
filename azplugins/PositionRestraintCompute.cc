// Copyright (c) 2016-2017, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: wes_reinhart

/*!
 * \file PositionRestraintCompute.cc
 * \brief Definition of PositionRestraintCompute
 */

#include "PositionRestraintCompute.h"

namespace azplugins
{
/*!
 * \param sysdef SystemDefinition containing the ParticleData to compute forces on
 * \param group A group of particles
 */
PositionRestraintCompute::PositionRestraintCompute(std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<ParticleGroup> group)
        : ForceCompute(sysdef), m_group(group), m_has_warned(false)
    {
    m_exec_conf->msg->notice(5) << "Constructing PositionRestraintCompute" << std::endl;;

    GPUArray<Scalar4> ref_pos(m_pdata->getN(), m_exec_conf);
    m_ref_pos.swap(ref_pos);

    setForceConstant(Scalar(0.0), Scalar(0.0), Scalar(0.0));

    // MPI is not supported (communication between ranks not implemented)
    #ifdef ENABLE_MPI
    if(m_exec_conf->getNRanks() > 1)
        {
        m_exec_conf->msg->error() << "restrain.position: MPI is not supported" << std::endl;
        throw std::runtime_error("restrain.position: MPI is not supported");
        }
    #endif

    setInitialPositions();

    m_logname_list.push_back("restraint_position_energy");
    }

PositionRestraintCompute::~PositionRestraintCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying PositionRestraintCompute" << std::endl;;
    }

void PositionRestraintCompute::setInitialPositions()
    {
    assert(m_ref_pos);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_ref_pos(m_ref_pos, access_location::host, access_mode::overwrite);

    // copy data to ref
    memcpy(h_ref_pos.data, h_pos.data, sizeof(Scalar4) * m_pdata->getN());
    }

void PositionRestraintCompute::setPosition(unsigned int tag, Scalar4 &pos)
    {
    ArrayHandle<Scalar4> h_ref_pos(m_ref_pos, access_location::host, access_mode::overwrite);
    h_ref_pos.data[tag] = pos;
    }

std::vector<std::string> PositionRestraintCompute::getProvidedLogQuantities()
    {
    return m_logname_list;
    }

Scalar PositionRestraintCompute::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    compute(timestep);
    if (quantity == m_logname_list[0])
        {
        return calcEnergySum();
        }
    else
        {
        m_exec_conf->msg->error() << "restrain.position: " << quantity
                                  << " is not a valid log quantity for PositionRestraintCompute" << std::endl;
        throw std::runtime_error("Error getting log value");
        }
    }

/*!
 * \param timestep Current timestep
 */
void PositionRestraintCompute::computeForces(unsigned int timestep)
    {
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);
    // zero the forces and virial
    memset((void*)h_force.data, 0, sizeof(Scalar4) * m_force.getNumElements());
    memset((void*)h_virial.data, 0, sizeof(Scalar) * m_virial.getNumElements());

    // the logger always thinks virial will be computed
    PDataFlags flags = this->m_pdata->getFlags();
    if (!m_has_warned && (flags[pdata_flag::pressure_tensor] || flags[pdata_flag::isotropic_virial]))
        {
        m_exec_conf->msg->warning() <<
            "Restraints do not support virial calculation, pressure will be inaccurate" << std::endl;
        m_has_warned = true;
        }

    ArrayHandle<Scalar4> h_ref_pos(m_ref_pos, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_member_idx(m_group->getIndexArray(), access_location::host, access_mode::read);

    const BoxDim& box = m_pdata->getBox();
    for (unsigned int cur_idx = 0; cur_idx < m_group->getNumMembers(); ++cur_idx)
        {
        const unsigned int cur_p = h_member_idx.data[cur_idx];
        const Scalar4 cur_pos_type = h_pos.data[cur_p];
        const Scalar3 cur_pos = make_scalar3(cur_pos_type.x, cur_pos_type.y, cur_pos_type.z);

        const unsigned int cur_tag = h_tag.data[cur_p];
        const Scalar4 cur_ref_pos_type = h_ref_pos.data[cur_tag];
        const Scalar3 cur_ref_pos = make_scalar3(cur_ref_pos_type.x, cur_ref_pos_type.y, cur_ref_pos_type.z);

        // compute distance between current and initial position
        Scalar3 dr = box.minImage(cur_pos - cur_ref_pos);

        // termwise squaring for energy calculation
        const Scalar3 dr2 = make_scalar3(dr.x*dr.x, dr.y*dr.y, dr.z*dr.z);

        const Scalar3 force = make_scalar3(-m_k.x*dr.x, -m_k.y*dr.y, -m_k.z*dr.z);

        // F = -k x, U = 0.5 kx^2
        h_force.data[cur_p] = make_scalar4(force.x,
                                           force.y,
                                           force.z,
                                           Scalar(0.5)*dot(m_k, dr2));
        }
    }

namespace detail
{
/*!
 * \param m Python module to export to
 */
void export_PositionRestraintCompute(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_< PositionRestraintCompute, std::shared_ptr<PositionRestraintCompute> >
        (m, "PositionRestraintCompute", py::base<ForceCompute>() )
        .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup> >())
        .def("setForceConstant", &PositionRestraintCompute::setForceConstant)
        .def("setPosition", &PositionRestraintCompute::setPosition);
    }
} // end namespace detail
} // end namespace azplugins
