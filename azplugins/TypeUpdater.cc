// Copyright (c) 2016, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file TypeUpdater.cc
 * \brief Definition of TypeUpdater
 */

#include "TypeUpdater.h"

namespace azplugins
{

/*!
 * \param sysdef System definition
 *
 * The system is initialized in a configuration that will be invalid on the
 * first check of the types and region. This constructor requires that the user
 * properly initialize the system via setters.
 */
TypeUpdater::TypeUpdater(std::shared_ptr<SystemDefinition> sysdef)
        : Updater(sysdef), m_inside_type(0xffffffff), m_outside_type(0xffffffff),
          m_z_lo(1.0), m_z_hi(-1.0), m_check_types(true), m_check_region(true)
    {
    m_exec_conf->msg->notice(5) << "Constructing TypeUpdater" << std::endl;

    // subscribe to number of type signal, in case number of types is allowed to decrease in future versions of hoomd
    m_pdata->getNumTypesChangeSignal().connect<TypeUpdater, &TypeUpdater::requestCheckTypes>(this);
    // subscribe to box change signal, to ensure region stays in box
    m_pdata->getBoxChangeSignal().connect<TypeUpdater, &TypeUpdater::requestCheckRegion>(this);
    }

/*!
 * \param sysdef System definition
 * \param inside_type Type id of particles inside region
 * \param outside_type Type id of particles outside region
 * \param z_lo Lower bound of region in z
 * \param z_hi Upper bound of region in z
 */
TypeUpdater::TypeUpdater(std::shared_ptr<SystemDefinition> sysdef,
                         unsigned int inside_type,
                         unsigned int outside_type,
                         Scalar z_lo,
                         Scalar z_hi)
        : Updater(sysdef), m_inside_type(inside_type), m_outside_type(outside_type),
          m_z_lo(z_lo), m_z_hi(z_hi), m_check_types(true), m_check_region(true)
    {
    m_exec_conf->msg->notice(5) << "Constructing TypeUpdater" << std::endl;

    // subscribe to number of type signal, in case number of types is allowed to decrease in future versions of hoomd
    m_pdata->getNumTypesChangeSignal().connect<TypeUpdater, &TypeUpdater::requestCheckTypes>(this);
    // subscribe to box change signal, to ensure region stays in box
    m_pdata->getBoxChangeSignal().connect<TypeUpdater, &TypeUpdater::requestCheckRegion>(this);
    }

TypeUpdater::~TypeUpdater()
    {
    m_exec_conf->msg->notice(5) << "Destroying TypeUpdater" << std::endl;
    m_pdata->getNumTypesChangeSignal().disconnect<TypeUpdater, &TypeUpdater::requestCheckTypes>(this);
    m_pdata->getBoxChangeSignal().disconnect<TypeUpdater, &TypeUpdater::requestCheckRegion>(this);
    }

void TypeUpdater::update(unsigned int timestep)
    {
    if (m_check_types)
        {
        checkTypes();
        m_check_types = false;
        }

    if (m_check_region)
        {
        checkRegion();
        m_check_region = false;
        }

    changeTypes(timestep);
    }

void TypeUpdater::changeTypes(unsigned int timestep)
    {
    if (m_prof) m_prof->push("type update");

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);

    for (unsigned int i=0; i < m_pdata->getN(); ++i)
        {
        const Scalar4 postype = h_pos.data[i];
        const Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
        unsigned int type = __scalar_as_int(postype.w);

        // only check region if type is one that can be flipped
        if (type == m_inside_type || type == m_outside_type)
            {
            // test for overlap as for an AABB
            bool inside = !(pos.z > m_z_hi || pos.z < m_z_lo);
            if (inside)
                {
                type = m_inside_type;
                }
            else
                {
                type = m_outside_type;
                }
            }
        h_pos.data[i] = make_scalar4(pos.x, pos.y, pos.z, __int_as_scalar(type));
        }

    if (m_prof) m_prof->pop();
    }

void TypeUpdater::checkTypes() const
    {
    if (m_inside_type >= m_pdata->getNTypes())
        {
        m_exec_conf->msg->error() << "TypeUpdater: inside type id " << m_inside_type
                                  << " is not a valid particle type." << std::endl;
        throw std::runtime_error("Invalid inside type for TypeUpdater");
        }

    if (m_outside_type >= m_pdata->getNTypes())
        {
        m_exec_conf->msg->error() << "TypeUpdater: outside type id " << m_outside_type
                                  << " is not a valid particle type." << std::endl;
        throw std::runtime_error("Invalid outside type for TypeUpdater");
        }

    if (m_inside_type == m_outside_type)
        {
        const std::string name = m_pdata->getNameByType(m_inside_type);
        m_exec_conf->msg->error() << "TypeUpdater: inside and outside type (" << name << ") cannot match." << std::endl;
        throw std::runtime_error("Inside and outside types cannot match in TypeUpdater");
        }
    }

void TypeUpdater::checkRegion() const
    {
    // region cannot be inverted
    if (m_z_lo > m_z_hi)
        {
        m_exec_conf->msg->error() << "TypeUpdater: lower z bound " << m_z_lo << " > upper z bound " << m_z_hi << "." << std::endl;
        throw std::runtime_error("Lower and upper region bounds inverted in TypeUpdater");
        }

    // region cannot cross global box boundaries
    const BoxDim& global_box = m_pdata->getGlobalBox();
    const Scalar3 global_lo = global_box.getLo();
    if (m_z_lo < global_lo.z)
        {
        m_exec_conf->msg->error() << "TypeUpdater: lower z bound " << m_z_lo
                                  << " lies outside simulation box lower bound " << global_lo.z << "." << std::endl;
        throw std::runtime_error("Lower bound outside simulation box in TypeUpdater");
        }
    const Scalar3 global_hi = global_box.getHi();
    if (m_z_hi > global_hi.z)
        {
        m_exec_conf->msg->error() << "TypeUpdater: upper z bound " << m_z_hi
                                  << " lies outside simulation box upper bound " << global_hi.z << "." << std::endl;
        throw std::runtime_error("Upper bound outside simulation box in TypeUpdater");
        }
    }

namespace detail
{
/*!
 * \param m Python module to export to
 */
void export_TypeUpdater(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_< TypeUpdater, std::shared_ptr<TypeUpdater> >(m, "TypeUpdater", py::base<Updater>())
        .def(py::init< std::shared_ptr<SystemDefinition> >())
        .def(py::init<std::shared_ptr<SystemDefinition>, unsigned int, unsigned int, Scalar, Scalar>())
        .def_property("inside", &TypeUpdater::getInsideType, &TypeUpdater::setInsideType)
        .def_property("outside", &TypeUpdater::getOutsideType, &TypeUpdater::setOutsideType)
        .def_property("lo", &TypeUpdater::getRegionLo, &TypeUpdater::setRegionLo)
        .def_property("hi", &TypeUpdater::getRegionHi, &TypeUpdater::setRegionHi);
    }
}

} // end namespace azplugins
