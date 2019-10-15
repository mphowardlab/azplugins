// Copyright (c) 2018-2019, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file ImplicitEvaporator.cc
 * \brief Definition of ImplicitEvaporator
 */

#include "ImplicitEvaporator.h"

namespace azplugins
{
/*!
 * \param sysdef System definition
 * \param interf Position of the interface
 */
ImplicitEvaporator::ImplicitEvaporator(std::shared_ptr<SystemDefinition> sysdef,
                                       std::shared_ptr<Variant> interf)
        : ForceCompute(sysdef), m_interf(interf), m_has_warned(false)
    {
    // allocate memory per type for parameters
    GPUArray<Scalar4> params(m_pdata->getNTypes(), m_exec_conf);
    m_params.swap(params);

    // connect to type change to resize type data arrays
    m_pdata->getNumTypesChangeSignal().connect<ImplicitEvaporator, &ImplicitEvaporator::reallocateParams>(this);
    }

ImplicitEvaporator::~ImplicitEvaporator()
    {
    m_pdata->getNumTypesChangeSignal().disconnect<ImplicitEvaporator, &ImplicitEvaporator::reallocateParams>(this);
    }

/*!
 * \param timestep Current timestep
 *
 * This method only checks for warnings about the virial.
 * Deriving classes should implement the actual force compute method.
 */
void ImplicitEvaporator::computeForces(unsigned int timestep)
    {
    PDataFlags flags = m_pdata->getFlags();
    if (!m_has_warned && (flags[pdata_flag::pressure_tensor] || flags[pdata_flag::isotropic_virial]))
        {
        m_exec_conf->msg->warning() << "ImplicitPlaneEvaporator does not compute its virial contribution, pressure may be inaccurate" << std::endl;
        m_has_warned = true;
        }
    }

namespace detail
{
/*!
 * \param m Python module to export to
 */
void export_ImplicitEvaporator(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<ImplicitEvaporator,std::shared_ptr<ImplicitEvaporator>>(m, "ImplicitEvaporator", py::base<ForceCompute>())
        .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<Variant>>())
        .def("setParams", &ImplicitEvaporator::setParams);
    }
} // end namespace detail

} // end namespace azplugins
