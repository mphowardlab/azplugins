// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file ImplicitEvaporatorGPU.cc
 * \brief Definition of ImplicitEvaporatorGPU
 */

#include "ImplicitEvaporatorGPU.h"

namespace azplugins
{

/*!
 * \param sysdef System definition
 * \param interf Position of the interface
 */
ImplicitEvaporatorGPU::ImplicitEvaporatorGPU(std::shared_ptr<SystemDefinition> sysdef,
                                             std::shared_ptr<Variant> interf)
        : ImplicitEvaporator(sysdef, interf)
    {
    m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "implicit_evap", m_exec_conf));
    }

namespace detail
{
/*!
 * \param m Python module to export to
 */
void export_ImplicitEvaporatorGPU(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<ImplicitEvaporatorGPU,std::shared_ptr<ImplicitEvaporatorGPU>>(m, "ImplicitEvaporatorGPU", py::base<ImplicitEvaporator>())
        .def(py::init<std::shared_ptr<SystemDefinition>,std::shared_ptr<Variant>>())
    ;
    }
} // end namespace detail

} // end namespace azplugins
