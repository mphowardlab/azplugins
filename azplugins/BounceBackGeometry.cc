// Copyright (c) 2016-2018, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file BounceBackGeometry.cc
 * \brief Implementation of valid bounce back geometries.
 */

#include "BounceBackGeometry.h"

namespace azplugins
{
namespace detail
{

void export_boundary(pybind11::module& m)
    {
    namespace py = pybind11;
    py::enum_<mpcd::detail::boundary>(m, "boundary")
        .value("no_slip", mpcd::detail::boundary::no_slip)
        .value("slip", mpcd::detail::boundary::slip);
    }

void export_SlitGeometry(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<mpcd::detail::SlitGeometry, std::shared_ptr<mpcd::detail::SlitGeometry> >(m, "SlitGeometry")
        .def(py::init<Scalar, Scalar, mpcd::detail::boundary>())
        .def("getH", &mpcd::detail::SlitGeometry::getH)
        .def("getVelocity", &mpcd::detail::SlitGeometry::getVelocity)
        .def("getBoundaryCondition", &mpcd::detail::SlitGeometry::getBoundaryCondition);
    }

} // end namespace detail
} // end namespace azplugins
