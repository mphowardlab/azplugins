// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

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
    py::class_<mpcd::detail::SlitGeometry, std::shared_ptr<const mpcd::detail::SlitGeometry> >(m, "SlitGeometry")
        .def(py::init<Scalar, Scalar, mpcd::detail::boundary>())
        .def("getH", &mpcd::detail::SlitGeometry::getH)
        .def("getVelocity", &mpcd::detail::SlitGeometry::getVelocity)
        .def("getBoundaryCondition", &mpcd::detail::SlitGeometry::getBoundaryCondition);
    }

} // end namespace detail
} // end namespace azplugins
