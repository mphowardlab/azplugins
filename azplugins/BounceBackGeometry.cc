// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file BounceBackGeometry.cc
 * \brief Implementation of valid bounce back geometries.
 *
 * This file is empty because all used geometries are implemented in hoomd v2.6.0
 * Users can add custom geometries here.
 */


#include "BounceBackGeometry.h"

namespace azplugins
{
namespace detail
{

void export_SineGeometry(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<SineGeometry, std::shared_ptr<SineGeometry> >(m, "SineGeometry")
    .def(py::init<Scalar, Scalar, Scalar, Scalar,Scalar, mpcd::detail::boundary>())
    .def("getHwide", &SineGeometry::getHwide)
    .def("getHnarrow", &SineGeometry::getHnarrow)
    .def("getRepetitions", &SineGeometry::getRepetitions)
    .def("getVelocity", &SineGeometry::getVelocity)
    .def("getBoundaryCondition", &SineGeometry::getBoundaryCondition);
    }

} // end namespace detail
} // end namespace azplugins
