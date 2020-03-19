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

void export_AntiSymCosGeometry(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<AntiSymCosGeometry, std::shared_ptr<AntiSymCosGeometry> >(m, "AntiSymCosGeometry")
    .def(py::init<Scalar, Scalar, Scalar, unsigned int,Scalar, mpcd::detail::boundary>())
    .def("getHwide", &AntiSymCosGeometry::getHwide)
    .def("getHnarrow", &AntiSymCosGeometry::getHnarrow)
    .def("getRepetitions", &AntiSymCosGeometry::getRepetitions)
    .def("getVelocity", &AntiSymCosGeometry::getVelocity)
    .def("getBoundaryCondition", &AntiSymCosGeometry::getBoundaryCondition);
    }

void export_SymCosGeometry(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<SymCosGeometry, std::shared_ptr<SymCosGeometry> >(m, "SymCosGeometry")
    .def(py::init<Scalar, Scalar, Scalar, unsigned int,Scalar, mpcd::detail::boundary>())
    .def("getHwide", &SymCosGeometry::getHwide)
    .def("getHnarrow", &SymCosGeometry::getHnarrow)
    .def("getRepetitions", &SymCosGeometry::getRepetitions)
    .def("getVelocity", &SymCosGeometry::getVelocity)
    .def("getBoundaryCondition", &SymCosGeometry::getBoundaryCondition);
    }

} // end namespace detail
} // end namespace azplugins
