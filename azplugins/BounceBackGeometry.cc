// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

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

void export_SinusoidalChannel(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<SinusoidalChannel, std::shared_ptr<SinusoidalChannel> >(m, "SinusoidalChannel")
    .def(py::init<Scalar, Scalar, Scalar, unsigned int, Scalar, mpcd::detail::boundary>())
    .def("getAmplitude", &SinusoidalChannel::getAmplitude)
    .def("getHnarrow", &SinusoidalChannel::getHnarrow)
    .def("getRepetitions", &SinusoidalChannel::getRepetitions)
    .def("getBoundaryCondition", &SinusoidalChannel::getBoundaryCondition);
    }

void export_SinusoidalExpansionConstriction(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<SinusoidalExpansionConstriction, std::shared_ptr<SinusoidalExpansionConstriction> >(m, "SinusoidalExpansionConstriction")
    .def(py::init<Scalar, Scalar, Scalar, unsigned int, Scalar, mpcd::detail::boundary>())
    .def("getHwide", &SinusoidalExpansionConstriction::getHwide)
    .def("getHnarrow", &SinusoidalExpansionConstriction::getHnarrow)
    .def("getRepetitions", &SinusoidalExpansionConstriction::getRepetitions)
    .def("getBoundaryCondition", &SinusoidalExpansionConstriction::getBoundaryCondition);
    }

} // end namespace detail
} // end namespace azplugins
