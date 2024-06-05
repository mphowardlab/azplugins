// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#include "ParabolicFlow.h"

namespace hoomd
    {
namespace azplugins
    {
namespace detail
    {
void export_ParabolicFlow(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<ParabolicFlow, std::shared_ptr<ParabolicFlow>>(m, "ParabolicFlow")
        .def(py::init<Scalar, Scalar>())
        .def_property("mean_velocity",
                      &ParabolicFlow::getMeanVelocity,
                      &ParabolicFlow::setMeanVelocity)
        .def_property("separation", &ParabolicFlow::getSeparation, &ParabolicFlow::setSeparation);
    }
    } // namespace detail
    } // namespace azplugins
    } // namespace hoomd
