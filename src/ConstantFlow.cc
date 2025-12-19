// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#include "ConstantFlow.h"

namespace hoomd
    {
namespace azplugins
    {
namespace detail
    {
void export_ConstantFlow(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<ConstantFlow, std::shared_ptr<ConstantFlow>>(m, "ConstantFlow")
        .def(py::init<Scalar3>())
        .def_property(
            "velocity",
            [](const ConstantFlow& U)
            {
                const auto field = U.getVelocity();
                return pybind11::make_tuple(field.x, field.y, field.z);
            },
            [](ConstantFlow& U, const pybind11::tuple& field)
            {
                U.setVelocity(make_scalar3(pybind11::cast<Scalar>(field[0]),
                                           pybind11::cast<Scalar>(field[1]),
                                           pybind11::cast<Scalar>(field[2])));
            });
    }
    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd
