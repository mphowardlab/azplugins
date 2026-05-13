// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#include "VariantInterpolated.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>

namespace hoomd
    {
namespace azplugins
    {
namespace detail
    {
namespace py = pybind11;

void export_VariantInterpolated(py::module& m)
    {
    py::class_<VariantInterpolated, Variant, std::shared_ptr<VariantInterpolated>>(
        m,
        "VariantInterpolated")
        .def(py::init(
            [](py::array_t<Scalar, py::array::c_style | py::array::forcecast> data,
               unsigned int n,
               Scalar t_lo,
               Scalar t_hi)

            {
                if (data.ndim() < 1 || data.ndim() > 2)
                    throw std::runtime_error("data must be 1D or 2D (N, 1).");
                if (data.ndim() == 2 && data.shape(1) != 1)
                    throw std::runtime_error("data must have shape (N,) or (N, 1).");

                const py::ssize_t N_ss = data.shape(0);
                if (N_ss < 2)
                    throw std::runtime_error("data must contain at least 2 rows.");

                const unsigned int N = static_cast<unsigned int>(N_ss);
                const Scalar* data_ptr = data.data();

                return std::make_shared<VariantInterpolated>(data_ptr, N, t_lo, t_hi);
            }))
        .def_property_readonly("t_lo", &VariantInterpolated::getTLo)
        .def_property_readonly("t_hi", &VariantInterpolated::getTHi);
    }
    } // namespace detail
    } // namespace azplugins
    } // namespace hoomd
