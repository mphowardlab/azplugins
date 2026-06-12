// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#include "PerturbedLennardJonesEvap.h"

#include <algorithm>

namespace hoomd
    {
namespace azplugins
    {
namespace detail
    {
namespace py = pybind11;

void export_PerturbedLennardJonesEvap(py::module& m)
    {
    py::class_<PerturbedLennardJonesEvap, ForceCompute, std::shared_ptr<PerturbedLennardJonesEvap>>(
        m,
        "PerturbedLennardJonesEvap")
        .def(py::init(
            [](std::shared_ptr<SystemDefinition> sysdef,
               std::shared_ptr<hoomd::md::NeighborList> nlist,
               Scalar rcut,
               Scalar epsilon,
               Scalar sigma,
               Scalar time_scale_factor,
               bool energy_shift,
               py::array_t<Scalar, py::array::c_style | py::array::forcecast>
                   attraction_scale_factor_data,
               py::array_t<unsigned int, py::array::c_style | py::array::forcecast>
                   attraction_scale_factor_shape,
               py::array_t<Scalar, py::array::c_style | py::array::forcecast> domain,
               std::shared_ptr<VariantInterpolated> variant)
            {
                if (attraction_scale_factor_shape.size() != 2)
                    throw std::runtime_error("lambda_shape must have 2 elements");
                if (domain.size() != 4)
                    throw std::runtime_error(
                        "domain must have 4 elements [y_lo, y_hi, t_lo, t_hi]");

                const unsigned int* shape_ptr = attraction_scale_factor_shape.data();
                const Scalar* data_ptr = attraction_scale_factor_data.data();
                const Scalar* dom_ptr = domain.data();

                if (attraction_scale_factor_data.size()
                    != static_cast<py::ssize_t>(shape_ptr[0] * shape_ptr[1]))
                    throw std::runtime_error(
                        "attraction_scale_factor_data size does not match lambda_shape");

                PairParametersPerturbedLennardJonesEvap params(epsilon, sigma);
                return std::make_shared<PerturbedLennardJonesEvap>(sysdef,
                                                                   nlist,
                                                                   rcut,
                                                                   time_scale_factor,
                                                                   params,
                                                                   energy_shift,
                                                                   data_ptr,
                                                                   shape_ptr,
                                                                   dom_ptr,
                                                                   variant);
            }))
        .def_property_readonly("rcut", &PerturbedLennardJonesEvap::getRCut)
        .def_property_readonly("epsilon", &PerturbedLennardJonesEvap::getEpsilon)
        .def_property_readonly("sigma", &PerturbedLennardJonesEvap::getSigma);
    }

    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd
