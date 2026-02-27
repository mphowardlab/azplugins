// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file ChebyshevAnisotropicPairPotential.h
 * \brief Definition of ChebyshevAnisotropicPairPotential
 */

#include "ChebyshevAnisotropicPairPotential.h"
#include "LinearInterpolator5D.h"

namespace hoomd
    {
namespace azplugins
    {

ChebyshevAnisotropicPairPotential::ChebyshevAnisotropicPairPotential(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<hoomd::md::NeighborList> nlist,
    const Scalar* domain,
    const float r_cut,
    const unsigned int* terms,
    const Scalar* coeffs,
    unsigned int Nterms,
    const Scalar* r0_data,
    const unsigned int* r0_shape)
    : ForceCompute(sysdef), m_nlist(nlist), m_r_cut(r_cut), m_Nterms(Nterms)
    {
    }

ChebyshevAnisotropicPairPotential::~ChebyshevAnisotropicPairPotential() { }

void ChebyshevAnisotropicPairPotential::computeForces(uint64_t timestep)
    {
    if (m_nlist)
        {
        m_nlist->compute(timestep);
        }

    const unsigned int N = m_pdata->getN();

    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_torque(m_torque, access_location::host, access_mode::readwrite);

    for (unsigned int i = 0; i < N; ++i)
        {
        h_force.data[i] = make_scalar4(Scalar(0), Scalar(0), Scalar(0), Scalar(0));
        h_torque.data[i] = make_scalar4(Scalar(0), Scalar(0), Scalar(0), Scalar(0));
        }
    }

namespace detail
    {

void export_ChebyshevAnisotropicPairPotential(pybind11::module& m)
    {
    namespace py = pybind11;
    using NL = hoomd::md::NeighborList;

    py::class_<ChebyshevAnisotropicPairPotential,
               std::shared_ptr<ChebyshevAnisotropicPairPotential>>(
        m,
        "ChebyshevAnisotropicPairPotential",
        py::base<hoomd::ForceCompute>())
        .def(py::init(
            [](std::shared_ptr<SystemDefinition> sysdef,
               std::shared_ptr<NL> nlist,
               py::array_t<Scalar, py::array::c_style | py::array::forcecast> domain,
               float r_cut,
               py::array_t<unsigned int, py::array::c_style | py::array::forcecast> terms,
               py::array_t<Scalar, py::array::c_style | py::array::forcecast> coeffs,
               py::array_t<Scalar, py::array::c_style | py::array::forcecast> r0_data)
            {
                // domain must be (5,2) - rho is always in (0, 1)
                if (domain.ndim() != 2 || domain.shape(0) != 5 || domain.shape(1) != 2)
                    {
                    throw std::runtime_error("domain must have shape (5,2).");
                    }

                // terms must be (Nterms,6)
                if (terms.ndim() != 2 || terms.shape(1) != 6)
                    {
                    throw std::runtime_error("terms must have shape (Nterms,6).");
                    }

                const unsigned int Nterms = static_cast<unsigned int>(terms.shape(0));

                // coeffs must be (Nterms,)
                if (coeffs.ndim() != 1 || static_cast<unsigned int>(coeffs.shape(0)) != Nterms)
                    {
                    throw std::runtime_error("coeffs must have shape (Nterms,).");
                    }

                // r0_data must be 5D
                if (r0_data.ndim() != 5)
                    {
                    throw std::runtime_error("r0_data must be a 5D array.");
                    }

                // Infer r0_shape from r0_data.shape
                std::array<unsigned int, 5> r0_shape;
                for (unsigned int k = 0; k < 5; ++k)
                    {
                    const auto dim = r0_data.shape(k);
                    if (dim < 2)
                        {
                        throw std::runtime_error("r0_data has invalid dimension(s).");
                        }
                    r0_shape[k] = static_cast<unsigned int>(dim);
                    }

                return std::make_shared<ChebyshevAnisotropicPairPotential>(sysdef,
                                                                           nlist,
                                                                           domain.data(),
                                                                           r_cut,
                                                                           terms.data(),
                                                                           coeffs.data(),
                                                                           Nterms,
                                                                           r0_data.data(),
                                                                           r0_shape.data());
            }))
        .def_property_readonly("r_cut", &ChebyshevAnisotropicPairPotential::getRCut)
        .def_property_readonly("n_terms", &ChebyshevAnisotropicPairPotential::getNTerms);
    }

    } // end namespace detail
    } // namespace azplugins
    } // namespace hoomd
