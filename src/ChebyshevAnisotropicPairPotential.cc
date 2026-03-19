// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file ChebyshevAnisotropicPairPotential.h
 * \brief Definition of ChebyshevAnisotropicPairPotential
 */

#include "ChebyshevAnisotropicPairPotential.h"
#include "LinearInterpolator5D.h"

#include "hoomd/BoxDim.h"
#include "hoomd/VectorMath.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace hoomd
    {
namespace azplugins
    {

ChebyshevAnisotropicPairPotential::ChebyshevAnisotropicPairPotential(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<hoomd::md::NeighborList> nlist,
    const Scalar* domain,
    const Scalar r_cut,
    const unsigned int* terms,
    const Scalar* coeffs,
    unsigned int Nterms,
    const Scalar* r0_data,
    const unsigned int* r0_shape)
    : ForceCompute(sysdef), m_nlist(nlist), m_r_cut(r_cut), m_Nterms(Nterms)
    {
        {
        GPUArray<Scalar2> domain_arr(5, m_exec_conf);
        m_domain.swap(domain_arr);

        ArrayHandle<Scalar2> h_domain(m_domain, access_location::host, access_mode::readwrite);
        for (unsigned int d = 0; d < 5; ++d)
            {
            h_domain.data[d] = make_scalar2(domain[2 * d], domain[2 * d + 1]);
            }
        }

        // terms: shape (Nterms, 6), stored flat
        {
        GPUArray<unsigned int> terms_arr(static_cast<size_t>(Nterms) * 6, m_exec_conf);
        m_terms.swap(terms_arr);

        ArrayHandle<unsigned int> h_terms(m_terms, access_location::host, access_mode::readwrite);
        std::copy(terms, terms + static_cast<size_t>(Nterms) * 6, h_terms.data);
        }

        // coeffs: shape (Nterms,)
        {
        GPUArray<Scalar> coeffs_arr(Nterms, m_exec_conf);
        m_coeffs.swap(coeffs_arr);

        ArrayHandle<Scalar> h_coeffs(m_coeffs, access_location::host, access_mode::readwrite);
        std::copy(coeffs, coeffs + Nterms, h_coeffs.data);
        }

        // r0_shape: length 5
        {
        GPUArray<unsigned int> shape_arr(5, m_exec_conf);
        m_r0_shape.swap(shape_arr);

        ArrayHandle<unsigned int> h_shape(m_r0_shape,
                                          access_location::host,
                                          access_mode::readwrite);
        std::copy(r0_shape, r0_shape + 5, h_shape.data);
        }

    // r0_data: flat array, length = product(r0_shape)
    size_t n_r0 = 1;
    for (unsigned int d = 0; d < 5; ++d)
        {
        n_r0 *= static_cast<size_t>(r0_shape[d]);
        }

        {
        GPUArray<Scalar> r0_arr(n_r0, m_exec_conf);
        m_r0_data.swap(r0_arr);

        ArrayHandle<Scalar> h_r0(m_r0_data, access_location::host, access_mode::readwrite);
        std::copy(r0_data, r0_data + n_r0, h_r0.data);
        }
    }

ChebyshevAnisotropicPairPotential::~ChebyshevAnisotropicPairPotential() { }

void ChebyshevAnisotropicPairPotential::computeForces(uint64_t timestep)
    {
    // start by updating the neighborlist
    m_nlist->compute(timestep);

    // access neighbor list, particle data, and simulation box.
    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(),
                                        access_location::host,
                                        access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(),
                                      access_location::host,
                                      access_mode::read);
    ArrayHandle<size_t> h_head_list(m_nlist->getHeadList(),
                                    access_location::host,
                                    access_mode::read);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                       access_location::host,
                                       access_mode::read);
    ArrayHandle<Scalar2> h_domain(m_domain, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_r0_data(m_r0_data, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_r0_shape(m_r0_shape, access_location::host, access_mode::read);

    const BoxDim box = m_pdata->getGlobalBox();
    const Scalar rcutsq = m_r_cut * m_r_cut;
    const Scalar h = Scalar(1.0e-6);

    Scalar lo[5];
    Scalar hi[5];
    for (unsigned int d = 0; d < 5; ++d)
        {
        lo[d] = h_domain.data[d].x;
        hi[d] = h_domain.data[d].y;
        }

    LinearInterpolator5D<Scalar> interp(h_r0_data.data, h_r0_shape.data, lo, hi);

    // need to start from a zero force and torque
    m_force.zeroFill();
    m_torque.zeroFill();

    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_torque(m_torque, access_location::host, access_mode::readwrite);

    const unsigned int N = m_pdata->getN();

    for (unsigned int i = 0; i < N; ++i)
        {
        // particle i position and orientation
        const Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        const quat<Scalar> q_i(h_orientation.data[i]);
        const quat<Scalar> q_i_conj = conj(q_i);

        // initialize current particle force and torque
        Scalar3 fi = make_scalar3(0, 0, 0);
        Scalar3 ti = make_scalar3(0, 0, 0);

        const size_t myHead = h_head_list.data[i];
        const unsigned int size = (unsigned int)h_n_neigh.data[i];

        for (unsigned int k = 0; k < size; ++k)
            {
            // access the index
            const unsigned int j = h_nlist.data[myHead + k];
            assert(j < m_pdata->getN() + m_pdata->getNGhosts());

            const Scalar3 pj = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
            Scalar3 dx = pi - pj;
            // apply periodic boundary conditions
            dx = box.minImage(dx);

            // cut-off check
            const Scalar rsq = dot(dx, dx);
            if (rsq > rcutsq)
                {
                continue;
                }

            // particle j, orientation quaternion
            const quat<Scalar> q_j(h_orientation.data[j]);
            // dx is in lab frame, so rotate dx by conj(q_i)
            const vec3<Scalar> dx_lab(dx.x, dx.y, dx.z);
            const vec3<Scalar> dx_body = rotate(q_i_conj, dx_lab);
            // relative orientation of j with respect to i
            const quat<Scalar> q_rel = q_i_conj * q_j;

            // convert position to spherical coordinates
            const Scalar r = fast::sqrt(dot(dx_body, dx_body));
            Scalar theta = Scalar(0);
            Scalar phi = Scalar(0);

            if (r > Scalar(0))
                {
                theta = std::atan2(dx_body.y, dx_body.x);
                if (theta < Scalar(0))
                    {
                    theta += Scalar(2.0) * M_PI;
                    }

                Scalar cosphi = dx_body.z / r;
                if (cosphi < Scalar(-1))
                    {
                    cosphi = Scalar(-1);
                    }
                else if (cosphi > Scalar(1))
                    {
                    cosphi = Scalar(1);
                    }

                phi = std::acos(cosphi);
                }

            // get the columns of an active rotation matrix
            const vec3<Scalar> ex = rotate(q_rel, vec3<Scalar>(1, 0, 0));
            const vec3<Scalar> ey = rotate(q_rel, vec3<Scalar>(0, 1, 0));
            const vec3<Scalar> ez = rotate(q_rel, vec3<Scalar>(0, 0, 1));

            Scalar alpha = Scalar(0);
            Scalar beta = Scalar(0);
            Scalar gamma = Scalar(0);

            // get the rotation angles by R_ZXZ (body-fixed) = R_q
            if (ez.z < Scalar(-1))
                {
                beta = Scalar(M_PI);
                }
            else if (ez.z > Scalar(1))
                {
                beta = Scalar(0);
                }
            else
                {
                beta = std::acos(ez.z);
                }

            if (beta > Scalar(1e-7) && beta < Scalar(M_PI - 1e-7))
                {
                alpha = std::atan2(ez.x, -ez.y);
                gamma = std::atan2(ex.z, ey.z);
                }
            else if (beta <= Scalar(1e-7))
                {
                alpha = Scalar(0);
                gamma = std::atan2(ex.y, ex.x);
                }
            else
                {
                alpha = Scalar(0);
                gamma = std::atan2(-ex.y, ex.x);
                }

            if (alpha < Scalar(0))
                {
                alpha += Scalar(2) * M_PI;
                }
            if (gamma < Scalar(0))
                {
                gamma += Scalar(2) * M_PI;
                }

            // compute r0 and its derivatives
            const Scalar r0 = interp(theta, phi, alpha, beta, gamma);
            Scalar dr0_dtheta = Scalar(0);
            Scalar dr0_dphi = Scalar(0);
            Scalar dr0_dalpha = Scalar(0);
            Scalar dr0_dbeta = Scalar(0);
            Scalar dr0_dgamma = Scalar(0);

            // d r0 / d theta
            if (theta - h < lo[0])
                {
                dr0_dtheta = (interp(theta + h, phi, alpha, beta, gamma) - r0) / h;
                }
            else if (theta + h > hi[0])
                {
                dr0_dtheta = (r0 - interp(theta - h, phi, alpha, beta, gamma)) / h;
                }
            else
                {
                dr0_dtheta = (interp(theta + h, phi, alpha, beta, gamma)
                              - interp(theta - h, phi, alpha, beta, gamma))
                             / (Scalar(2) * h);
                }

            // d r0 / d phi
            if (phi - h < lo[1])
                {
                dr0_dphi = (interp(theta, phi + h, alpha, beta, gamma) - r0) / h;
                }
            else if (phi + h > hi[1])
                {
                dr0_dphi = (r0 - interp(theta, phi - h, alpha, beta, gamma)) / h;
                }
            else
                {
                dr0_dphi = (interp(theta, phi + h, alpha, beta, gamma)
                            - interp(theta, phi - h, alpha, beta, gamma))
                           / (Scalar(2) * h);
                }

            // d r0 / d alpha
            if (alpha - h < lo[2])
                {
                dr0_dalpha = (interp(theta, phi, alpha + h, beta, gamma) - r0) / h;
                }
            else if (alpha + h > hi[2])
                {
                dr0_dalpha = (r0 - interp(theta, phi, alpha - h, beta, gamma)) / h;
                }
            else
                {
                dr0_dalpha = (interp(theta, phi, alpha + h, beta, gamma)
                              - interp(theta, phi, alpha - h, beta, gamma))
                             / (Scalar(2) * h);
                }

            // d r0 / d beta
            if (beta - h < lo[3])
                {
                dr0_dbeta = (interp(theta, phi, alpha, beta + h, gamma) - r0) / h;
                }
            else if (beta + h > hi[3])
                {
                dr0_dbeta = (r0 - interp(theta, phi, alpha, beta - h, gamma)) / h;
                }
            else
                {
                dr0_dbeta = (interp(theta, phi, alpha, beta + h, gamma)
                             - interp(theta, phi, alpha, beta - h, gamma))
                            / (Scalar(2) * h);
                }

            // d r0 / d gamma
            if (gamma - h < lo[4])
                {
                dr0_dgamma = (interp(theta, phi, alpha, beta, gamma + h) - r0) / h;
                }
            else if (gamma + h > hi[4])
                {
                dr0_dgamma = (r0 - interp(theta, phi, alpha, beta, gamma - h)) / h;
                }
            else
                {
                dr0_dgamma = (interp(theta, phi, alpha, beta, gamma + h)
                              - interp(theta, phi, alpha, beta, gamma - h))
                             / (Scalar(2) * h);
                }

            // compute J
            }

        h_force.data[i].x += fi.x;
        h_force.data[i].y += fi.y;
        h_force.data[i].z += fi.z;
        h_force.data[i].w += Scalar(0.0);

        h_torque.data[i].x += ti.x;
        h_torque.data[i].y += ti.y;
        h_torque.data[i].z += ti.z;
        h_torque.data[i].w += Scalar(0.0);
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
               Scalar r_cut,
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
        .def_property_readonly("num_terms", &ChebyshevAnisotropicPairPotential::getNTerms);
    }

    } // end namespace detail
    } // namespace azplugins
    } // namespace hoomd
