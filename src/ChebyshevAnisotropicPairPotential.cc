// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file ChebyshevAnisotropicPairPotential.cc
 * \brief Definition of ChebyshevAnisotropicPairPotential
 */

#include "ChebyshevAnisotropicPairPotential.h"
#include "LinearInterpolator5D.h"

namespace hoomd
    {
namespace azplugins
    {

//! Scale a coordinate from [lo, hi] to the Chebyshev domain [-1, 1].
static inline Scalar scaleToChebDomain(Scalar x, Scalar lo, Scalar hi)
    {
    return (Scalar(2) * (x - lo) / (hi - lo)) - Scalar(1);
    }

//! Evaluate Chebyshev polynomials of the first kind and their derivatives
//! from degree 0 up to max_deg, using the three-term recurrence relation.
/*!
    T_0(x) = 1                       T'_0(x) = 0
    T_1(x) = x                       T'_1(x) = 1
    T_{n+1}(x) = 2x T_n - T_{n-1}   T'_{n+1}(x) = 2 T_n + 2x T'_n - T'_{n-1}
*/
static inline void evaluateChebyshev(Scalar x, unsigned int max_deg, Scalar* T, Scalar* dT)
    {
    T[0] = Scalar(1);
    dT[0] = Scalar(0);

    if (max_deg == 0)
        return;

    T[1] = x;
    dT[1] = Scalar(1);

    const Scalar two_x = Scalar(2) * x;
    for (unsigned int n = 1; n < max_deg; ++n)
        {
        T[n + 1] = two_x * T[n] - T[n - 1];
        dT[n + 1] = Scalar(2) * T[n] + two_x * dT[n] - dT[n - 1];
        }
    }

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

        const Index2D domain_index(2, 5);
        ArrayHandle<Scalar2> h_domain(m_domain, access_location::host, access_mode::readwrite);
        for (unsigned int d = 0; d < 5; ++d)
            {
            h_domain.data[d] = make_scalar2(domain[domain_index(0, d)], domain[domain_index(1, d)]);
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
    unsigned int n_r0 = 1;
    for (unsigned int d = 0; d < 5; ++d)
        {
        n_r0 *= r0_shape[d];
        }

        {
        GPUArray<Scalar> r0_arr(n_r0, m_exec_conf);
        m_r0_data.swap(r0_arr);

        ArrayHandle<Scalar> h_r0(m_r0_data, access_location::host, access_mode::readwrite);
        std::copy(r0_data, r0_data + n_r0, h_r0.data);
        }

    // neighbor list subscriber
    Scalar max_r0 = *std::max_element(r0_data, r0_data + n_r0);
    m_nlist_r_cut = std::ceil(max_r0 + m_r_cut);

    m_r_cut_nlist = std::make_shared<GPUArray<Scalar>>(1, m_exec_conf);
        {
        ArrayHandle<Scalar> h_r_cut_nlist(*m_r_cut_nlist,
                                          access_location::host,
                                          access_mode::overwrite);
        h_r_cut_nlist.data[0] = m_nlist_r_cut;
        }
    m_nlist->addRCutMatrix(m_r_cut_nlist);
    m_nlist->notifyRCutMatrixChange();
    }

ChebyshevAnisotropicPairPotential::~ChebyshevAnisotropicPairPotential()
    {
    if (m_attached)
        {
        m_nlist->removeRCutMatrix(m_r_cut_nlist);
        }
    }

void ChebyshevAnisotropicPairPotential::notifyDetach()
    {
    if (m_attached)
        {
        m_nlist->removeRCutMatrix(m_r_cut_nlist);
        }
    m_attached = false;
    }

void ChebyshevAnisotropicPairPotential::computeForces(uint64_t timestep)
    {
    // start by updating the neighborlist
    m_nlist->compute(timestep);

    // check neighbor list storage mode
    const bool third_law = (m_nlist->getStorageMode() == hoomd::md::NeighborList::half);
    // access neighbor list, particle data, and simulation box
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
    ArrayHandle<unsigned int> h_terms(m_terms, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_coeffs(m_coeffs, access_location::host, access_mode::read);

    const BoxDim box = m_pdata->getGlobalBox();
    const Scalar nlist_rcutsq = m_nlist_r_cut * m_nlist_r_cut;
    const Scalar fd_step = Scalar(1.0e-6);

    LinearInterpolator5D<Scalar> interp(h_r0_data.data, h_r0_shape.data, h_domain.data);

    // determine the maximum Chebyshev degree needed for each of the 6 coordinates
    unsigned int max_deg[6] = {0, 0, 0, 0, 0, 0};
    for (unsigned int t = 0; t < m_Nterms; ++t)
        {
        for (unsigned int c = 0; c < 6; ++c)
            {
            const unsigned int deg = h_terms.data[t * 6 + c];
            if (deg > max_deg[c])
                max_deg[c] = deg;
            }
        }

    // chain-rule scale factors: d(x_scaled)/d(x) = 2 / (hi - lo)
    Scalar cheb_scale[6];
    cheb_scale[0] = Scalar(2);
    for (unsigned int d = 0; d < 5; ++d)
        {
        cheb_scale[d + 1] = Scalar(2) / (h_domain.data[d].y - h_domain.data[d].x);
        }

    // flat 1D Chebyshev storage
    unsigned int max_deg_global = 0;
    for (unsigned int c = 0; c < 6; ++c)
        {
        if (max_deg[c] > max_deg_global)
            max_deg_global = max_deg[c];
        }

    const Index2D cheb_idx(max_deg_global + 1, 6);
    std::vector<Scalar> cheb_T_flat(cheb_idx.getNumElements());
    std::vector<Scalar> cheb_dT_flat(cheb_idx.getNumElements());

    // zero force and torque
    m_force.zeroFill();
    m_torque.zeroFill();

    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_torque(m_torque, access_location::host, access_mode::readwrite);

    const unsigned int N = m_pdata->getN();
    //! Euler-angle singularity tolerance for the alpha/gamma extraction.
    const Scalar euler_singularity_tol = Scalar(1e-7);

    //! beta threshold for the Jacobian (avoids 1/sin(beta) singulrity).
    const Scalar beta_tol = Scalar(1e-5);

    for (unsigned int i = 0; i < N; ++i)
        {
        // particle i position and orientation
        const Scalar3 pos_i = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        const quat<Scalar> q_i(h_orientation.data[i]);
        const quat<Scalar> q_i_conj = conj(q_i);

        // initialize particle force, torque, and energy
        Scalar3 fi = make_scalar3(0, 0, 0);
        Scalar3 ti = make_scalar3(0, 0, 0);
        Scalar pei = Scalar(0);

        const size_t myHead = h_head_list.data[i];
        const unsigned int size = (unsigned int)h_n_neigh.data[i];

        for (unsigned int k = 0; k < size; ++k)
            {
            // access the index
            const unsigned int j = h_nlist.data[myHead + k];
            assert(j < m_pdata->getN() + m_pdata->getNGhosts());

            const Scalar3 pos_j = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
            Scalar3 dx = pos_i - pos_j;
            // apply periodic boundary conditions
            dx = box.minImage(dx);
            // Neighbor-list cutoff check (center-center distance).
            const Scalar rsq = dot(dx, dx);
            if (rsq > nlist_rcutsq)
                {
                continue;
                }

            // particle j, orientation quaternion
            const quat<Scalar> q_j(h_orientation.data[j]);
            // dx is in lab frame, so rotate dx by conj(q_i)
            const vec3<Scalar> dx_body = rotate(q_i_conj, vec3<Scalar>(dx));

            // Relative orientation of j with respect to i:
            //     q_rel = conj(q_i) * q_j
            // ref:
            // https://www.mathworks.com/help/fusion/ug/rotations-orientation-and-quaternions.html
            const quat<Scalar> q_rel = q_i_conj * q_j;

            // convert position to spherical coordinates
            const Scalar r = fast::sqrt(dot(dx_body, dx_body));
            Scalar theta = Scalar(0);
            Scalar phi = Scalar(0);

            // skip overlapping particles.
            if (r < Scalar(1e-12))
                {
                continue;
                }

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

            // Build rotation matrix from the relative quaternion and extract
            // ZXZ Euler angles (alpha, beta, gamma)
            const rotmat3<Scalar> R(q_rel);

            Scalar alpha = Scalar(0);
            Scalar beta = Scalar(0);
            Scalar gamma = Scalar(0);

            if (R.row2.z < Scalar(-1))
                {
                beta = Scalar(M_PI);
                }
            else if (R.row2.z > Scalar(1))
                {
                beta = Scalar(0);
                }
            else
                {
                beta = std::acos(R.row2.z);
                }

            if (beta >= euler_singularity_tol && beta <= Scalar(M_PI) - euler_singularity_tol)
                {
                alpha = std::atan2(R.row0.z, -R.row1.z);
                gamma = std::atan2(R.row2.x, R.row2.y);
                if (alpha < Scalar(0))
                    {
                    alpha += Scalar(2) * M_PI;
                    }
                }
            else
                {
                alpha = Scalar(0);
                gamma
                    = std::atan2((beta <= euler_singularity_tol) ? R.row0.y : -R.row0.y, R.row0.x);
                }

            if (gamma < Scalar(0))
                {
                gamma += Scalar(2) * M_PI;
                }

            // move phi and beta away from 0 and pi to avoid 1/sin(beta or phi)
            // singularity in the Jacobian (used the same threshold as beta).
            if (phi < beta_tol)
                phi = beta_tol;
            else if (phi > Scalar(M_PI) - beta_tol)
                phi = Scalar(M_PI) - beta_tol;

            if (beta < beta_tol)
                beta = beta_tol;
            else if (beta > Scalar(M_PI) - beta_tol)
                beta = Scalar(M_PI) - beta_tol;

            // compute r0 and all 5 derivatives
            Scalar r0;
            Scalar dr0[5];
            interp.valueAndDerivatives(theta, phi, alpha, beta, gamma, fd_step, r0, dr0);
            const Scalar dr0_dtheta = dr0[0];
            const Scalar dr0_dphi = dr0[1];
            const Scalar dr0_dalpha = dr0[2];
            const Scalar dr0_dbeta = dr0[3];
            const Scalar dr0_dgamma = dr0[4];

            // compute rho
            const Scalar inv_r = Scalar(1) / r;
            const Scalar inv_r0 = Scalar(1) / r0;
            const Scalar inv_r0_rcut = Scalar(1) / (r0 + m_r_cut);
            const Scalar rho_denom = inv_r0_rcut - inv_r0;
            const Scalar rho_num = inv_r - inv_r0;
            Scalar rho = rho_num / rho_denom;

            if (rho > Scalar(1))
                {
                continue;
                }

            // save raw rho for energy extrapolation if rho < 0
            const Scalar rho_energy = rho;
            if (rho < Scalar(0))
                {
                rho = Scalar(0);
                }

            // drho/dr and drho/dr0
            const Scalar inv_r_sq = inv_r * inv_r;
            const Scalar inv_r0_sq = inv_r0 * inv_r0;
            const Scalar inv_r0_rcut_sq = inv_r0_rcut * inv_r0_rcut;
            const Scalar rho_denom_sq = rho_denom * rho_denom;

            const Scalar drho_dr = -inv_r_sq / rho_denom;
            const Scalar drho_dr0
                = (inv_r0_sq * rho_denom - rho_num * (inv_r0_sq - inv_r0_rcut_sq)) / rho_denom_sq;

            // Chebyshev evaluation: scale each coordinate to [-1,1]
            // and evaluate polynomials + derivatives up to max degree.
            evaluateChebyshev(scaleToChebDomain(rho, Scalar(0), Scalar(1)),
                              max_deg[0],
                              cheb_T_flat.data() + cheb_idx(0, 0),
                              cheb_dT_flat.data() + cheb_idx(0, 0));

            const Scalar ang_coords[5] = {theta, phi, alpha, beta, gamma};
            for (unsigned int c = 0; c < 5; ++c)
                {
                evaluateChebyshev(
                    scaleToChebDomain(ang_coords[c], h_domain.data[c].x, h_domain.data[c].y),
                    max_deg[c + 1],
                    cheb_T_flat.data() + cheb_idx(0, c + 1),
                    cheb_dT_flat.data() + cheb_idx(0, c + 1));
                }

            // evaluate u and du/d(coord_k)
            Scalar u = Scalar(0);
            Scalar du[6] = {Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0)};

            for (unsigned int t = 0; t < m_Nterms; ++t)
                {
                const unsigned int* degs = h_terms.data + 6 * t;
                const Scalar coeff = h_coeffs.data[t];

                Scalar T_vals[6];
                Scalar dT_vals[6];
                for (unsigned int c = 0; c < 6; ++c)
                    {
                    T_vals[c] = cheb_T_flat[cheb_idx(degs[c], c)];
                    dT_vals[c] = cheb_dT_flat[cheb_idx(degs[c], c)];
                    }

                Scalar prefix[7];
                prefix[0] = Scalar(1);
                for (unsigned int c = 0; c < 6; ++c)
                    {
                    prefix[c + 1] = prefix[c] * T_vals[c];
                    }

                Scalar suffix[7];
                suffix[6] = Scalar(1);
                for (int c = 5; c >= 0; --c)
                    {
                    suffix[c] = suffix[c + 1] * T_vals[c];
                    }

                u += coeff * prefix[6];

                for (unsigned int c = 0; c < 6; ++c)
                    {
                    du[c] += coeff * dT_vals[c] * cheb_scale[c] * prefix[c] * suffix[c + 1];
                    }
                }

            // linear extrapolation for energy when rho < 0
            u = (rho_energy < Scalar(0)) ? (u + rho_energy * du[0]) : u;

            // Jacobian matrix J (6x6).
            // J maps the potential-derivative vector
            // [du/drho, du/dtheta, du/dphi, du/dalpha, du/dbeta, du/dgamma]
            // to the lab-frame force and torque:
            // [F_x, F_y, F_z, tau_x, tau_y, tau_z]
            Scalar s_th, c_th;
            fast::sincos(theta, s_th, c_th);
            Scalar s_ph, c_ph;
            fast::sincos(phi, s_ph, c_ph);
            Scalar s_b, c_b;
            fast::sincos(beta, s_b, c_b);
            Scalar s_a, c_a;
            fast::sincos(alpha, s_a, c_a);

            const Scalar inv_r_s_ph = inv_r / s_ph;
            const Scalar inv_s_b = Scalar(1) / s_b;

            // common products involving drho_dr0 and r0 derivatives
            const Scalar A = drho_dr0 * dr0_dtheta * inv_r_s_ph;
            const Scalar B = drho_dr0 * dr0_dphi * inv_r;
            const Scalar C = drho_dr0 * dr0_dalpha * inv_s_b;
            const Scalar D = drho_dr0 * dr0_dgamma * inv_s_b;

            // force (lab frame)
            const Scalar f_x = (-c_th * s_ph * drho_dr + s_th * A - c_th * c_ph * B) * du[0]
                               + (s_th * inv_r_s_ph) * du[1] + (-c_th * c_ph * inv_r) * du[2];

            const Scalar f_y = (-s_th * s_ph * drho_dr - c_th * A - s_th * c_ph * B) * du[0]
                               + (-c_th * inv_r_s_ph) * du[1] + (-s_th * c_ph * inv_r) * du[2];

            const Scalar f_z = (-c_ph * drho_dr + s_ph * B) * du[0] + (s_ph * inv_r) * du[2];

            // torque (lab frame)
            const Scalar tau_x = (c_b * s_a * C - c_a * drho_dr0 * dr0_dbeta - s_a * D) * du[0]
                                 + (c_b * s_a * inv_s_b) * du[3] + (-c_a) * du[4]
                                 + (-s_a * inv_s_b) * du[5];

            const Scalar tau_y = (-c_b * c_a * C - s_a * drho_dr0 * dr0_dbeta + c_a * D) * du[0]
                                 + (-c_a * c_b * inv_s_b) * du[3] + (-s_a) * du[4]
                                 + (c_a * inv_s_b) * du[5];

            const Scalar tau_z = (-drho_dr0 * dr0_dalpha) * du[0] + (-Scalar(1)) * du[3];

            // accumulate
            fi.x += f_x;
            fi.y += f_y;
            fi.z += f_z;

            ti.x += tau_x;
            ti.y += tau_y;
            ti.z += tau_z;

            pei += u;

            // Newton's third law for half neighbor list
            if (third_law)
                {
                h_force.data[j].x -= f_x;
                h_force.data[j].y -= f_y;
                h_force.data[j].z -= f_z;
                h_force.data[j].w += Scalar(0.5) * u;

                h_torque.data[j].x -= tau_x;
                h_torque.data[j].y -= tau_y;
                h_torque.data[j].z -= tau_z;
                }
            }

        h_force.data[i].x += fi.x;
        h_force.data[i].y += fi.y;
        h_force.data[i].z += fi.z;
        h_force.data[i].w += Scalar(0.5) * pei;

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
                if (domain.ndim() != 2 || domain.shape(0) != 5 || domain.shape(1) != 2)
                    {
                    throw std::runtime_error("domain must have shape (5,2).");
                    }

                if (terms.ndim() != 2 || terms.shape(1) != 6)
                    {
                    throw std::runtime_error("terms must have shape (Nterms,6).");
                    }

                const unsigned int Nterms = static_cast<unsigned int>(terms.shape(0));

                if (coeffs.ndim() != 1 || static_cast<unsigned int>(coeffs.shape(0)) != Nterms)
                    {
                    throw std::runtime_error("coeffs must have shape (Nterms,).");
                    }

                if (r0_data.ndim() != 5)
                    {
                    throw std::runtime_error("r0_data must be a 5D array.");
                    }

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
