// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file ChebyshevAnisotropicPairPotential.h
 * \brief Templated class for the Chebyshev anisotropic pair potential.
 *
 * The class is templated on a \c ShapeSymmetryT parameter that provides
 * a symmetry reduction of the angular coordinates.
 */

#ifndef AZPLUGINS_CHEBYSHEV_ANISOTROPIC_PAIR_POTENTIAL_H_
#define AZPLUGINS_CHEBYSHEV_ANISOTROPIC_PAIR_POTENTIAL_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <pybind11/pybind11.h>
#include <vector>

#include "hoomd/BoxDim.h"
#include "hoomd/ForceCompute.h"
#include "hoomd/GPUArray.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/VectorMath.h"
#include "hoomd/md/NeighborList.h"

#include "LinearInterpolator5D.h"
#include "ShapeSymmetry.h"

namespace hoomd
    {
namespace azplugins
    {

//! Scale a coordinate from [lo, hi] to the Chebyshev domain [-1, 1].
inline Scalar chebScale(Scalar x, Scalar lo, Scalar hi)
    {
    return (Scalar(2) * (x - lo) / (hi - lo)) - Scalar(1);
    }

//! Evaluate Chebyshev polynomials of the first kind and their derivatives
//! from degree 0 up to max_deg, using the three-term recurrence relation.
/*!
    T_0(x) = 1                       T'_0(x) = 0
    T_1(x) = x                       T'_1(x) = 1
    T_{n+1}(x) = 2x T_n - T_{n-1}   T'_{n+1}(x) = 2 T_n + 2x T'_n - T'_{n-1}


    \param x        Evaluation point in [-1, 1]
    \param max_deg  Highest polynomial degree to compute
    \param T        Output: T[n] = T_n(x)  for n = 0 .. max_deg  (size >= max_deg+1)
    \param dT       Output: dT[n] = T'_n(x) for n = 0 .. max_deg (size >= max_deg+1)
*/
inline void chebEvaluate(Scalar x, unsigned int max_deg, Scalar* T, Scalar* dT)
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

//! Chebyshev anisotropic pair potential, templated on a symmetry reducer.
/*!
 * \tparam ShapeSymmetryT A class providing static methods
 *         \c domain_lower(i) and \c domain_upper(i) returning the bounds
 *         of the redcued domain for the i-th angular coordinate (0..4),
 *         and a static \c reduce(theta, phi, alpha, beta, gamma) method that
 *         maps the angles into a fundamental domain and returns the applied
 *         rotation as a quaternion.  See \c ShapeSymmetry.h.
 */
template<class ShapeSymmetryT>
class PYBIND11_EXPORT ChebyshevAnisotropicPairPotential : public ForceCompute
    {
    public:
    static constexpr unsigned int num_coordinates = 6;
    static constexpr unsigned int num_angle_coordinates = num_coordinates - 1;

    ChebyshevAnisotropicPairPotential(std::shared_ptr<SystemDefinition> sysdef,
                                      std::shared_ptr<hoomd::md::NeighborList> nlist,
                                      const Scalar r_cut,
                                      const unsigned int* terms,
                                      const Scalar* coeffs,
                                      unsigned int Nterms,
                                      const Scalar* r0_data,
                                      const unsigned int* r0_shape);
    //! Destructor
    virtual ~ChebyshevAnisotropicPairPotential();

    //! Detach from the neighbor list (called when removing from simulation)
    virtual void notifyDetach();

    // Getters
    std::shared_ptr<hoomd::md::NeighborList> getNeighborList() const
        {
        return m_nlist;
        }

    /// Read-only cutoff radius
    Scalar getRCut() const
        {
        return m_r_cut;
        }

    /// Read-only number of Chebyshev terms
    unsigned int getNTerms() const
        {
        return m_Nterms;
        }

    protected:
    // member variables

    std::shared_ptr<hoomd::md::NeighborList> m_nlist; //!< Neighbor list

    Scalar m_r_cut;       //!< Cut-off distance in approximation domain
    Scalar m_nlist_r_cut; //!< Neighbor-list cutoff = ceil(max(r0) + r_cut)

    std::shared_ptr<GPUArray<Scalar>> m_r_cut_nlist; //!< r_cut matrix shared with nlist
    bool m_attached = true;                          //!< Whether attached to the simulation

    GPUArray<unsigned int> m_terms; //!< Chebyshev term list (Nterms x num_coordinates)
    GPUArray<Scalar> m_coeffs;      //!< Coefficients corresponding to each term
    unsigned int m_Nterms;          //!< Number of terms

    GPUArray<Scalar> m_r0_data;        //!< R0 data
    GPUArray<unsigned int> m_r0_shape; //!< Points per dimension to sample r0

    std::array<unsigned int, num_coordinates>
        m_max_deg; //!< Maximum Chebyshev degree per coordinate
    std::array<Scalar, num_coordinates>
        m_cheb_scale;                   //!< Chain-rule scale factors for each coordinate
    unsigned int m_max_deg_global;      //!< Maximum Chebyshev degree over all coordinates
    Index2D m_cheb_idx;                 //!< Indexer for flat Chebyshev scratch storage
    std::vector<Scalar> m_cheb_T_flat;  //!< Chebyshev polynomial scratch storage
    std::vector<Scalar> m_cheb_dT_flat; //!< Chebyshev derivative scratch storage

    void initializeChebyshevData();
    void computeForces(uint64_t timestep) override;
    };

// Constructor
template<class ShapeSymmetryT>
ChebyshevAnisotropicPairPotential<ShapeSymmetryT>::ChebyshevAnisotropicPairPotential(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<hoomd::md::NeighborList> nlist,
    const Scalar r_cut,
    const unsigned int* terms,
    const Scalar* coeffs,
    unsigned int Nterms,
    const Scalar* r0_data,
    const unsigned int* r0_shape)
    : ForceCompute(sysdef), m_nlist(nlist), m_r_cut(r_cut), m_Nterms(Nterms), m_max_deg_global(0),
      m_cheb_idx()
    {
        // terms: shape (Nterms, num_coordinates), stored flat
        {
        GPUArray<unsigned int> terms_arr(static_cast<size_t>(Nterms) * num_coordinates,
                                         m_exec_conf);
        m_terms.swap(terms_arr);

        ArrayHandle<unsigned int> h_terms(m_terms, access_location::host, access_mode::overwrite);
        std::copy(terms, terms + static_cast<size_t>(Nterms) * num_coordinates, h_terms.data);
        }

        // coeffs: shape (Nterms,)
        {
        GPUArray<Scalar> coeffs_arr(Nterms, m_exec_conf);
        m_coeffs.swap(coeffs_arr);

        ArrayHandle<Scalar> h_coeffs(m_coeffs, access_location::host, access_mode::overwrite);
        std::copy(coeffs, coeffs + Nterms, h_coeffs.data);
        }

        // r0_shape: length num_angle_coordinates
        {
        GPUArray<unsigned int> shape_arr(num_angle_coordinates, m_exec_conf);
        m_r0_shape.swap(shape_arr);

        ArrayHandle<unsigned int> h_shape(m_r0_shape,
                                          access_location::host,
                                          access_mode::overwrite);
        std::copy(r0_shape, r0_shape + num_angle_coordinates, h_shape.data);
        }

    // r0_data: flat array, length = product(r0_shape)
    unsigned int n_r0 = 1;
    for (unsigned int d = 0; d < num_angle_coordinates; ++d)
        {
        n_r0 *= r0_shape[d];
        }

        {
        GPUArray<Scalar> r0_arr(n_r0, m_exec_conf);
        m_r0_data.swap(r0_arr);

        ArrayHandle<Scalar> h_r0(m_r0_data, access_location::host, access_mode::overwrite);
        std::copy(r0_data, r0_data + n_r0, h_r0.data);
        }

    // neighbor list subscriber
    Scalar max_r0 = *std::max_element(r0_data, r0_data + n_r0);
    m_nlist_r_cut = std::ceil(max_r0 + m_r_cut);

    const Index2D typpair_idx(m_pdata->getNTypes());
    m_r_cut_nlist = std::make_shared<GPUArray<Scalar>>(typpair_idx.getNumElements(), m_exec_conf);
        {
        ArrayHandle<Scalar> h_r_cut_nlist(*m_r_cut_nlist,
                                          access_location::host,
                                          access_mode::overwrite);
        std::fill(h_r_cut_nlist.data,
                  h_r_cut_nlist.data + typpair_idx.getNumElements(),
                  m_nlist_r_cut);
        }
    m_nlist->addRCutMatrix(m_r_cut_nlist);
    m_nlist->notifyRCutMatrixChange();

    initializeChebyshevData();
    }

// Destructor
template<class ShapeSymmetryT>
ChebyshevAnisotropicPairPotential<ShapeSymmetryT>::~ChebyshevAnisotropicPairPotential()
    {
    if (m_attached)
        {
        m_nlist->removeRCutMatrix(m_r_cut_nlist);
        }
    }

// notifyDetach
template<class ShapeSymmetryT>
void ChebyshevAnisotropicPairPotential<ShapeSymmetryT>::notifyDetach()
    {
    if (m_attached)
        {
        m_nlist->removeRCutMatrix(m_r_cut_nlist);
        }
    m_attached = false;
    }

// initializeChebyshevData
template<class ShapeSymmetryT>
void ChebyshevAnisotropicPairPotential<ShapeSymmetryT>::initializeChebyshevData()
    {
    m_max_deg.fill(0);

    ArrayHandle<unsigned int> h_terms(m_terms, access_location::host, access_mode::read);
    for (unsigned int t = 0; t < m_Nterms; ++t)
        {
        for (unsigned int c = 0; c < num_coordinates; ++c)
            {
            const unsigned int deg = h_terms.data[t * num_coordinates + c];
            if (deg > m_max_deg[c])
                {
                m_max_deg[c] = deg;
                }
            }
        }

    m_cheb_scale[0] = Scalar(2);
    for (unsigned int d = 0; d < num_angle_coordinates; ++d)
        {
        m_cheb_scale[d + 1]
            = Scalar(2) / (ShapeSymmetryT::domain_upper(d) - ShapeSymmetryT::domain_lower(d));
        }

    m_max_deg_global = 0;
    for (unsigned int c = 0; c < num_coordinates; ++c)
        {
        if (m_max_deg[c] > m_max_deg_global)
            {
            m_max_deg_global = m_max_deg[c];
            }
        }

    m_cheb_idx = Index2D(m_max_deg_global + 1, num_coordinates);
    m_cheb_T_flat.resize(m_cheb_idx.getNumElements());
    m_cheb_dT_flat.resize(m_cheb_idx.getNumElements());
    }

// computeForces
template<class ShapeSymmetryT>
void ChebyshevAnisotropicPairPotential<ShapeSymmetryT>::computeForces(uint64_t timestep)
    {
    // start by updating the neighborlist
    m_nlist->compute(timestep);

    // check neighbor list storage mode
    const bool use_third_law = (m_nlist->getStorageMode() == hoomd::md::NeighborList::half);
    const unsigned int N_local = m_pdata->getN();
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
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_r0_data(m_r0_data, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_r0_shape(m_r0_shape, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_terms(m_terms, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_coeffs(m_coeffs, access_location::host, access_mode::read);

    const BoxDim box = m_pdata->getGlobalBox();
    const Scalar nlist_rcutsq = m_nlist_r_cut * m_nlist_r_cut;
    const Scalar fd_step = Scalar(1.0e-6);

    Scalar domain_lower[num_angle_coordinates];
    Scalar domain_upper[num_angle_coordinates];
    for (unsigned int d = 0; d < num_angle_coordinates; ++d)
        {
        domain_lower[d] = ShapeSymmetryT::domain_lower(d);
        domain_upper[d] = ShapeSymmetryT::domain_upper(d);
        }

    LinearInterpolator5D<Scalar> interp(h_r0_data.data,
                                        h_r0_shape.data,
                                        domain_lower,
                                        domain_upper);

    m_force.zeroFill();
    m_torque.zeroFill();

    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_torque(m_torque, access_location::host, access_mode::readwrite);

    const unsigned int N = m_pdata->getN();

    //! Euler-angle singularity tolerance for the alpha/gamma extraction.
    const Scalar euler_singularity_tol = Scalar(1e-7);

    const Scalar phi_eval_min = domain_lower[1];
    const Scalar phi_eval_max = domain_upper[1];
    const Scalar beta_eval_min = domain_lower[3];
    const Scalar beta_eval_max = domain_upper[3];

    for (unsigned int i = 0; i < N; ++i)
        {
        // Per-pair position and orientation are loaded inside the loop after
        // sorting by tag (on full lists)
        // Initialize particle force, torque, and energy
        Scalar3 fi = make_scalar3(0, 0, 0);
        Scalar3 ti = make_scalar3(0, 0, 0);
        Scalar pei = Scalar(0);

        const size_t myHead = h_head_list.data[i];
        const unsigned int size = (unsigned int)h_n_neigh.data[i];

        for (unsigned int k = 0; k < size; ++k)
            {
            // Access the index
            const unsigned int j = h_nlist.data[myHead + k];
            assert(j < m_pdata->getN() + m_pdata->getNGhosts());

            // Sort the pair by tag
            unsigned int eval_a = i;
            unsigned int eval_b = j;
            bool i_is_eval_a = true;
            if (!use_third_law)
                {
                const unsigned int tag_i = h_tag.data[i];
                const unsigned int tag_j = h_tag.data[j];
                if (tag_j < tag_i)
                    {
                    eval_a = j;
                    eval_b = i;
                    i_is_eval_a = false;
                    }
                }

            const Scalar3 pos_a
                = make_scalar3(h_pos.data[eval_a].x, h_pos.data[eval_a].y, h_pos.data[eval_a].z);
            const Scalar3 pos_b
                = make_scalar3(h_pos.data[eval_b].x, h_pos.data[eval_b].y, h_pos.data[eval_b].z);
            Scalar3 dx = pos_a - pos_b;
            // Apply periodic boundary conditions
            dx = box.minImage(dx);
            // Neighbor-list cutoff check (center-center distance).
            const Scalar rsq = dot(dx, dx);
            if (rsq > nlist_rcutsq)
                {
                continue;
                }

            const quat<Scalar> q_a(h_orientation.data[eval_a]);
            const quat<Scalar> q_b(h_orientation.data[eval_b]);
            const quat<Scalar> q_a_conj = conj(q_a);
            // dx is in lab frame, so rotate dx by conj(q_a)
            const vec3<Scalar> dx_body = rotate(q_a_conj, vec3<Scalar>(dx));
            // Relative orientation of eval_b with respect to eval_a:
            //     q_rel = conj(q_a) * q_b
            // ref:
            // https://www.mathworks.com/help/fusion/ug/rotations-orientation-and-quaternions.html
            const quat<Scalar> q_rel = q_a_conj * q_b;

            // Convert position to spherical coordinates
            // Skip overlapping particles.
            const Scalar r = fast::sqrt(dot(dx_body, dx_body));
            if (r < Scalar(1e-12))
                {
                continue;
                }

            Scalar theta = std::atan2(dx_body.y, dx_body.x);
            if (theta < Scalar(0))
                theta += Scalar(2.0) * M_PI;

            Scalar cosphi = dx_body.z / r;
            if (cosphi < Scalar(-1))
                cosphi = Scalar(-1);
            else if (cosphi > Scalar(1))
                cosphi = Scalar(1);
            Scalar phi = slow::acos(cosphi);

            // Build rotation matrix from the relative quaternion and extract
            // ZXZ Euler angles (alpha, beta, gamma)
            const rotmat3<Scalar> R(q_rel);

            Scalar alpha = Scalar(0);
            Scalar beta = Scalar(0);
            Scalar gamma = Scalar(0);

            Scalar clamped_r22 = R.row2.z;
            if (clamped_r22 < Scalar(-1))
                clamped_r22 = Scalar(-1);
            else if (clamped_r22 > Scalar(1))
                clamped_r22 = Scalar(1);
            beta = slow::acos(clamped_r22);

            if (beta > euler_singularity_tol && beta < Scalar(M_PI) - euler_singularity_tol)
                {
                alpha = std::atan2(R.row0.z, -R.row1.z);
                gamma = std::atan2(R.row2.x, R.row2.y);
                if (alpha < Scalar(0))
                    alpha += Scalar(2) * M_PI;
                }
            else
                {
                alpha = Scalar(0);
                gamma
                    = std::atan2((beta <= euler_singularity_tol) ? R.row0.y : -R.row0.y, R.row0.x);
                }

            if (gamma < Scalar(0))
                gamma += Scalar(2) * M_PI;

            // Symmetry reduction
            // The symmetry evaluator maps (theta, phi, alpha, beta, gamma)
            // into the reduced domain and returns the cumulative
            // quaternion rotation that was applied.  We keep the transformation so
            // that forces and torques can be rotated back to the
            // original frame at the end.
            const quat<Scalar> sym_transformation
                = ShapeSymmetryT::reduce(theta, phi, alpha, beta, gamma);

            // check phi and beta away from 0 and pi to avoid 1/sin(beta or phi)
            // singularity in the Jacobian (used the same threshold as beta).
            if (beta < beta_eval_min)
                beta = beta_eval_min;
            else if (beta > beta_eval_max)
                beta = beta_eval_max;

            if (phi < phi_eval_min)
                phi = phi_eval_min;
            else if (phi > phi_eval_max)
                phi = phi_eval_max;

            // Compute r0 and all 5 derivatives
            Scalar r0;
            Scalar dr0[num_angle_coordinates];
            interp.valueAndDerivatives(theta, phi, alpha, beta, gamma, fd_step, r0, dr0);
            const Scalar dr0_dtheta = dr0[0];
            const Scalar dr0_dphi = dr0[1];
            const Scalar dr0_dalpha = dr0[2];
            const Scalar dr0_dbeta = dr0[3];
            const Scalar dr0_dgamma = dr0[4];

            // Compute rho
            const Scalar inv_r = Scalar(1) / r;
            const Scalar inv_r0 = Scalar(1) / r0;
            const Scalar inv_r0_rcut = Scalar(1) / (r0 + m_r_cut);
            const Scalar rho_denom = inv_r0_rcut - inv_r0;
            const Scalar rho_num = inv_r - inv_r0;
            Scalar rho = rho_num / rho_denom;

            if (rho > Scalar(1))
                continue;

            // Save raw rho for energy extrapolation if rho < 0
            const Scalar rho_energy = rho;
            if (rho < Scalar(0))
                rho = Scalar(0);

            // Compute drho/dr and drho/dr0
            const Scalar inv_r_sq = inv_r * inv_r;
            const Scalar inv_r0_sq = inv_r0 * inv_r0;
            const Scalar inv_r0_rcut_sq = inv_r0_rcut * inv_r0_rcut;
            const Scalar rho_denom_sq = rho_denom * rho_denom;

            const Scalar drho_dr = -inv_r_sq / rho_denom;
            const Scalar drho_dr0
                = (inv_r0_sq * rho_denom - rho_num * (inv_r0_sq - inv_r0_rcut_sq)) / rho_denom_sq;

            // Chebyshev evaluation: scale each coordinate to [-1,1]
            // and evaluate polynomials + derivatives up to max degree.
            chebEvaluate(chebScale(rho, Scalar(0), Scalar(1)),
                         m_max_deg[0],
                         m_cheb_T_flat.data() + m_cheb_idx(0, 0),
                         m_cheb_dT_flat.data() + m_cheb_idx(0, 0));

            const Scalar ang_coords[num_angle_coordinates] = {theta, phi, alpha, beta, gamma};
            for (unsigned int c = 0; c < 5; ++c)
                {
                chebEvaluate(chebScale(ang_coords[c], domain_lower[c], domain_upper[c]),
                             m_max_deg[c + 1],
                             m_cheb_T_flat.data() + m_cheb_idx(0, c + 1),
                             m_cheb_dT_flat.data() + m_cheb_idx(0, c + 1));
                }

            // Evaluate u and du/d(coord_k)
            Scalar u = Scalar(0);
            Scalar du[num_coordinates]
                = {Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0)};

            for (unsigned int t = 0; t < m_Nterms; ++t)
                {
                const unsigned int* degs = h_terms.data + num_coordinates * t;
                const Scalar coeff = h_coeffs.data[t];

                Scalar T_vals[num_coordinates];
                Scalar dT_vals[num_coordinates];
                for (unsigned int c = 0; c < num_coordinates; ++c)
                    {
                    T_vals[c] = m_cheb_T_flat[m_cheb_idx(degs[c], c)];
                    dT_vals[c] = m_cheb_dT_flat[m_cheb_idx(degs[c], c)];
                    }

                Scalar prefix[num_coordinates + 1];
                prefix[0] = Scalar(1);
                for (unsigned int c = 0; c < num_coordinates; ++c)
                    prefix[c + 1] = prefix[c] * T_vals[c];

                Scalar suffix[num_coordinates + 1];
                suffix[num_coordinates] = Scalar(1);
                for (int c = static_cast<int>(num_coordinates) - 1; c >= 0; --c)
                    suffix[c] = suffix[c + 1] * T_vals[c];

                u += coeff * prefix[num_coordinates];

                for (unsigned int c = 0; c < num_coordinates; ++c)
                    du[c] += coeff * dT_vals[c] * m_cheb_scale[c] * prefix[c] * suffix[c + 1];
                }

            // Linear extrapolation for energy when rho < 0
            const Scalar u_energy = (rho_energy < Scalar(0)) ? (u + rho_energy * du[0]) : u;

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

            const Scalar A = drho_dr0 * dr0_dtheta * inv_r_s_ph;
            const Scalar B = drho_dr0 * dr0_dphi * inv_r;
            const Scalar C = drho_dr0 * dr0_dalpha * inv_s_b;
            const Scalar D = drho_dr0 * dr0_dgamma * inv_s_b;

            const Scalar f_x_red = (-c_th * s_ph * drho_dr + s_th * A - c_th * c_ph * B) * du[0]
                                   + (s_th * inv_r_s_ph) * du[1] + (-c_th * c_ph * inv_r) * du[2];

            const Scalar f_y_red = (-s_th * s_ph * drho_dr - c_th * A - s_th * c_ph * B) * du[0]
                                   + (-c_th * inv_r_s_ph) * du[1] + (-s_th * c_ph * inv_r) * du[2];

            const Scalar f_z_red = (-c_ph * drho_dr + s_ph * B) * du[0] + (s_ph * inv_r) * du[2];

            const Scalar tau_x_red = (c_b * s_a * C - c_a * drho_dr0 * dr0_dbeta - s_a * D) * du[0]
                                     + (c_b * s_a * inv_s_b) * du[3] + (-c_a) * du[4]
                                     + (-s_a * inv_s_b) * du[5];

            const Scalar tau_y_red = (-c_b * c_a * C - s_a * drho_dr0 * dr0_dbeta + c_a * D) * du[0]
                                     + (-c_a * c_b * inv_s_b) * du[3] + (-s_a) * du[4]
                                     + (c_a * inv_s_b) * du[5];

            const Scalar tau_z_red = (-drho_dr0 * dr0_dalpha) * du[0] + (-Scalar(1)) * du[3];

            // Rotate back to original frame
            const quat<Scalar> sym_inv = conj(sym_transformation);
            const vec3<Scalar> f_red(f_x_red, f_y_red, f_z_red);
            const vec3<Scalar> tau_red(tau_x_red, tau_y_red, tau_z_red);
            const vec3<Scalar> f_lab = rotate(sym_inv, f_red);
            const vec3<Scalar> tau_lab = rotate(sym_inv, tau_red);

            const Scalar f_x = f_lab.x;
            const Scalar f_y = f_lab.y;
            const Scalar f_z = f_lab.z;
            const Scalar tau_x = tau_lab.x;
            const Scalar tau_y = tau_lab.y;
            const Scalar tau_z = tau_lab.z;

            // Writeback
            const Scalar sign = i_is_eval_a ? Scalar(1) : Scalar(-1);

            fi.x += sign * f_x;
            fi.y += sign * f_y;
            fi.z += sign * f_z;

            ti.x += sign * tau_x;
            ti.y += sign * tau_y;
            ti.z += sign * tau_z;

            pei += u_energy;

            // Newton's third law for half neighbor list
            if (use_third_law && j < N_local)
                {
                h_force.data[j].x -= f_x;
                h_force.data[j].y -= f_y;
                h_force.data[j].z -= f_z;
                h_force.data[j].w += Scalar(0.5) * u_energy;

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

//! Export one subclass of ChebyshevAnisotropicPairPotential to python.
/*!
 * \param m    pybind11 module.
 * \param name Name the class should have in the python module (must be
 *             unique per symmetry).
 * \tparam ShapeSymmetryT Symmetry evaluator type.
 */
template<class ShapeSymmetryT>
void export_ChebyshevAnisotropicPairPotential(pybind11::module& m, const std::string& name)
    {
    namespace py = pybind11;
    using NL = hoomd::md::NeighborList;
    using Pot = ChebyshevAnisotropicPairPotential<ShapeSymmetryT>;

    py::class_<Pot, ForceCompute, std::shared_ptr<Pot>>(m, name.c_str())
        .def(py::init(
            [](std::shared_ptr<SystemDefinition> sysdef,
               std::shared_ptr<NL> nlist,
               Scalar r_cut,
               py::array_t<unsigned int, py::array::c_style | py::array::forcecast> terms,
               py::array_t<Scalar, py::array::c_style | py::array::forcecast> coeffs,
               py::array_t<Scalar, py::array::c_style | py::array::forcecast> r0_data)
            {
                // Terms must be (Nterms,6)
                if (terms.ndim() != 2 || terms.shape(1) != 6)
                    {
                    throw std::runtime_error("terms must have shape (Nterms,6).");
                    }

                const unsigned int Nterms = static_cast<unsigned int>(terms.shape(0));

                // Coeffs must be (Nterms,)
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

                return std::make_shared<Pot>(sysdef,
                                             nlist,
                                             r_cut,
                                             terms.data(),
                                             coeffs.data(),
                                             Nterms,
                                             r0_data.data(),
                                             r0_shape.data());
            }))
        .def_property_readonly("r_cut", &Pot::getRCut)
        .def_property_readonly("num_terms", &Pot::getNTerms);
    }

    } // end namespace detail
    } // namespace azplugins
    } // namespace hoomd

#endif // AZPLUGINS_CHEBYSHEV_ANISOTROPIC_PAIR_POTENTIAL_H_
