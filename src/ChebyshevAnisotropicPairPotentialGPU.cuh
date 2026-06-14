// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

// File modified from HOOMD-blue
// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/VectorMath.h"

#include "LinearInterpolator5D.h"
#include "ShapeSymmetry.h"

#include <cstddef>

// CUB is only needed for block-wide reduction.
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
#endif

/*! \file ChebyshevAnisotropicPairPotentialGPU.cuh
    \brief Defines templated GPU kernel code for the Chebyshev anisotropic pair potential.
*/

#ifndef AZPLUGINS_CHEBYSHEV_ANISOTROPIC_PAIR_POTENTIAL_GPU_CUH_
#define AZPLUGINS_CHEBYSHEV_ANISOTROPIC_PAIR_POTENTIAL_GPU_CUH_

namespace hoomd
    {
namespace azplugins
    {
namespace kernel
    {

//! Maximum Chebyshev degree any single coordinate can have.
static constexpr unsigned int chebyshev_max_degree = 32;

//! Wraps arguments to gpu_compute_chebyshev_pair_forces
struct chebyshev_pair_args_t
    {
    //! Construct a chebyshev_pair_args_t
    chebyshev_pair_args_t(Scalar4* _d_force,
                          Scalar4* _d_torque,
                          const unsigned int _N,
                          const Scalar4* _d_pos,
                          const Scalar4* _d_orientation,
                          const unsigned int* _d_tag,
                          const BoxDim& _box,
                          const unsigned int* _d_n_neigh,
                          const unsigned int* _d_nlist,
                          const size_t* _d_head_list,
                          const Scalar* _d_r0_data,
                          const unsigned int* _d_r0_shape,
                          const unsigned int* _d_terms,
                          const Scalar* _d_coeffs,
                          const unsigned int _Nterms,
                          const Scalar _r_cut,
                          const Scalar _nlist_r_cut,
                          const Scalar* _domain_lower,
                          const Scalar* _domain_upper,
                          const unsigned int _block_size)
        : d_force(_d_force), d_torque(_d_torque), N(_N), d_pos(_d_pos),
          d_orientation(_d_orientation), d_tag(_d_tag), box(_box), d_n_neigh(_d_n_neigh),
          d_nlist(_d_nlist), d_head_list(_d_head_list), d_r0_data(_d_r0_data),
          d_r0_shape(_d_r0_shape), d_terms(_d_terms), d_coeffs(_d_coeffs), Nterms(_Nterms),
          r_cut(_r_cut), nlist_r_cut(_nlist_r_cut), block_size(_block_size)
        {
        for (unsigned int d = 0; d < 5; ++d)
            {
            domain_lower[d] = _domain_lower[d];
            domain_upper[d] = _domain_upper[d];
            }
        };

    Scalar4* d_force;             //!< Force to write out
    Scalar4* d_torque;            //!< Troque to write out
    const unsigned int N;         //!< Number of particles
    const Scalar4* d_pos;         //!< particle positions
    const Scalar4* d_orientation; //!< particle orientations
    const unsigned int* d_tag;    //!< tags for pair sorting
    const BoxDim box;             //!< Simulation box in GPU format
    const unsigned int*
        d_n_neigh;               //!< Device array listing the number of neighbors on each particle
    const unsigned int* d_nlist; //!< Device array listing the neighbors of each particle
    const size_t* d_head_list;   //!< Head list indexes for accessing d_nlist
    const Scalar* d_r0_data;     //!< 5-D r0 grid
    const unsigned int* d_r0_shape; //!< Shape of r0
    const unsigned int* d_terms;    //!< (Nterms x 6) Chebyshev term table
    const Scalar* d_coeffs;         //!< Chebyshev coefficients (length Nterms)
    const unsigned int Nterms;      //!< Number of Chebyshev terms
    const Scalar r_cut;             //!< Surface cutoff in approximation domain
    const Scalar nlist_r_cut;       //!< Center-center neighbour-list cutoff
    Scalar domain_lower[5];         //!< Domain lower bounds
    Scalar domain_upper[5];         //!< Domain upper bounds
    const unsigned int block_size;  //!< Block size to execute
    };

//! Driver function for each symmetry.
template<class ShapeSymmetryT>
__attribute__((visibility("default"))) hipError_t
gpu_compute_chebyshev_pair_forces(const chebyshev_pair_args_t& args);

#ifdef __HIPCC__
//! Evaluate Chebyshev polynomials of the first kind and their derivatives.
__device__ inline void chebyshev_eval_device(Scalar x, unsigned int max_deg, Scalar* T, Scalar* dT)
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

//! Scale a coordinate from [lo, hi] to the Chebyshev domain [-1, 1].
__device__ inline Scalar chebyshev_scale_device(Scalar x, Scalar lo, Scalar hi)
    {
    return (Scalar(2) * (x - lo) / (hi - lo)) - Scalar(1);
    }

template<class ShapeSymmetryT, unsigned int BLOCK_SIZE>
__global__ void gpu_compute_chebyshev_pair_forces_kernel(chebyshev_pair_args_t args)
    {
    // one block per particle.
    // each block is responsible for just one particle i.
    // for each neighbor j, the first 6 threads fill the
    // univariate Chebyshev tables for the 6 coordinates into shared-memory,
    // then every thread evaluates a chunk of the term list (reading
    // from the shared memory)
    const unsigned int i = blockIdx.x;
    if (i >= args.N)
        return;

    const unsigned int tid = threadIdx.x;
    const unsigned int nthreads = blockDim.x;

    const Scalar nlist_rcutsq = args.nlist_r_cut * args.nlist_r_cut;
    const Scalar fd_step = Scalar(1.0e-6);
    const Scalar euler_singularity_tol = Scalar(1e-7);

    constexpr unsigned int n_coords = 6;
    constexpr unsigned int n_angles = 5;

    LinearInterpolator5D<Scalar> interp(args.d_r0_data,
                                        args.d_r0_shape,
                                        args.domain_lower,
                                        args.domain_upper);

    // dynamic shared memory
    // univariate T, univariate dT, & CUB reduce temp storage
    extern __shared__ unsigned char smem[];
    const unsigned int stride = chebyshev_max_degree + 1;
    Scalar* s_T = reinterpret_cast<Scalar*>(smem);
    Scalar* s_dT = s_T + n_coords * stride;
    // CUB temp storage
    void* s_cub = reinterpret_cast<void*>(s_dT + n_coords * stride);

    // CUB block-reduce
    typedef hipcub::BlockReduce<Scalar, BLOCK_SIZE> BlockReduceT;

    unsigned int max_deg[n_coords] = {0, 0, 0, 0, 0, 0};
    for (unsigned int t = 0; t < args.Nterms; ++t)
        {
        for (unsigned int c = 0; c < n_coords; ++c)
            {
            const unsigned int deg = args.d_terms[t * n_coords + c];
            if (deg > max_deg[c])
                max_deg[c] = deg;
            }
        }

    // Chain-rule scale factors d(x_scaled)/d(x)
    Scalar cheb_scale[n_coords];
    cheb_scale[0] = Scalar(2);
    for (unsigned int d = 0; d < n_angles; ++d)
        cheb_scale[d + 1] = Scalar(2) / (args.domain_upper[d] - args.domain_lower[d]);

    // Half list
    typename BlockReduceT::TempStorage& cub_temp
        = *reinterpret_cast<typename BlockReduceT::TempStorage*>(s_cub);

    const size_t myHead = args.d_head_list[i];
    const unsigned int size = args.d_n_neigh[i];
    const unsigned int tag_i = args.d_tag[i];

    Scalar3 fi = make_scalar3(0, 0, 0);
    Scalar3 ti = make_scalar3(0, 0, 0);
    Scalar ei = Scalar(0);

    for (unsigned int k = 0; k < size; ++k)
        {
        const unsigned int j = args.d_nlist[myHead + k];

        // Skip pairs owned by the other block.
        const unsigned int tag_j = args.d_tag[j];
        if (tag_i > tag_j)
            continue;

        const unsigned int eval_a = i;
        const unsigned int eval_b = j;

        const Scalar3 pos_a
            = make_scalar3(args.d_pos[eval_a].x, args.d_pos[eval_a].y, args.d_pos[eval_a].z);
        const Scalar3 pos_b
            = make_scalar3(args.d_pos[eval_b].x, args.d_pos[eval_b].y, args.d_pos[eval_b].z);
        Scalar3 dx = pos_a - pos_b;
        dx = args.box.minImage(dx);

        const Scalar rsq = dot(dx, dx);
        if (rsq > nlist_rcutsq)
            continue;

        const quat<Scalar> q_a(args.d_orientation[eval_a]);
        const quat<Scalar> q_b(args.d_orientation[eval_b]);
        const quat<Scalar> q_a_conj = conj(q_a);
        const vec3<Scalar> dx_body = rotate(q_a_conj, vec3<Scalar>(dx));
        const quat<Scalar> q_rel = q_a_conj * q_b;

        const Scalar r = fast::sqrt(dot(dx_body, dx_body));
        if (r < Scalar(1e-12))
            continue;

        Scalar theta = std::atan2(dx_body.y, dx_body.x);
        if (theta < Scalar(0))
            theta += Scalar(2.0) * Scalar(M_PI);

        Scalar cosphi = dx_body.z / r;
        if (cosphi < Scalar(-1))
            cosphi = Scalar(-1);
        else if (cosphi > Scalar(1))
            cosphi = Scalar(1);
        Scalar phi = slow::acos(cosphi);

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
                alpha += Scalar(2) * Scalar(M_PI);
            }
        else
            {
            alpha = Scalar(0);
            gamma = std::atan2((beta <= euler_singularity_tol) ? R.row0.y : -R.row0.y, R.row0.x);
            }
        if (gamma < Scalar(0))
            gamma += Scalar(2) * Scalar(M_PI);

        const quat<Scalar> sym_transformation
            = ShapeSymmetryT::reduce(theta, phi, alpha, beta, gamma);

        const Scalar phi_lo = args.domain_lower[1];
        const Scalar beta_lo = args.domain_lower[3];
        if (beta < beta_lo)
            beta = beta_lo;
        else if (beta > Scalar(M_PI) - beta_lo)
            beta = Scalar(M_PI) - beta_lo;
        if (phi < phi_lo)
            phi = phi_lo;
        else if (phi > Scalar(M_PI) - phi_lo)
            phi = Scalar(M_PI) - phi_lo;

        Scalar r0;
        Scalar dr0[n_angles];
        interp.valueAndDerivatives(theta, phi, alpha, beta, gamma, fd_step, r0, dr0);
        const Scalar dr0_dtheta = dr0[0];
        const Scalar dr0_dphi = dr0[1];
        const Scalar dr0_dalpha = dr0[2];
        const Scalar dr0_dbeta = dr0[3];
        const Scalar dr0_dgamma = dr0[4];

        const Scalar inv_r = Scalar(1) / r;
        const Scalar inv_r0 = Scalar(1) / r0;
        const Scalar inv_r0_rcut = Scalar(1) / (r0 + args.r_cut);
        const Scalar rho_denom = inv_r0_rcut - inv_r0;
        const Scalar rho_num = inv_r - inv_r0;
        Scalar rho = rho_num / rho_denom;

        if (rho > Scalar(1))
            continue;

        const Scalar rho_energy = rho;
        if (rho < Scalar(0))
            rho = Scalar(0);

        const Scalar inv_r_sq = inv_r * inv_r;
        const Scalar inv_r0_sq = inv_r0 * inv_r0;
        const Scalar inv_r0_rcut_sq = inv_r0_rcut * inv_r0_rcut;
        const Scalar rho_denom_sq = rho_denom * rho_denom;

        const Scalar drho_dr = -inv_r_sq / rho_denom;
        const Scalar drho_dr0
            = (inv_r0_sq * rho_denom - rho_num * (inv_r0_sq - inv_r0_rcut_sq)) / rho_denom_sq;

        // populate univariate Chebyshev tables in shared memory
        if (tid < n_coords)
            {
            const unsigned int c = tid;
            Scalar x_scaled;
            if (c == 0)
                x_scaled = chebyshev_scale_device(rho, Scalar(0), Scalar(1));
            else
                {
                const Scalar ang[n_angles] = {theta, phi, alpha, beta, gamma};
                x_scaled = chebyshev_scale_device(ang[c - 1],
                                                  args.domain_lower[c - 1],
                                                  args.domain_upper[c - 1]);
                }
            chebyshev_eval_device(x_scaled, max_deg[c], s_T + c * stride, s_dT + c * stride);
            }
        __syncthreads();

        // each thread evaluates a chunk of the term list
        Scalar u = Scalar(0);
        Scalar du[n_coords] = {Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0)};
        for (unsigned int t = tid; t < args.Nterms; t += nthreads)
            {
            const unsigned int* degs = args.d_terms + n_coords * t;
            const Scalar coeff = args.d_coeffs[t];

            Scalar T_vals[n_coords];
            Scalar dT_vals[n_coords];
            for (unsigned int c = 0; c < n_coords; ++c)
                {
                T_vals[c] = s_T[c * stride + degs[c]];
                dT_vals[c] = s_dT[c * stride + degs[c]];
                }

            Scalar prefix[n_coords + 1];
            prefix[0] = Scalar(1);
            for (unsigned int c = 0; c < n_coords; ++c)
                prefix[c + 1] = prefix[c] * T_vals[c];

            Scalar suffix[n_coords + 1];
            suffix[n_coords] = Scalar(1);
            for (int c = static_cast<int>(n_coords) - 1; c >= 0; --c)
                suffix[c] = suffix[c + 1] * T_vals[c];

            u += coeff * prefix[n_coords];
            for (unsigned int c = 0; c < n_coords; ++c)
                du[c] += coeff * dT_vals[c] * cheb_scale[c] * prefix[c] * suffix[c + 1];
            }
        __syncthreads();

        const Scalar u_energy = (rho_energy < Scalar(0)) ? (u + rho_energy * du[0]) : u;

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

        const quat<Scalar> sym_inv = conj(sym_transformation);
        const vec3<Scalar> f_lab = rotate(sym_inv, vec3<Scalar>(f_x_red, f_y_red, f_z_red));
        const vec3<Scalar> tau_lab = rotate(sym_inv, vec3<Scalar>(tau_x_red, tau_y_red, tau_z_red));

        // block reduction (one pair total).
        Scalar fx = BlockReduceT(cub_temp).Sum(f_lab.x);
        __syncthreads();
        Scalar fy = BlockReduceT(cub_temp).Sum(f_lab.y);
        __syncthreads();
        Scalar fz = BlockReduceT(cub_temp).Sum(f_lab.z);
        __syncthreads();
        Scalar tx = BlockReduceT(cub_temp).Sum(tau_lab.x);
        __syncthreads();
        Scalar ty = BlockReduceT(cub_temp).Sum(tau_lab.y);
        __syncthreads();
        Scalar tz = BlockReduceT(cub_temp).Sum(tau_lab.z);
        __syncthreads();
        Scalar ue = BlockReduceT(cub_temp).Sum(u_energy);
        __syncthreads();

        // i is accumulated locally.
        // only j need a per-pair atomicAdd because
        // it is owned by another block.
        if (tid == 0)
            {
            fi.x += fx;
            fi.y += fy;
            fi.z += fz;
            ti.x += tx;
            ti.y += ty;
            ti.z += tz;
            ei += Scalar(0.5) * ue;

            atomicAdd(&args.d_force[j].x, -fx);
            atomicAdd(&args.d_force[j].y, -fy);
            atomicAdd(&args.d_force[j].z, -fz);
            atomicAdd(&args.d_force[j].w, Scalar(0.5) * ue);
            atomicAdd(&args.d_torque[j].x, -tx);
            atomicAdd(&args.d_torque[j].y, -ty);
            atomicAdd(&args.d_torque[j].z, -tz);
            }
        }

    // One atomic set for i
    if (tid == 0)
        {
        atomicAdd(&args.d_force[i].x, fi.x);
        atomicAdd(&args.d_force[i].y, fi.y);
        atomicAdd(&args.d_force[i].z, fi.z);
        atomicAdd(&args.d_force[i].w, ei);
        atomicAdd(&args.d_torque[i].x, ti.x);
        atomicAdd(&args.d_torque[i].y, ti.y);
        atomicAdd(&args.d_torque[i].z, ti.z);
        }
    }

template<class ShapeSymmetryT, unsigned int BLOCK_SIZE>
inline void launch_chebyshev_kernel(const chebyshev_pair_args_t& args)
    {
    constexpr unsigned int n_coords = 6;
    const unsigned int stride = chebyshev_max_degree + 1;
    const size_t univariate_bytes = static_cast<size_t>(2) * n_coords * stride * sizeof(Scalar);

    typedef hipcub::BlockReduce<Scalar, BLOCK_SIZE> BlockReduceT;
    const size_t cub_bytes = sizeof(typename BlockReduceT::TempStorage);

    const size_t shared_bytes = univariate_bytes + cub_bytes;

    hipLaunchKernelGGL((gpu_compute_chebyshev_pair_forces_kernel<ShapeSymmetryT, BLOCK_SIZE>),
                       dim3(args.N),
                       dim3(BLOCK_SIZE),
                       shared_bytes,
                       0,
                       args);
    }

//! Driver function for each symmetry.
template<class ShapeSymmetryT>
__attribute__((visibility("default"))) hipError_t
gpu_compute_chebyshev_pair_forces(const chebyshev_pair_args_t& args)
    {
    if (args.N == 0)
        return hipSuccess;

    // tune later with a profiler.
    const unsigned int requested = args.block_size ? args.block_size : 32;

    if (requested <= 32)
        launch_chebyshev_kernel<ShapeSymmetryT, 32>(args);
    else if (requested <= 64)
        launch_chebyshev_kernel<ShapeSymmetryT, 64>(args);
    else if (requested <= 128)
        launch_chebyshev_kernel<ShapeSymmetryT, 128>(args);
    else
        launch_chebyshev_kernel<ShapeSymmetryT, 256>(args);

    return hipSuccess;
    }

#endif // __HIPCC__

    } // end namespace kernel
    } // namespace azplugins
    } // namespace hoomd

#endif // AZPLUGINS_CHEBYSHEV_ANISOTROPIC_PAIR_POTENTIAL_GPU_CUH_
