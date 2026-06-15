// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.



#ifndef AZPLUGINS_PERTURBED_LENNARD_JONES_EVAP_GPU_CUH_
#define AZPLUGINS_PERTURBED_LENNARD_JONES_EVAP_GPU_CUH_

#include "hip/hip_runtime.h"
#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/TextureTools.h"

#include "LinearInterpolator2D.h"

#include <assert.h>

namespace hoomd
    {
namespace azplugins
    {
namespace gpu
    {

//! Wraps arguments to kernel driver
struct compute_perturbed_lennard_jones_evap_args_t
    {
    //! Construct a compute_perturbed_lennard_jones_evap_args_t
    compute_perturbed_lennard_jones_evap_args_t(Scalar4* _d_force,
                                                const Scalar4* _d_pos,
                                                const BoxDim& _box,
                                                const unsigned int _N,
                                                const unsigned int* _d_n_neigh,
                                                const unsigned int* _d_nlist,
                                                const size_t* _d_head_list,
                                                const Scalar* _d_attraction_scale_factor_data,
                                                const uint2 _attraction_scale_factor_shape,
                                                const Scalar4 _domain,
                                                const Scalar _scaled_t,
                                                const Scalar _interface_height,
                                                const Scalar _lj1,
                                                const Scalar _lj2,
                                                const Scalar _epsilon_x_4,
                                                const Scalar _rcutsq,
                                                const Scalar _rwcasq,
                                                const bool _energy_shift,
                                                const unsigned int _block_size)
        : d_force(_d_force), d_pos(_d_pos), box(_box), d_n_neigh(_d_n_neigh), d_nlist(_d_nlist),
          d_head_list(_d_head_list),
          d_attraction_scale_factor_data(_d_attraction_scale_factor_data),
          attraction_scale_factor_shape(_attraction_scale_factor_shape), domain(_domain),
          scaled_t(_scaled_t), interface_height(_interface_height), lj1(_lj1), lj2(_lj2),
          epsilon_x_4(_epsilon_x_4), rcutsq(_rcutsq), rwcasq(_rwcasq), energy_shift(_energy_shift),
          block_size(_block_size) { };

    Scalar4* d_force;                             //!< Force to write out
    const Scalar4* d_pos;                         //!< Particle positions
    const BoxDim box;                             //!< Box dimensions
    const unsigned int N;                         //!< Number of particles
    const unsigned int* d_n_neigh;                //!< Number of neighbors per particle
    const unsigned int* d_nlist;                  //!< Neighbor list
    const size_t* d_head_list;                    //!< Head list indexing into the neighbor list
    const Scalar* d_attraction_scale_factor_data; //!< Flattened (y, t) table data
    const uint2 attraction_scale_factor_shape;    //!< Table shape (ny, nt)
    const Scalar4 domain;                         //!< Table domain [y_lo, y_hi, t_lo, t_hi]
    const Scalar scaled_t;                        //!< Scaled simulation time
    const Scalar interface_height;                //!< Interface height at this timestep
    const Scalar lj1;                             //!< 4 epsilon sigma^12
    const Scalar lj2;                             //!< 4 epsilon sigma^6
    const Scalar epsilon_x_4;                     //!< 4 epsilon
    const Scalar rcutsq;                          //!< Squared cutoff radius
    const Scalar rwcasq;                          //!< Squared WCA (potential minimum) radius
    const bool energy_shift;                      //!< Whether to shift the energy at the cutoff
    const unsigned int block_size;                //!< Block size to execute
    };

hipError_t compute_perturbed_lennard_jones_evap_forces(
    const compute_perturbed_lennard_jones_evap_args_t& args);

#ifdef __HIPCC__

namespace kernel
    {

__device__ __forceinline__ Scalar clamp_scaled_y(Scalar y, Scalar height)
    {
    const Scalar s = y / height;
    if (!(s > Scalar(0.0)))
        return Scalar(0.0);
    if (s > Scalar(1.0))
        return Scalar(1.0);
    return s;
    }

__global__ void
compute_perturbed_lennard_jones_evap_forces(Scalar4* d_force,
                                            const Scalar4* d_pos,
                                            const BoxDim& box,
                                            const unsigned int N,
                                            const unsigned int* d_n_neigh,
                                            const unsigned int* d_nlist,
                                            const size_t* d_head_list,
                                            const LinearInterpolator2D<Scalar>& interp,
                                            const Scalar interface_height,
                                            const Scalar scaled_t,
                                            const Scalar lj1,
                                            const Scalar lj2,
                                            const Scalar epsilon_x_4,
                                            const Scalar rcutsq,
                                            const Scalar rwcasq,
                                            const bool energy_shift)
    {
    // start by identifying which particle we are to handle
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    const Scalar4 postype_i = __ldg(d_pos + idx);
    const Scalar3 pos_i = make_scalar3(postype_i.x, postype_i.y, postype_i.z);

    const Scalar scaled_y_i = clamp_scaled_y(postype_i.y, interface_height);
    const Scalar attraction_scale_factor_i = interp(scaled_y_i, scaled_t);

    // initialize the force and energy to 0
    Scalar3 fi = make_scalar3(0, 0, 0);
    Scalar pei(0);

    // load in the length of the neighbor list for this thread
    const unsigned int n_neigh = d_n_neigh[idx];
    const size_t head = d_head_list[idx];

    // loop over neighbors
    for (unsigned int k = 0; k < n_neigh; ++k)
        {
        const unsigned int j = d_nlist[head + k];
        if (j == idx)
            continue;

        const Scalar4 postype_j = __ldg(d_pos + j);
        const Scalar3 pos_j = make_scalar3(postype_j.x, postype_j.y, postype_j.z);

        const Scalar scaled_y_j = clamp_scaled_y(postype_j.y, interface_height);
        const Scalar attraction_scale_factor_j = interp(scaled_y_j, scaled_t);

        // minimum-image
        Scalar3 dx = pos_i - pos_j;
        dx = box.minImage(dx);

        const Scalar rsq = dot(dx, dx);
        const Scalar attraction_scale_factor_avg
            = Scalar(0.5) * (attraction_scale_factor_i + attraction_scale_factor_j);

        const Scalar wca_shift
            = epsilon_x_4 * (Scalar(1.0) - attraction_scale_factor_avg) / Scalar(4.0);

        if (rsq < rcutsq && lj1 != 0)
            {
            const Scalar r2inv = Scalar(1) / rsq;
            const Scalar r6inv = r2inv * r2inv * r2inv;

            Scalar pair_eng = r6inv * (lj1 * r6inv - lj2);
            Scalar force_divr = r6inv * r2inv * (Scalar(12.0) * lj1 * r6inv - Scalar(6.0) * lj2);

            if (rsq < rwcasq)
                {
                pair_eng += wca_shift;
                }
            else
                {
                pair_eng *= attraction_scale_factor_avg;
                force_divr *= attraction_scale_factor_avg;
                }

            if (energy_shift)
                {
                const Scalar rcut2inv = Scalar(1.0) / rcutsq;
                const Scalar rcut6inv = rcut2inv * rcut2inv * rcut2inv;

                Scalar pair_eng_shift = rcut6inv * (lj1 * rcut6inv - lj2);

                if (rcutsq < rwcasq)
                    {
                    pair_eng_shift += wca_shift;
                    }
                else
                    {
                    pair_eng_shift *= attraction_scale_factor_avg;
                    }

                // apply the shift to the pair energy
                pair_eng -= pair_eng_shift;
                }

            fi.x += force_divr * dx.x;
            fi.y += force_divr * dx.y;
            fi.z += force_divr * dx.z;

            pei += pair_eng;
            }
        }

    d_force[idx] = make_scalar4(fi.x, fi.y, fi.z, Scalar(0.5) * pei);
    }

    } // end namespace kernel

hipError_t
compute_perturbed_lennard_jones_evap_forces(const compute_perturbed_lennard_jones_evap_args_t& args)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(
        &attr,
        reinterpret_cast<const void*>(&kernel::compute_perturbed_lennard_jones_evap_forces));
    max_block_size = attr.maxThreadsPerBlock;

    const unsigned int run_block_size = min(args.block_size, max_block_size);

    // setup the grid to run the kernel
    dim3 grid(args.N / run_block_size + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    // run the kernel
    hipLaunchKernelGGL((kernel::compute_perturbed_lennard_jones_evap_forces),
                       grid,
                       threads,
                       args.d_force,
                       args.d_pos,
                       args.N,
                       args.d_n_neigh,
                       args.d_nlist,
                       args.d_head_list,
                       args.interp,
                       args.scaled_t,
                       args.interface_height,
                       args.lj1,
                       args.lj2,
                       args.epsilon_x_4,
                       args.rcutsq,
                       args.rwcasq,
                       args.energy_shift);

    return hipSuccess;
    }
#endif // __HIPCC__

    } // end namespace gpu
    } // end namespace azplugins
    } // end namespace hoomd

#endif // AZPLUGINS_PERTURBED_LENNARD_JONES_EVAP_GPU_CUH_
