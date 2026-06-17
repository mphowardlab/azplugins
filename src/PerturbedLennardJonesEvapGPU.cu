// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#include "PerturbedLennardJonesEvapGPU.cuh"

namespace hoomd
    {
namespace azplugins
    {
namespace gpu
    {
namespace kernel
    {

__device__ __inline__ Scalar clamp_scaled_y(Scalar y, Scalar height)
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
    Scalar pei = 0;

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
compute_perturbed_lennard_jones_evap_forces(const perturbed_lennard_jones_evap_args_t& args)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(
        &attr,
        reinterpret_cast<const void*>(&kernel::compute_perturbed_lennard_jones_evap_forces));
    max_block_size = attr.maxThreadsPerBlock;

    const unsigned int run_block_size = min(args.block_size, max_block_size);

    dim3 grid(args.N / run_block_size + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    hipLaunchKernelGGL((azplugins::gpu::kernel::compute_perturbed_lennard_jones_evap_forces),
                       grid,
                       threads,
                       0,
                       0,
                       args.d_force,
                       args.d_pos,
                       args.box,
                       args.N,
                       args.d_n_neigh,
                       args.d_nlist,
                       args.d_head_list,
                       args.interp,
                       args.interface_height,
                       args.scaled_t,
                       args.lj1,
                       args.lj2,
                       args.epsilon_x_4,
                       args.rcutsq,
                       args.rwcasq,
                       args.energy_shift);

    return hipSuccess;
    }

    } // end namespace gpu
    } // end namespace azplugins
    } // end namespace hoomd
