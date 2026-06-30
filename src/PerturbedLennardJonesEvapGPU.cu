// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#include "PairEvaluatorPerturbedLennardJones.h"
#include "PerturbedLennardJonesEvapGPU.cuh"
#include "hoomd/WarpTools.cuh"

namespace hoomd
    {
namespace azplugins
    {
namespace detail
    {
namespace gpu
    {
namespace kernel
    {

__global__ void compute_attraction_scale_factor(Scalar* d_scale_factor,
                                                const Scalar4* d_pos,
                                                const unsigned int Ntot,
                                                const LinearInterpolator2D<Scalar> interp,
                                                const Scalar interface_height,
                                                const Scalar scaled_t,
                                                const Scalar y_lo)
    {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Ntot)
        return;

    const Scalar y = __ldg(d_pos + i).y;
    Scalar scaled_pos_y = (y - y_lo) / (interface_height - y_lo);

    if (!(scaled_pos_y > Scalar(0.0)))
        scaled_pos_y = Scalar(0.0);
    if (scaled_pos_y > Scalar(1.0))
        scaled_pos_y = Scalar(1.0);

    d_scale_factor[i] = interp(scaled_pos_y, scaled_t);
    }

__global__ void
compute_perturbed_lennard_jones_evap_forces(Scalar4* d_force,
                                            const Scalar4* d_pos,
                                            Scalar* d_scale_factor,
                                            const BoxDim box,
                                            const unsigned int N,
                                            const unsigned int n_ghost,
                                            const unsigned int* d_n_neigh,
                                            const unsigned int* d_nlist,
                                            const size_t* d_head_list,
                                            const Scalar rcutsq,
                                            PairParametersPerturbedLennardJones params,
                                            const bool energy_shift)
    {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    const Scalar4 postype_i = __ldg(d_pos + idx);
    const Scalar3 pos_i = make_scalar3(postype_i.x, postype_i.y, postype_i.z);

    const Scalar attraction_scale_factor_i = __ldg(d_scale_factor + idx);
    // initialize the force and energy to 0
    Scalar3 fi = make_scalar3(0, 0, 0);
    Scalar pei = 0;

    const unsigned int n_neigh = d_n_neigh[idx];
    const size_t head = d_head_list[idx];

    // loop over this particle's neighbors
    for (unsigned int k = 0; k < n_neigh; ++k)
        {
        const unsigned int j = __ldg(d_nlist + head + k);
        if (j == idx)
            continue;

        const Scalar4 postype_j = __ldg(d_pos + j);
        const Scalar3 pos_j = make_scalar3(postype_j.x, postype_j.y, postype_j.z);
        const Scalar attraction_scale_factor_j = __ldg(d_scale_factor + j);

        // minimum-image
        Scalar3 dx = pos_i - pos_j;
        dx = box.minImage(dx);
        const Scalar rsq = dot(dx, dx);

        params.attraction_scale_factor
            = Scalar(0.5) * (attraction_scale_factor_i + attraction_scale_factor_j);

        Scalar force_divr = Scalar(0.0);
        Scalar pair_eng = Scalar(0.0);
        PairEvaluatorPerturbedLennardJones eval(rsq, rcutsq, params);
        if (eval.evalForceAndEnergy(force_divr, pair_eng, energy_shift))
            {
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
    assert(args.block_size != 0);

    hipFuncAttributes attr;
    hipFuncGetAttributes(
        &attr,
        reinterpret_cast<const void*>(&kernel::compute_perturbed_lennard_jones_evap_forces));
    const unsigned max_block_size = attr.maxThreadsPerBlock;
    const unsigned int run_block_size = min(args.block_size, max_block_size);

    const unsigned int Ntot = args.N + args.n_ghost;

    dim3 threads(run_block_size, 1, 1);
    dim3 grid((Ntot + run_block_size - 1) / run_block_size, 1, 1);
    const Scalar ylo = args.box.getLo().y;
    hipLaunchKernelGGL((gpu::kernel::compute_attraction_scale_factor),
                       grid,
                       threads,
                       0,
                       0,
                       args.d_scale_factor,
                       args.d_pos,
                       Ntot,
                       args.interp,
                       args.interface_height,
                       args.scaled_t,
                       ylo);

    hipLaunchKernelGGL((gpu::kernel::compute_perturbed_lennard_jones_evap_forces),
                       grid,
                       threads,
                       0,
                       0,
                       args.d_force,
                       args.d_pos,
                       args.d_scale_factor,
                       args.box,
                       args.N,
                       args.n_ghost,
                       args.d_n_neigh,
                       args.d_nlist,
                       args.d_head_list,
                       args.rcutsq,
                       args.params,
                       args.energy_shift);

    return hipSuccess;
    }

    } // end namespace gpu
    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd
