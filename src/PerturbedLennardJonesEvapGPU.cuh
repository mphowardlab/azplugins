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
#include "PairEvaluatorPerturbedLennardJones.h"
#include <assert.h>

namespace hoomd
    {
namespace azplugins
    {
namespace detail
    {
namespace gpu
    {
//! Wraps arguments to kernel driver
struct perturbed_lennard_jones_evap_args_t
    {
    perturbed_lennard_jones_evap_args_t(Scalar4* _d_force,
                                        const Scalar4* _d_pos,
                                        Scalar* _d_scale_factor,
                                        const BoxDim _box,
                                        const unsigned int _N,
                                        const unsigned int _n_ghost,
                                        const unsigned int* _d_n_neigh,
                                        const unsigned int* _d_nlist,
                                        const size_t* _d_head_list,
                                        const LinearInterpolator2D<Scalar>& _interp,
                                        const Scalar _scaled_t,
                                        const Scalar _interface_height,
                                        const Scalar _rcutsq,
                                        const PairParametersPerturbedLennardJones* _d_params,
                                        const unsigned int _ntypes,
                                        const bool _energy_shift,
                                        const unsigned int _block_size)
        : d_force(_d_force), d_pos(_d_pos), d_scale_factor(_d_scale_factor), box(_box), N(_N),
          n_ghost(_n_ghost), d_n_neigh(_d_n_neigh), d_nlist(_d_nlist), d_head_list(_d_head_list),
          interp(_interp), scaled_t(_scaled_t), interface_height(_interface_height),
          rcutsq(_rcutsq), d_params(_d_params), ntypes(_ntypes), energy_shift(_energy_shift),
          block_size(_block_size) { };

    Scalar4* d_force;
    const Scalar4* d_pos;
    Scalar* d_scale_factor;
    const BoxDim box;
    const unsigned int N;
    const unsigned int n_ghost;
    const unsigned int* d_n_neigh;
    const unsigned int* d_nlist;
    const size_t* d_head_list;
    const LinearInterpolator2D<Scalar> interp;
    const Scalar scaled_t;
    const Scalar interface_height;
    const Scalar rcutsq;
    const PairParametersPerturbedLennardJones* d_params; //!< Per type-pair parameter table
    const unsigned int ntypes;                           //!< Number of particle types
    const bool energy_shift;
    const unsigned int block_size;
    };

hipError_t
compute_perturbed_lennard_jones_evap_forces(const perturbed_lennard_jones_evap_args_t& args);

    } // end namespace gpu
    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd

#endif // AZPLUGINS_PERTURBED_LENNARD_JONES_EVAP_GPU_CUH_
