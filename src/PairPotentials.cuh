// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file PairPotentials.cuh
 * \brief Declares driver function for computing pair forces on the GPU
 *
 * A templated function for each driver should be instantiated in its own .cu file.
 */

#ifndef AZPLUGINS_PAIR_POTENTIAL_DRIVERS_CUH_
#define AZPLUGINS_PAIR_POTENTIAL_DRIVERS_CUH_

#include "PairPotentials.h"
#include "hoomd/md/PotentialPairGPU.cuh"

namespace azplugins
    {
namespace gpu
    {
//! Pair potential compute kernel driver
/*!
 * \param pair_args Standard pair potential arguments
 * \param d_params Specific parameters required for the potential
 * \tparam evaluator Evaluator functor
 */
template<class evaluator>
cudaError_t compute_pair_potential(const pair_args_t& pair_args,
                                   const typename evaluator::param_type* d_params);

#ifdef NVCC
/*!
 * This implements the templated kernel driver when compiled in NVCC only. The template
 * must be specifically instantiated per potential in a cu file.
 */
template<class evaluator>
cudaError_t compute_pair_potential(const pair_args_t& pair_args,
                                   const typename evaluator::param_type* d_params)
    {
    return ::gpu_compute_pair_forces<evaluator>(pair_args, d_params);
    }
#endif
    } // end namespace gpu
    } // end namespace azplugins

#endif // AZPLUGINS_PAIR_POTENTIAL_DRIVERS_CUH_
