// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#include "PairPotentials.cuh"

namespace azplugins
    {
namespace gpu
    {

//! Kernel driver for Ashbaugh-Hatch pair potential
template cudaError_t compute_pair_potential<azplugins::detail::PairEvaluatorAshbaugh>(
    const pair_args_t& pair_args,
    const typename azplugins::detail::PairEvaluatorAshbaugh::param_type* d_params);

    } // end namespace gpu
    } // end namespace azplugins
