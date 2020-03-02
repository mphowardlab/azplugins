// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

#include "PairPotentials.cuh"

namespace azplugins
{
namespace gpu
{

//! Kernel driver for LJ 9-6 pair potential
template cudaError_t compute_pair_potential<azplugins::detail::PairEvaluatorLJ96>
     (const pair_args_t& pair_args,
     const typename azplugins::detail::PairEvaluatorLJ96::param_type *d_params);

} // end namespace gpu
} // end namespace azplugins
