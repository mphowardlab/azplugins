// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

#include "PairPotentials.cuh"

namespace azplugins
{
namespace gpu
{

//! Kernel driver for LJ 12-4 pair potential
template cudaError_t compute_pair_potential<azplugins::detail::PairEvaluatorLJ124>
     (const pair_args_t& pair_args,
     const typename azplugins::detail::PairEvaluatorLJ124::param_type *d_params);

} // end namespace gpu
} // end namespace azplugins
