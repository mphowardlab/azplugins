// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward / Everyone is free to add additional potentials

/*!
 * \file SpecialPairPotentials.cu
 * \brief Defines the driver functions for computing special pair forces on the GPU
 *
 * Each special pair potential evaluator needs to have an explicit instantiation of the
 * compute_special_pair_potential.
 */

#include "SpecialPairPotentials.cuh"

namespace azplugins
{
namespace gpu
{

//! LJ 9-6 special pair potential
typedef azplugins::detail::SpecialPairEvaluator<azplugins::detail::PairEvaluatorLJ96> SpecialPairEvaluatorLJ96;
//! Kernel driver for LJ 9-6 special pair potential
template cudaError_t compute_special_pair_potential<SpecialPairEvaluatorLJ96>
     (const bond_args_t& bond_args,
      const typename SpecialPairEvaluatorLJ96::param_type *d_params,
      unsigned int *d_flags);

} // end namespace gpu
} // end namespace azplugins
