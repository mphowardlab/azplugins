// Copyright (c) 2016-2017, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward / Everyone is free to add additional potentials

/*!
 * \file PairPotentials.cu
 * \brief Defines the driver functions for computing pair forces on the GPU
 *
 * Each pair potential evaluator needs to have an explicit instantiation of the
 * compute_pair_potential.
 */

#include "PairPotentials.cuh"

namespace azplugins
{
namespace gpu
{

//! Kernel driver for Ashbaugh-Hatch pair potential
template cudaError_t compute_pair_potential<azplugins::detail::PairEvaluatorAshbaugh>
    (const pair_args_t& pair_args,
     const typename azplugins::detail::PairEvaluatorAshbaugh::param_type *d_params);

//! Kernel driver for colloid (integrated Lennard-Jones) pair potential
template cudaError_t compute_pair_potential<azplugins::detail::PairEvaluatorColloid>
    (const pair_args_t& pair_args,
     const typename azplugins::detail::PairEvaluatorColloid::param_type *d_params);

//! Kernel driver for core-shifted Lennard-Jones pair potential
template cudaError_t compute_pair_potential<azplugins::detail::PairEvaluatorShiftedLJ>
    (const pair_args_t& pair_args,
     const typename azplugins::detail::PairEvaluatorShiftedLJ::param_type *d_params);

} // end namespace gpu
} // end namespace azplugins
