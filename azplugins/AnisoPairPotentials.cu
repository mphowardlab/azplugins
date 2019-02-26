// Copyright (c) 2018-2019, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward / Everyone is free to add additional potentials

/*!
 * \file AnisoPairPotentials.cu
 * \brief Defines the driver functions for computing pair forces on the GPU
 *
 * Each pair potential evaluator needs to have an explicit instantiation of the
 * compute_aniso_pair_potential.
 */

#include "AnisoPairPotentials.cuh"

namespace azplugins
{
namespace gpu
{

//! Kernel driver for Two-patch Morse anisotropic pair potential
template cudaError_t compute_aniso_pair_potential<azplugins::detail::AnisoPairEvaluatorTwoPatchMorse>
    (const a_pair_args_t& pair_args,
     const typename azplugins::detail::AnisoPairEvaluatorTwoPatchMorse::param_type *d_params);

} // end namespace gpu
} // end namespace azplugins
