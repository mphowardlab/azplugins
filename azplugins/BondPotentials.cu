// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward / Everyone is free to add additional potentials

/*!
 * \file BondPotentials.cu
 * \brief Defines the driver functions for computing bonded forces on the GPU
 *
 * Each bond potential evaluator needs to have an explicit instantiation of the
 * compute_bond_potential.
 */

#include "BondPotentials.cuh"

namespace azplugins
{
namespace gpu
{

//! Kernel driver for double well bond potential
template cudaError_t compute_bond_potential<azplugins::detail::BondEvaluatorDoubleWell>
    (const bond_args_t& bond_args,
     const typename azplugins::detail::BondEvaluatorDoubleWell::param_type *d_params,
     unsigned int *d_flags);

//! Kernel driver for fene  bond potential
template cudaError_t compute_bond_potential<azplugins::detail::BondEvaluatorFENE>
    (const bond_args_t& bond_args,
     const typename azplugins::detail::BondEvaluatorFENE::param_type *d_params,
     unsigned int *d_flags);

//! Kernel driver for Ashbaugh-Hatch 48-24  bond potential
template cudaError_t compute_bond_potential<azplugins::detail::BondEvaluatorFENEAsh24>
    (const bond_args_t& bond_args,
     const typename azplugins::detail::BondEvaluatorFENEAsh24::param_type *d_params,
     unsigned int *d_flags);
} // end namespace gpu
} // end namespace azplugins
