// Copyright (c) 2016-2018, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file SpecialPairPotentials.cuh
 * \brief Declares driver function for computing special pair forces on the GPU
 *
 * A templated function for each driver should be instantiated in
 * SpecialPairPotentials.cu.
 */

#ifndef AZPLUGINS_SPECIAL_PAIR_POTENTIAL_DRIVERS_CUH_
#define AZPLUGINS_SPECIAL_PAIR_POTENTIAL_DRIVERS_CUH_

#include "hoomd/md/PotentialBondGPU.cuh"
#include "SpecialPairPotentials.h"

namespace azplugins
{
namespace gpu
{
//! Special pair potential compute kernel driver
/*!
 * \param bond_args Standard special pair potential arguments
 * \param d_params Specific parameters required for the potential
 * \param d_flags Flags to mark if computation failed (output)
 * \tparam evaluator Evaluator functor
 */
template<class evaluator>
cudaError_t compute_special_pair_potential(const bond_args_t& bond_args,
                                           const typename evaluator::param_type *d_params,
                                           unsigned int *d_flags);

#ifdef NVCC
/*!
 * This implements the templated kernel driver when compiled in NVCC only. The template
 * must be specifically instantiated per potential in a cu file.
 */
template<class evaluator>
cudaError_t compute_special_pair_potential(const bond_args_t& bond_args,
                                           const typename evaluator::param_type *d_params,
                                           unsigned int *d_flags)
    {
    return ::gpu_compute_bond_forces<evaluator>(bond_args, d_params, d_flags);
    }
#endif
} // end namespace gpu
} // end namespace azplugins

#endif // AZPLUGINS_SPECIAL_PAIR_POTENTIAL_DRIVERS_CUH_
