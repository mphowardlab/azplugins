// Maintainer: mphoward / Everyone is free to add additional potentials

/*!
 * \file azplugins/PairiPotentialDrivers.cuh
 * \brief Declares driver functions for computing pair forces on the GPU
 *
 * One driver function should be declared for each pair potential. They
 * should be named like `compute_pair_my_great_potential`, and should
 * accept two arguments: a const reference to pair_args_t, and a pointer
 * to the parameter type required for your evaluator.
 */

#ifndef AZPLUGINS_PAIR_POTENTIAL_DRIVERS_CUH_
#define AZPLUGINS_PAIR_POTENTIAL_DRIVERS_CUH_

#include "hoomd/md/PotentialPairGPU.cuh"

// All pair potential evaluators must be included here
#include "PairEvaluatorAshbaugh.h"

namespace azplugins
{
namespace gpu
{
//! Compute pair forces on the GPU with PairEvaluatorAshbaugh
cudaError_t compute_pair_ashbaugh(const pair_args_t& pair_args,
                                  const azplugins::detail::PairEvaluatorAshbaugh::param_type *d_params);
} // end namespace gpu
} // end namespace azplugins

#endif // AZPLUGINS_PAIR_POTENTIAL_DRIVERS_CUH_
