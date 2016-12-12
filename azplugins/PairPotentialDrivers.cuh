// Maintainer: mphoward / Everyone is free to add additional potentials

/*!
 * \file azplugins/PairiPotentialDrivers.cuh
 * \brief Declares driver functions for computing pair forces on the GPU
 */

#ifndef AZPLUGINS_PAIR_POTENTIALDRIVERS_CUH_
#define AZPLUGINS_PAIR_POTENTIAL_DRIVERS_CUH_

#include "hoomd/md/PotentialPairGPU.cuh"

/* Any pair potentials defining custom parameter types in their evaluators
 * need to be included here.
 */
#include "PairEvaluatorAshbaugh.h"

namespace azplugins
{
namespace gpu
{
//! Compute pair forces on the GPU with PairEvaluatorAshbaugh
cudaError_t compute_pair_ashbaugh(const pair_args_t& pair_args,
                                  const azplugins::detail::ashbaugh_params *d_params);
} // end namespace gpu
} // end namespace azplugins

#endif // AZPLUGINS_PAIR_POTENTIAL_DRIVERS_CUH_
