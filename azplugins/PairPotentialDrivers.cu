// Maintainer: mphoward / Everyone is free to add additional potentials

/*!
 * \file PairPotentialDrivers.cu
 * \brief Defines the driver functions for computing pair forces on the GPU
 */

#include "PairPotentialDrivers.cuh"

// All pair evaluators need to be included below.
#include "PairEvaluatorAshbaugh.h"

namespace azplugins
{
namespace gpu
{

cudaError_t compute_pair_ashbaugh(const pair_args_t& pair_args,
                                  const azplugins::detail::ashbaugh_params *d_params)
    {
    return gpu_compute_pair_forces<azplugins::detail::PairEvaluatorAshbaugh>(pair_args, d_params);
    }

} // end namespace gpu
} // end namespace azplugins
