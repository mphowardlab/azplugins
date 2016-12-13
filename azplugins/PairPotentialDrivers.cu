// Maintainer: mphoward / Everyone is free to add additional potentials

/*!
 * \file PairPotentialDrivers.cu
 * \brief Defines the driver functions for computing pair forces on the GPU
 *
 * Each driver function should launch the gpu_compute_pair_forces kernel
 * templated on the pair evaluator.
 */

#include "PairPotentialDrivers.cuh"

namespace azplugins
{
namespace gpu
{

cudaError_t compute_pair_ashbaugh(const pair_args_t& pair_args,
                                  const azplugins::detail::PairEvaluatorAshbaugh::param_type *d_params)
    {
    return gpu_compute_pair_forces<azplugins::detail::PairEvaluatorAshbaugh>(pair_args, d_params);
    }

} // end namespace gpu
} // end namespace azplugins
