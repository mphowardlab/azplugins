// Maintainer: mphoward / Everyone is free to add additional potentials

/*!
 * \file PairPotentials.h
 * \brief Defines names of all pair potentials templated on evaluators.
 *
 * In HOOMD-blue, pair potentials are templated on a base class ForceCompute called PotentialPair,
 * which uses an evaluator functor to compute the actual details of the pair potential.
 * This avoids code duplication of calling the neighbor list, computing pair distances, etc.
 *
 * To add a new pair potential, take the following steps:
 *  1. Create an evaluator functor for your potential, for example PairEvaluatorMyGreatPotential.h.
 *     This file should be included below. You can follow one of the other evaluator functors as
 *     an example for the details.
 *
 *  2. Declare a CUDA driver function in PairPotentialDrivers.cuh. These functions drive the launching
 *     of a templated CUDA kernel for the pair potential. See the documentation there for implementation details.
 *
 *  3. Define the driver function in PairPotentialDrivers.cu. The driver function needs to launch
 *     the templated kernel.
 *
 *  4. Expose the pair potential on the python level in pair.py.
 *
 *  5. Write a unit test for the potential in test-py. Two types of tests should be conducted: one that
 *     checks that all methods work on the python object, and one that validates the force and energy between
 *     particle pairs at fixed distances.
 */

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "hoomd/md/PotentialPair.h"
#ifdef ENABLE_CUDA
#include "hoomd/md/PotentialPairGPU.h"
#include "PairPotentialDrivers.cuh"
#endif

// All pair potential evaluators must be included here
#include "PairEvaluatorAshbaugh.h"

namespace azplugins
{

typedef ::PotentialPair<azplugins::detail::PairEvaluatorAshbaugh> PairPotentialAshbaugh;

#ifdef ENABLE_CUDA
typedef ::PotentialPairGPU<azplugins::detail::PairEvaluatorAshbaugh, azplugins::gpu::compute_pair_ashbaugh> PairPotentialAshbaughGPU;
#endif // ENABLE_CUDA

} // end namespace azplugins
