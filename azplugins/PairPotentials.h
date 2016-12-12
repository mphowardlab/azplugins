// Maintainer: mphoward / Everyone is free to add additional potentials

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
