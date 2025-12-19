// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

// See CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off
#include "HarmonicBarrierGPU.cuh"
#include "@_geometry@BarrierEvaluator.h"

#define BARRIER_EVALUATOR @_geometry@BarrierEvaluator
// clang-format on

namespace hoomd
    {
namespace azplugins
    {
namespace gpu
    {

template __attribute__((visibility("default"))) cudaError_t
compute_harmonic_barrier<BARRIER_EVALUATOR>(Scalar4* d_force,
                                            Scalar* d_virial,
                                            const Scalar4* d_pos,
                                            const Scalar2* d_params,
                                            const BoxDim& global_box,
                                            const BARRIER_EVALUATOR& evaluator,
                                            const unsigned int N,
                                            const unsigned int ntypes,
                                            const unsigned int block_size);

    } // end namespace gpu
    } // namespace azplugins
    } // end namespace hoomd
