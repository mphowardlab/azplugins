// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

// See CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off
#include "hoomd/md/AnisoPotentialPairGPU.cuh"
#include "AnisoPairEvaluator@_evaluator@.h"

#define EVALUATOR_CLASS AnisoPairEvaluator@_evaluator@
// clang-format on

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
template __attribute__((visibility("default"))) hipError_t
gpu_compute_pair_aniso_forces<hoomd::azplugins::detail::EVALUATOR_CLASS>(
    const a_pair_args_t& pair_args,
    const hoomd::azplugins::detail::EVALUATOR_CLASS::param_type* d_params,
    const hoomd::azplugins::detail::EVALUATOR_CLASS::shape_type* d_shape_param);
    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
