// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

// Adapted from hoomd/md/export_PotentialExternalWallGPUKernel.cu.inc of HOOMD-blue.
// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// See CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off
#include "hoomd/md/PotentialExternalGPU.cuh"
#include "hoomd/md/EvaluatorWalls.h"
#include "WallEvaluator@_evaluator@.h"

#define EVALUATOR_CLASS hoomd::md::EvaluatorWalls<hoomd::azplugins::detail::WallEvaluator@_evaluator@>
// clang-format on

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
template __attribute__((visibility("default"))) hipError_t
gpu_compute_potential_external_forces<EVALUATOR_CLASS>(
    const external_potential_args_t& external_potential_args,
    const typename EVALUATOR_CLASS::param_type* d_params,
    const typename EVALUATOR_CLASS::field_type* d_field);
    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
