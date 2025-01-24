// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

// Adapted from hoomd/md/export_PotentialBondGPUKernel.cu.inc of HOOMD-blue.
// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// See CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off
#include "hoomd/md/PotentialBondGPU.cuh"
#include "BondEvaluator@_evaluator@.h"

#define EVALUATOR_CLASS BondEvaluator@_evaluator@
// clang-format on

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
template __attribute__((visibility("default"))) hipError_t
gpu_compute_bond_forces<hoomd::azplugins::detail::EVALUATOR_CLASS, 2>(
    const kernel::bond_args_t<2>& bond_args,
    const typename hoomd::azplugins::detail::EVALUATOR_CLASS::param_type* d_params,
    unsigned int* d_flags);
    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
