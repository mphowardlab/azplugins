// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

// clang-format off
#include "ImagePotentialBondGPU.cuh"
#include "hoomd/md/EvaluatorBondHarmonic.h"
// clang-format on

namespace hoomd
    {
namespace azplugins
    {
namespace kernel
    {
template __attribute__((visibility("default"))) hipError_t
gpu_compute_bond_forces<hoomd::md::EvaluatorBondHarmonic, 2>(
    const kernel::bond_args_t<2>& bond_args,
    const hoomd::md::EvaluatorBondHarmonic::param_type* d_params,
    unsigned int* d_flags);

    } // end namespace kernel
    } // end namespace azplugins
    } // end namespace hoomd
