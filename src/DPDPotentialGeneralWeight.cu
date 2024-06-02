// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#include "DPDPotentials.cuh"

namespace azplugins
    {
namespace gpu
    {

//! Kernel driver for modified DPD potential
template cudaError_t compute_dpd_potential<azplugins::detail::DPDEvaluatorGeneralWeight>(
    const dpd_pair_args_t& dpd_args,
    const typename azplugins::detail::DPDEvaluatorGeneralWeight::param_type* d_params);

    } // end namespace gpu
    } // end namespace azplugins
