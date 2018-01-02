// Copyright (c) 2016-2017, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward / Everyone is free to add additional potentials

/*!
 * \file DPDPotentials.cu
 * \brief Defines the driver functions for computing DPD forces on the GPU
 *
 * Each DPD potential evaluator needs to have an explicit instantiation of the
 * compute_dpd_potential.
 */

#include "DPDPotentials.cuh"

namespace azplugins
{
namespace gpu
{

//! Kernel driver for modified DPD potential
template cudaError_t compute_dpd_potential<azplugins::detail::DPDEvaluatorGeneralWeight>
    (const dpd_pair_args_t& dpd_args,
     const typename azplugins::detail::DPDEvaluatorGeneralWeight::param_type *d_params);

} // end namespace gpu
} // end namespace azplugins
