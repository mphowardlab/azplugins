// Copyright (c) 2016-2018, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file DPDPotentials.cuh
 * \brief Declares driver function for computing DPD forces on the GPU
 *
 * A templated function for each driver should be instantiated in
 * DPDPotentials.cu.
 */

#ifndef AZPLUGINS_DPD_POTENTIALS_CUH_
#define AZPLUGINS_DPD_POTENTIALS_CUH_

#include "hoomd/md/PotentialPairDPDThermoGPU.cuh"
#include "DPDPotentials.h"

namespace azplugins
{
namespace gpu
{
//! DPD potential compute kernel driver
/*!
 * \param dpd_args Standard DPD potential arguments
 * \param d_params Specific parameters required for the potential
 * \tparam evaluator Evaluator functor
 */
template<class evaluator>
cudaError_t compute_dpd_potential(const dpd_pair_args_t& dpd_args, const typename evaluator::param_type *d_params);

#ifdef NVCC
/*!
 * This implements the templated kernel driver when compiled in NVCC only. The template
 * must be specifically instantiated per potential in a cu file.
 */
template<class evaluator>
cudaError_t compute_dpd_potential(const dpd_pair_args_t& dpd_args, const typename evaluator::param_type *d_params)
    {
    return ::gpu_compute_dpd_forces<evaluator>(dpd_args, d_params);
    }
#endif
} // end namespace gpu
} // end namespace azplugins

#endif // AZPLUGINS_DPD_POTENTIALS_CUH_
