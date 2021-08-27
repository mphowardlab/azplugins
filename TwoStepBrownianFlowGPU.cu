// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

/*!
 * \file TwoStepBrownianFlowGPU.cu
 * \brief Definition of kernel drivers and kernels for TwoStepBrownianFlowGPU
 */

#include "TwoStepBrownianFlowGPU.cuh"
#include "FlowFields.h"

namespace azplugins
{
namespace gpu
{
//! Explicit instantiation of ConstantFlow integrator
template cudaError_t brownian_flow<azplugins::ConstantFlow>(Scalar4 *d_pos,
                                                            int3 *d_image,
                                                            const BoxDim& box,
                                                            const Scalar4 *d_net_force,
                                                            const unsigned int *d_tag,
                                                            const unsigned int *d_group,
                                                            const Scalar *d_diameter,
                                                            const Scalar lambda,
                                                            const Scalar *d_gamma,
                                                            const unsigned int ntypes,
                                                            const azplugins::ConstantFlow& flow_field,
                                                            const unsigned int N,
                                                            const Scalar dt,
                                                            const Scalar T,
                                                            const unsigned int timestep,
                                                            const unsigned int seed,
                                                            bool noiseless,
                                                            bool use_lambda,
                                                            const unsigned int block_size);
//! Explicit instantiation of ParabolicFlow integrator
template cudaError_t brownian_flow<azplugins::ParabolicFlow>(Scalar4 *d_pos,
                                                             int3 *d_image,
                                                             const BoxDim& box,
                                                             const Scalar4 *d_net_force,
                                                             const unsigned int *d_tag,
                                                             const unsigned int *d_group,
                                                             const Scalar *d_diameter,
                                                             const Scalar lambda,
                                                             const Scalar *d_gamma,
                                                             const unsigned int ntypes,
                                                             const azplugins::ParabolicFlow& flow_field,
                                                             const unsigned int N,
                                                             const Scalar dt,
                                                             const Scalar T,
                                                             const unsigned int timestep,
                                                             const unsigned int seed,
                                                             bool noiseless,
                                                             bool use_lambda,
                                                             const unsigned int block_size);
//! Explicit instantiation of QuiescentFluid integrator
template cudaError_t brownian_flow<azplugins::QuiescentFluid>(Scalar4 *d_pos,
                                                              int3 *d_image,
                                                              const BoxDim& box,
                                                              const Scalar4 *d_net_force,
                                                              const unsigned int *d_tag,
                                                              const unsigned int *d_group,
                                                              const Scalar *d_diameter,
                                                              const Scalar lambda,
                                                              const Scalar *d_gamma,
                                                              const unsigned int ntypes,
                                                              const azplugins::QuiescentFluid& flow_field,
                                                              const unsigned int N,
                                                              const Scalar dt,
                                                              const Scalar T,
                                                              const unsigned int timestep,
                                                              const unsigned int seed,
                                                              bool noiseless,
                                                              bool use_lambda,
                                                              const unsigned int block_size);
} // end namespace gpu
} // end namespace azplugins
