// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt


#ifndef AZPLUGINS_MPCD_ANTI_SYM_COS_GEOMETRY_FILLER_GPU_CUH_
#define AZPLUGINS_MPCD_ANTI_SYM_COS_GEOMETRY_FILLER_GPU_CUH_

/*!
 * \file SinusoidalExpansionConstrictionFillerGPU.cuh
 * \brief Declaration of CUDA kernels for mpcd::SinusoidalExpansionConstrictionFillerGPU
 */

#include <cuda_runtime.h>

#include "MPCDSinusoidalChannel.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"


namespace azplugins
{
namespace gpu
{

//! Draw virtual particles in the SineGeometry
cudaError_t anti_sym_cos_draw_particles(Scalar4 *d_pos,
                                Scalar4 *d_vel,
                                unsigned int *d_tag,
                                const azplugins::detail::SinusoidalChannel& geom,
                                const Scalar pi_period_div_L,
                                const Scalar amplitude,
                                const Scalar H_narrow,
                                const Scalar thickness,
                                const BoxDim& box,
                                const Scalar mass,
                                const unsigned int type,
                                const unsigned int N_fill,
                                const unsigned int first_tag,
                                const unsigned int first_idx,
                                const Scalar kT,
                                const unsigned int timestep,
                                const unsigned int seed,
                                const unsigned int block_size);

} // end namespace gpu
} // end namespace azplugins

#endif // AZPLUGINS_MPCD_ANTI_SYM_COS_GEOMETRY_FILLER_GPU_CUH_
