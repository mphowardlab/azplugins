// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt


#ifndef AZPLUGINS_SINE_EXPANSION_CONSTRICTION_GEOMETRY_FILLER_GPU_CUH_
#define AZPLUGINS_SINE_EXPANSION_CONSTRICTION_GEOMETRY_FILLER_GPU_CUH_

/*!
 * \file SinusoidalExpansionConstrictionFillerGPU.cuh
 * \brief Declaration of CUDA kernels for mpcd::SinusoidalExpansionConstrictionFillerGPU
 */

#include <cuda_runtime.h>

#include "SinusoidalExpansionConstrictionGeometry.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"


namespace azplugins
{
namespace gpu
{

//! Draw virtual particles in the SineGeometry
cudaError_t sin_expansion_constriction_draw_particles(Scalar4 *d_pos,
                                                      Scalar4 *d_vel,
                                                      unsigned int *d_tag,
                                                      const azplugins::detail::SinusoidalExpansionConstriction& geom,
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

#endif // AZPLUGINS_SINE_EXPANSION_CONSTRICTION_GEOMETRY_FILLER_GPU_CUH_
