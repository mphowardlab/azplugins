// Copyright (c) 2018-2019, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt


#ifndef AZPLUGINS_MPCD_SINE_GEOMETRY_FILLER_GPU_CUH_
#define AZPLUGINS_MPCD_SINE_GEOMETRY_FILLER_GPU_CUH_

/*!
 * \file mpcd/SineGeometryFillerGPU.cuh
 * \brief Declaration of CUDA kernels for mpcd::SineGeometryFillerGPU
 */

#include <cuda_runtime.h>

#include "MPCDSineGeometry.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"


namespace azplugins
{
namespace gpu
{

//! Draw virtual particles in the SineGeometry
cudaError_t sine_draw_particles(Scalar4 *d_pos,
                                Scalar4 *d_vel,
                                unsigned int *d_tag,
                                const azplugins::detail::SineGeometry& geom,
                                const Scalar m_pi_period_div_L,
                                const Scalar m_amplitude,
                                const Scalar m_H_narrow,
                                const Scalar m_thickness,
                                const BoxDim& box,
                                const Scalar mass,
                                const unsigned int type,
                                const unsigned int N_lo,
                                const unsigned int N_hi,
                                const unsigned int first_tag,
                                const unsigned int first_idx,
                                const Scalar kT,
                                const unsigned int timestep,
                                const unsigned int seed,
                                const unsigned int block_size);

} // end namespace gpu
} // end namespace azplugins

#endif // AZPLUGINS_MPCD_SINE_GEOMETRY_FILLER_GPU_CUH_
