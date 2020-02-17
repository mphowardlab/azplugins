// Copyright (c) 2018-2019, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt

#include "hoomd/mpcd/ExternalField.h"
#include "hoomd/GPUPolymorph.cuh"

template mpcd::BlockForce* hoomd::gpu::device_new(Scalar,Scalar,Scalar);
template mpcd::ConstantForce* hoomd::gpu::device_new(Scalar3);
template mpcd::SineForce* hoomd::gpu::device_new(Scalar,Scalar);
template void hoomd::gpu::device_delete(mpcd::ExternalField*);

#include "hoomd/mpcd/ConfinedStreamingMethodGPU.cuh"
#include "MPCDSineGeometry.h"


//! Template instantiation of sine geometry streaming
template cudaError_t mpcd::gpu::confined_stream<azplugins::detail::SineGeometry>
    (const mpcd::gpu::stream_args_t& args, const azplugins::detail::SineGeometry& geom);
