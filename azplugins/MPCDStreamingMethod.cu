// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt

#include "hoomd/mpcd/ExternalField.h"
#include "hoomd/GPUPolymorph.cuh"

template mpcd::BlockForce* hoomd::gpu::device_new(Scalar,Scalar,Scalar);
template mpcd::ConstantForce* hoomd::gpu::device_new(Scalar3);
template mpcd::SineForce* hoomd::gpu::device_new(Scalar,Scalar);
template void hoomd::gpu::device_delete(mpcd::ExternalField*);

#include "hoomd/mpcd/ConfinedStreamingMethodGPU.cuh"
#include "SinusoidalExpansionConstrictionGeometry.h"
#include "SinusoidalChannelGeometry.h"

//! Template instantiation of symmetric cosine geometry streaming
template cudaError_t mpcd::gpu::confined_stream<azplugins::detail::SinusoidalExpansionConstriction>
    (const mpcd::gpu::stream_args_t& args, const azplugins::detail::SinusoidalExpansionConstriction& geom);

template cudaError_t mpcd::gpu::confined_stream<azplugins::detail::SinusoidalChannel>
        (const mpcd::gpu::stream_args_t& args, const azplugins::detail::SinusoidalChannel& geom);
