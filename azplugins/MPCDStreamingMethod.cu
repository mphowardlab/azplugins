
#include "hoomd/mpcd/ConfinedStreamingMethodGPU.cuh"
#include "hoomd/GPUPolymorph.cuh"
#include "MPCDSineGeometry.h"


//! Template instantiation of sine geometry streaming
template cudaError_t mpcd::gpu::confined_stream<azplugins::detail::SineGeometry>
    (const mpcd::gpu::stream_args_t& args, const azplugins::detail::SineGeometry& geom);

