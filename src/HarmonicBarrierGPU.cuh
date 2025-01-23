// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_HARMONIC_BARRIER_GPU_CUH_
#define AZPLUGINS_HARMONIC_BARRIER_GPU_CUH_

#include "hoomd/HOOMDMath.h"
#include <cuda_runtime.h>

namespace hoomd
    {
namespace azplugins
    {
namespace gpu
    {

//! Kernel driver to evaluate PlanarHarmonicBarrierGPU force
template<class BarrierEvaluatorT>
cudaError_t compute_harmonic_barrier(Scalar4* d_force,
                                     Scalar* d_virial,
                                     const Scalar4* d_pos,
                                     const Scalar2* d_params,
                                     const BarrierEvaluatorT& evaluator,
                                     const unsigned int N,
                                     const unsigned int ntypes,
                                     const unsigned int block_size);

#ifdef __HIPCC__
namespace kernel
    {

/*!
 * \param d_force Particle forces
 * \param d_pos Particle positions
 * \param d_params Per-type parameters
 * \param evaluator Barrier evaluator
 * \param N Number of particles
 * \param ntypes Number of types
 *
 * Using one thread per particle, the force of the harmonic potential is computed
 * per-particle. The per-particle-type parameters are cached into shared memory.
 * This method does not compute the virial.
 *
 */
template<class BarrierEvaluatorT>
__global__ void compute_harmonic_barrier(Scalar4* d_force,
                                         const Scalar4* d_pos,
                                         const Scalar2* d_params,
                                         const BarrierEvaluatorT evaluator,
                                         const unsigned int N,
                                         const unsigned int ntypes)
    {
    // load per-type parameters into shared memory
    extern __shared__ Scalar2 s_params[];
    for (unsigned int cur_offset = 0; cur_offset < ntypes; cur_offset += blockDim.x)
        {
        if (cur_offset + threadIdx.x < ntypes)
            {
            s_params[cur_offset + threadIdx.x] = d_params[cur_offset + threadIdx.x];
            }
        }
    __syncthreads();

    // one thread per particle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    const Scalar4 postype = d_pos[idx];
    const Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    const unsigned int type = __scalar_as_int(postype.w);

    const Scalar2 params = s_params[type];

    d_force[idx] = evaluator(pos, params.x, params.y);
    }
    } // end namespace kernel

/*!
 * \param d_force Particle forces
 * \param d_virial Particle virial
 * \param d_pos Particle positions
 * \param d_params Per-type parameters
 * \param evaluator Barrier evaluator
 * \param N Number of particles
 * \param ntypes Number of types
 * \param block_size Number of threads per block
 *
 * The virial contribution is set to zero.
 */
template<class BarrierEvaluatorT>
cudaError_t compute_harmonic_barrier(Scalar4* d_force,
                                     Scalar* d_virial,
                                     const Scalar4* d_pos,
                                     const Scalar2* d_params,
                                     const BarrierEvaluatorT& evaluator,
                                     const unsigned int N,
                                     const unsigned int ntypes,
                                     const unsigned int block_size)
    {
    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)kernel::compute_harmonic_barrier<BarrierEvaluatorT>);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);
    unsigned int shared_size = sizeof(Scalar2) * ntypes;

    dim3 grid(N / run_block_size + 1);
    kernel::compute_harmonic_barrier<<<grid, run_block_size, shared_size>>>(d_force,
                                                                            d_pos,
                                                                            d_params,
                                                                            evaluator,
                                                                            N,
                                                                            ntypes);

    // zero the virial
    cudaMemset(d_virial, 0, 6 * sizeof(Scalar) * N);

    return cudaSuccess;
    }

#endif // __HIPCC__

    } // end namespace gpu
    } // end namespace azplugins
    } // end namespace hoomd

#endif // AZPLUGINS_HARMONIC_BARRIER_GPU_CUH_
