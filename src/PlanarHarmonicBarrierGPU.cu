// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file PlanarHarmonicBarrierGPU.cu
 * \brief Definition of kernel drivers and kernels for PlanarHarmonicBarrierGPU
 */

#include "PlanarHarmonicBarrierGPU.cuh"

namespace hoomd
    {
namespace azplugins
    {
namespace gpu
    {
namespace kernel
    {

/*!
 * \param d_force Particle forces
 * \param d_virial Particle virial
 * \param d_pos Particle positions
 * \param d_params Per-type parameters
 * \param interf_origin Position of interface origin
 * \param N Number of particles
 * \param ntypes Number of types
 *
 * Using one thread per particle, the force of the harmonic potential is computed
 * per-particle. The per-particle-type parameters are cached into shared memory.
 * This method does not compute the virial.
 *
 */
__global__ void compute_implicit_evap_force(Scalar4* d_force,
                                            Scalar* d_virial,
                                            const Scalar4* d_pos,
                                            const Scalar4* d_params,
                                            const Scalar interf_origin,
                                            const unsigned int N,
                                            const unsigned int ntypes)
    {
    // load per-type parameters into shared memory
    extern __shared__ Scalar4 s_params[];
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

    const Scalar4 postype_i = d_pos[idx];
    const Scalar z_i = postype_i.z;
    const unsigned int type_i = __scalar_as_int(postype_i.w);

    const Scalar4 params = s_params[type_i];
    const Scalar k = params.x;
    const Scalar offset = params.y;
    const Scalar g = params.z;
    const Scalar cutoff = params.w;

    const Scalar dz = z_i - (interf_origin + offset);
    if (cutoff < Scalar(0.0) || dz < Scalar(0.0))
        return;

    Scalar fz(0.0), e(0.0);
    if (dz < cutoff) // harmonic
        {
        fz = -k * dz;
        e = Scalar(-0.5) * fz * dz; // (k/2) dz^2
        }
    else // linear
        {
        fz = -g;
        e = Scalar(0.5) * k * cutoff * cutoff + g * (dz - cutoff);
        }

    d_force[idx] = make_scalar4(0.0, 0.0, fz, e);
    }
    } // end namespace kernel

/*!
 * \param d_force Particle forces
 * \param d_virial Particle virial
 * \param d_pos Particle positions
 * \param d_params Per-type parameters
 * \param interf_origin Position of interface origin
 * \param N Number of particles
 * \param ntypes Number of types
 * \param block_size Number of threads per block
 *
 * This kernel driver is a wrapper around kernel::compute_implicit_evap_force.
 * The forces and virial are both set to zero before calculation.
 */
cudaError_t compute_implicit_evap_force(Scalar4* d_force,
                                        Scalar* d_virial,
                                        const Scalar4* d_pos,
                                        const Scalar4* d_params,
                                        const Scalar interf_origin,
                                        const unsigned int N,
                                        const unsigned int ntypes,
                                        const unsigned int block_size)
    {
    // zero the force and virial datasets before launch
    cudaMemset(d_force, 0, sizeof(Scalar4) * N);
    cudaMemset(d_virial, 0, 6 * sizeof(Scalar) * N);

    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)kernel::compute_implicit_evap_force);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);
    unsigned int shared_size = sizeof(Scalar4) * ntypes;

    dim3 grid(N / run_block_size + 1);
    kernel::compute_implicit_evap_force<<<grid, run_block_size, shared_size>>>(d_force,
                                                                               d_virial,
                                                                               d_pos,
                                                                               d_params,
                                                                               interf_origin,
                                                                               N,
                                                                               ntypes);
    return cudaSuccess;
    }

    } // end namespace gpu
    } // end namespace azplugins
    } // end namespace hoomd
