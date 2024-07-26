// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file ImplicitDropletEvaporatorGPU.cu
 * \brief Definition of kernel drivers and kernels for ImplicitDropletEvaporatorGPU
 */

#include "ImplicitDropletEvaporatorGPU.cuh"

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
__global__ void compute_implicit_evap_droplet_force(Scalar4* d_force,
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
    const Scalar3 pos_i = make_scalar3(postype_i.x, postype_i.y, postype_i.z);
    const unsigned int type_i = __scalar_as_int(postype_i.w);

    const Scalar4 params = s_params[type_i];
    const Scalar k = params.x;
    const Scalar offset = params.y;
    const Scalar g = params.z;
    const Scalar cutoff = params.w;
    // exit if interaction is off
    if (cutoff < Scalar(0.0))
        return;

    // get distances and direction of force
    const Scalar r_i = fast::sqrt(dot(pos_i, pos_i));
    const Scalar dr = r_i - (interf_origin + offset);
    if (!(r_i > Scalar(0.0)) || dr < Scalar(0.0))
        return;
    const Scalar3 rhat = pos_i / r_i;

    Scalar3 f;
    Scalar e;
    if (dr < cutoff) // harmonic
        {
        f = -k * dr * rhat;
        e = Scalar(0.5) * k * (dr * dr); // (k/2) dr^2
        }
    else // linear
        {
        f = -g * rhat;
        e = Scalar(0.5) * k * cutoff * cutoff + g * (dr - cutoff);
        }

    d_force[idx] = make_scalar4(f.x, f.y, f.z, e);
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
cudaError_t compute_implicit_evap_droplet_force(Scalar4* d_force,
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

    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)kernel::compute_implicit_evap_droplet_force);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);
    unsigned int shared_size = sizeof(Scalar4) * ntypes;

    dim3 grid(N / run_block_size + 1);
    kernel::compute_implicit_evap_droplet_force<<<grid, run_block_size, shared_size>>>(
        d_force,
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
