// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

#ifndef AZPLUGINS_WALL_RESTRAINT_COMPUTE_GPU_CUH_
#define AZPLUGINS_WALL_RESTRAINT_COMPUTE_GPU_CUH_

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include "hoomd/md/WallData.h"

#include <cuda_runtime.h>

namespace azplugins
{
namespace gpu
{
//! Kernel driver to compute wall restraints on the GPU
template<class T>
cudaError_t compute_wall_restraint(Scalar4* forces,
                                   Scalar* virials,
                                   const unsigned int* group,
                                   const Scalar4* positions,
                                   const int3* images,
                                   const BoxDim& box,
                                   const T& wall,
                                   Scalar k,
                                   unsigned int N,
                                   unsigned int virial_pitch,
                                   unsigned int block_size);

// only compile kernels in NVCC
#ifdef NVCC
namespace kernel
{
//! Kernel to compute wall restraints on the GPU
/*!
 * \param forces Forces on particles
 * \param virials Virial per particle
 * \param group Indexes of particles in the group
 * \param positions Particle positions
 * \param images Particle images
 * \param box Global simulation box
 * \param wall WallData object defining surface
 * \param k Spring constant
 * \param N Number of particles
 * \param virial_pitch Pitch of the virial
 *
 * \tparam T WallData object type
 *
 * One thread per particle. Computes the harmonic potential with spring force \a k
 * based on the distance of the particle in the \a group from the surface defined by
 * the \a wall.
 */
template<class T>
__global__ void compute_wall_restraint(Scalar4* forces,
                                       Scalar* virials,
                                       const unsigned int* group,
                                       const Scalar4* positions,
                                       const int3* images,
                                       const BoxDim box,
                                       const T wall,
                                       Scalar k,
                                       unsigned int N,
                                       unsigned int virial_pitch)
    {
    // one thread per particle
    const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    const unsigned int pidx = group[idx];

    // unwrapped particle coordinate
    const Scalar4 pos = positions[pidx];
    const int3 image = images[pidx];
    const Scalar3 r = make_scalar3(pos.x, pos.y, pos.z);

    // vector to point from surface (inside is required but not used by this potential)
    bool inside;
    const vec3<Scalar> dr = vecPtToWall(wall, vec3<Scalar>(box.shift(r, image)), inside);

    // force points along the point-to-wall vector (cancellation of minus signs)
    const Scalar3 force = vec_to_scalar3(k*dr);

    // squared-distance gives energy
    const Scalar energy = Scalar(0.5)*k*dot(dr,dr);

    // virial is dyadic product of force with position (in this box)
    Scalar virial[6];
    virial[0] = force.x * r.x;
    virial[1] = force.x * r.y;
    virial[2] = force.x * r.z;
    virial[3] = force.y * r.y;
    virial[4] = force.y * r.z;
    virial[5] = force.z * r.z;

    forces[pidx] = make_scalar4(force.x, force.y, force.z, energy);
    for (unsigned int j=0; j < 6; ++j)
        virials[virial_pitch*j+pidx] = virial[j];
    }
} // end namespace kernel

/*!
 * \param forces Forces on particles
 * \param virials Virial per particle
 * \param group Indexes of particles in the group
 * \param positions Particle positions
 * \param images Particle images
 * \param box Global simulation box
 * \param wall WallData object defining surface
 * \param k Spring constant
 * \param N Number of particles
 * \param virial_pitch Pitch of the virial
 * \param block_size Number of threads per block
 *
 * \tparam T WallData object type
 *
 * \sa kernel::compute_wall_restraint
 */
template<class T>
cudaError_t compute_wall_restraint(Scalar4* forces,
                                   Scalar* virials,
                                   const unsigned int* group,
                                   const Scalar4* positions,
                                   const int3* images,
                                   const BoxDim& box,
                                   const T& wall,
                                   Scalar k,
                                   unsigned int N,
                                   unsigned int virial_pitch,
                                   unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)kernel::compute_wall_restraint<T>);
        max_block_size = attr.maxThreadsPerBlock;
        }

    const unsigned int run_block_size = min(block_size, max_block_size);
    const unsigned int num_blocks = (N+run_block_size-1)/run_block_size;

    kernel::compute_wall_restraint<<<num_blocks,run_block_size>>>
        (forces, virials, group, positions, images, box, wall, k, N, virial_pitch);

    return cudaSuccess;
    }

#endif

} // end namespace gpu
} // end namespace azplugins

#endif // AZPLUGINS_WALL_RESTRAINT_COMPUTE_GPU_CUH_
