// Copyright (c) 2018-2019, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

#include "PlaneRestraintComputeGPU.cuh"

namespace azplugins
{
namespace gpu
{
namespace kernel
{
//! Kernel to compute plane restraints on the GPU
/*!
 * \param forces Forces on particles
 * \param virials Virial per particle
 * \param group Indexes of particles in the group
 * \param positions Particle positions
 * \param images Particle images
 * \param box Global simulation box
 * \param o Point in the plane
 * \param n Normal to the plane
 * \param k Spring constant
 * \param N Number of particles
 * \param virial_pitch Pitch of the virial
 *
 * One thread per particle. Computes the harmonic potential with spring force \a k
 * based on the distance of the particle in the \a group from the plane defined by
 * the point-normal form.
 */
__global__ void compute_plane_restraint(Scalar4* forces,
                                        Scalar* virials,
                                        const unsigned int* group,
                                        const Scalar4* positions,
                                        const int3* images,
                                        const BoxDim box,
                                        Scalar3 o,
                                        Scalar3 n,
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

    // distance to point from plane
    const Scalar3 dr = box.shift(r, image) - o;
    const Scalar d = dot(dr,n);

    // force points along normal vector
    const Scalar3 force = -k*(d*n);

    // squared-distance gives energy
    const Scalar energy = Scalar(0.5)*k*(d*d);

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
 * \param o Point in the plane
 * \param n Normal to the plane
 * \param k Spring constant
 * \param N Number of particles
 * \param virial_pitch Pitch of the virial
 * \param block_size Number of threads per block
 *
 * \sa kernel::compute_plane_restraint
 */
cudaError_t compute_plane_restraint(Scalar4* forces,
                                    Scalar* virials,
                                    const unsigned int* group,
                                    const Scalar4* positions,
                                    const int3* images,
                                    const BoxDim box,
                                    Scalar3 o,
                                    Scalar3 n,
                                    Scalar k,
                                    unsigned int N,
                                    unsigned int virial_pitch,
                                    unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)kernel::compute_plane_restraint);
        max_block_size = attr.maxThreadsPerBlock;
        }

    const unsigned int run_block_size = min(block_size, max_block_size);
    const unsigned int num_blocks = (N+run_block_size-1)/run_block_size;

    kernel::compute_plane_restraint<<<num_blocks,run_block_size>>>
        (forces, virials, group, positions, images, box, o, n, k, N, virial_pitch);

    return cudaSuccess;
    }
} // end namespace gpu
} // end namespace azplugins
