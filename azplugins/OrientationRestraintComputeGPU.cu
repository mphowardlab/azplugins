// Copyright (c) 2016-2018, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: wes_reinhart

/*!
 * \file OrientationRestraintComputeGPU.cu
 * \brief Defines CUDA kernels for OrientationRestraintComputeGPU
 */

#include "OrientationRestraintComputeGPU.cuh"

namespace azplugins
{
namespace gpu
{
namespace kernel
{
/*!
 * \param d_force Particle forces
 * \param d_torque Particle torques
 * \param d_member_idx Indices of group members
 * \param d_orient Particle orientations
 * \param d_orient Particle reference orientations
 * \param d_tag Particle tags
 * \param k Field force constant
 * \param box Simulation box
 * \param N_mem Number of particles in the group
 *
 * Using one thread per particle, the potential and torque of the restraining field
 * is computed per-particle, relative to a reference orientation.
 *
 */
__global__ void compute_orientation_restraint(Scalar4 *d_force,
                                              Scalar4 *d_torque,
                                              const unsigned int *d_member_idx,
                                              const Scalar4 *d_orient,
                                              const Scalar4 *d_ref_orient,
                                              const unsigned int *d_tag,
                                              const Scalar k,
                                              const BoxDim box,
                                              const unsigned int N_mem)
    {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // one thread per particle
    if (idx >= N_mem)
        return;

    const unsigned int cur_p = d_member_idx[idx];
    const Scalar4 cur_orient = d_orient[cur_p];

    const unsigned int cur_tag = d_tag[cur_p];
    const Scalar4 cur_ref_orient = d_ref_orient[cur_tag];

    // convert patch vector in the body frame of each particle to space frame
    vec3<Scalar> n_i = rotate(quat<Scalar>(cur_orient), vec3<Scalar>(1.0, 0, 0));
    vec3<Scalar> n_ref = rotate(quat<Scalar>(cur_ref_orient), vec3<Scalar>(1.0, 0, 0));

    // compute dot product between current and initial orientation
    Scalar orient_dot = dot(n_i, n_ref);

    // compute energy
    // U = k * sin(theta)^2
    //   = k * [ 1 - cos(theta)^2 ]
    //   = k * [ 1 - (n_i \dot n_ref)^2 ]
    Scalar energy = k * ( Scalar(1.0) - orient_dot * orient_dot );

    // compute torque
    // T = -dU/d(n_i \dot n_ref) * (n_i x n_ref)
    //   = -k * [ 1 - 2 (n_i \dot n_ref) ] * (n_i x n_ref)
    // const Scalar dUddot = ( Scalar(1.0) - Scalar(2.0) * orient_dot );
    const Scalar dUddot = Scalar(-2.0) * k * orient_dot;
    vec3<Scalar> torque_dir = cross(n_i,n_ref);
    const Scalar3 torque = vec_to_scalar3(Scalar(-1.0) * dUddot * torque_dir );

    d_torque[cur_p] = make_scalar4(torque.x,
        torque.y,
        torque.z,
        0.0);

    d_force[cur_p] = make_scalar4(0.0, 0.0, 0.0, energy);

    }
} // end namespace kernel

/*!
 * \param d_force Particle forces
 * \param d_torque Particle torques
 * \param d_member_idx Indices of group members
 * \param d_orient Particle orientations
 * \param d_ref_orient Particle reference orientations
 * \param d_tag Particle tags
 * \param k Field force constant
 * \param box Simulation box
 * \param N Number of particles
 * \param N_mem Number of particles in the group
 * \param block_size Number of threads per block
 * \param compute_capability GPU compute capability
 *
 * This kernel driver is a wrapper around kernel::compute_orientation_restraint.
 * The forces and torques are both set to zero before calculation.
 *
 */
cudaError_t compute_orientation_restraint(Scalar4 *d_force,
                                          Scalar4 *d_torque,
                                          const unsigned int *d_member_idx,
                                          const Scalar4 *d_orient,
                                          const Scalar4 *d_ref_orient,
                                          const unsigned int *d_tag,
                                          const Scalar& k,
                                          const BoxDim& box,
                                          const unsigned int N,
                                          const unsigned int N_mem,
                                          const unsigned int block_size,
                                          const unsigned int compute_capability)
    {
    // asynchronous memset in the default stream will allow other simple hosts tasks to proceed before kernel launch
    cudaError_t error;
    error = cudaMemset(d_force, 0, sizeof(Scalar4)*N);
    error = cudaMemset(d_torque, 0, sizeof(Scalar4)*N);

    if (error != cudaSuccess)
        return error;

    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)kernel::compute_orientation_restraint);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);

    dim3 grid(N_mem / run_block_size + 1);

    kernel::compute_orientation_restraint<<<grid, run_block_size>>>(d_force,
                                                                    d_torque,
                                                                    d_member_idx,
                                                                    d_orient,
                                                                    d_ref_orient,
                                                                    d_tag,
                                                                    k,
                                                                    box,
                                                                    N_mem);
    return cudaSuccess;
    }

} // end namespace gpu
} // end namespace azplugins
