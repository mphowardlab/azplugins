// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2022, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

/*!
 * \file ImplicitEvaporatorGPU.cu
 * \brief Definition of kernel drivers and kernels for ImplicitEvaporatorGPU
 */

#include "RDFAnalyzerGPU.cuh"

namespace azplugins
{
namespace gpu
{
namespace kernel
{

//! Bins pair distances for particles in two groups
/*!
 * \param d_counts Counts of particles in each bin (output)
 * \param d_duplicates Number of duplicate particles between groups (output)
 * \param d_group_1 Particle indexes for group 1
 * \param d_group_2 Particle indexes for group 2
 * \param d_pos Particle positions
 * \param N_1 Number of particles in group 1
 * \param N_2 Number of particles in group 2
 * \param box Simulation box
 * \param rcutsq Cutoff radius squared for RDF
 * \param bin_width Bin width for RDF
 *
 * \b Implementation:
 * Using one thread per particle pair, an all-pairs distance check is evaluated. If
 * The particle indexes in the pair are identical, the distance of 0 is not recorded,
 * and a duplicate is counted by atomic increment. The bin counter is also incremented
 * atomically.
 */
__global__ void analyze_rdf_bin(unsigned int *d_counts,
                                unsigned int *d_duplicates,
                                const unsigned int *d_group_1,
                                const unsigned int *d_group_2,
                                const Scalar4 *d_pos,
                                const unsigned int N_1,
                                const unsigned int N_2,
                                const BoxDim box,
                                const Scalar rcutsq,
                                const Scalar bin_width)
    {
    // one thread per particle pair
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N_1 * N_2)
        return;

    // convert 1d index to 2d
    const unsigned int i = tid / N_2;
    const unsigned int j = tid % N_2;

    // load particle indexes
    const unsigned int idx_i = d_group_1[i];
    const unsigned int idx_j = d_group_2[j];

    // exit on duplicates
    if (idx_i == idx_j)
        {
        atomicInc(d_duplicates, 0xffffffff);
        return;
        }

    const Scalar4 postype_i = d_pos[idx_i];
    const Scalar3 r_i = make_scalar3(postype_i.x, postype_i.y, postype_i.z);
    const Scalar4 postype_j = d_pos[idx_j];
    const Scalar3 r_j = make_scalar3(postype_j.x, postype_j.y, postype_j.z);

    // distance calculation
    const Scalar3 dr = box.minImage(r_j - r_i);
    const Scalar drsq = dot(dr, dr);
    if (drsq < rcutsq)
        {
        const unsigned int bin = floor(sqrt(drsq) / bin_width);
        atomicInc(&d_counts[bin], 0xffffffff);
        }
    }

/*!
 * \param d_accum_rdf Accumulated RDF (output)
 * \param d_counts Number of counts to add in
 * \param num_bins Total number of bins in RDF
 * \param rcut RDF cutoff radius
 * \param bin_width RDF bin width
 * \param prefactor Prefactor for scaling RDF counts
 *
 * Using one thread per bin, the bin volume is computed to generate
 * the instantaneous value of the RDF in the bin from \a d_counts.
 * This is added into the accumulated RDF, which will be subsequently
 * averaged when it is read.
 */
__global__ void analyze_rdf_accumulate(double *d_accum_rdf,
                                       const unsigned int *d_counts,
                                       const unsigned int num_bins,
                                       const Scalar rcut,
                                       const Scalar bin_width,
                                       const double prefactor)
    {
    // one thread per bin
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_bins)
        return;

    const double r_in = bin_width * static_cast<double>(idx);
    const double r_out = min(r_in + bin_width, static_cast<double>(rcut));
    const double V_shell = (4.0*M_PI/3.0) * (r_out * r_out * r_out - r_in * r_in * r_in);
    d_accum_rdf[idx] += static_cast<double>(d_counts[idx]) * prefactor / V_shell;
    }
} // end namespace kernel

/*!
 * \param d_counts Counts of particles in each bin (output)
 * \param d_duplicates Number of duplicate particles between groups (output)
 * \param d_group_1 Particle indexes for group 1
 * \param d_group_2 Particle indexes for group 2
 * \param d_pos Particle positions
 * \param N_1 Number of particles in group 1
 * \param N_2 Number of particles in group 2
 * \param box Simulation box
 * \param rcutsq Cutoff radius squared for RDF
 * \param bin_width Bin width for RDF
 * \param block_size Number of threads per block
 *
 * This is a thin wrapper to call azplugins::gpu::kernel:analyze_rdf_bin .
 */
cudaError_t analyze_rdf_bin(unsigned int *d_counts,
                            unsigned int *d_duplicates,
                            const unsigned int *d_group_1,
                            const unsigned int *d_group_2,
                            const Scalar4 *d_pos,
                            const unsigned int N_1,
                            const unsigned int N_2,
                            const BoxDim& box,
                            const unsigned int num_bins,
                            const Scalar rcutsq,
                            const Scalar bin_width,
                            const unsigned int block_size)
    {
    // zero the force and virial datasets before launch
    cudaMemset(d_counts, 0, sizeof(unsigned int)*num_bins);

    // don't do anything after memset with an empty selection
    if (N_1 == 0 || N_2 == 0) return cudaSuccess;

    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)kernel::analyze_rdf_bin);
        max_block_size = attr.maxThreadsPerBlock;
        }

    const unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(N_1*N_2 / run_block_size + 1);
    kernel::analyze_rdf_bin<<<grid, run_block_size>>>(d_counts,
                                                  d_duplicates,
                                                  d_group_1,
                                                  d_group_2,
                                                  d_pos,
                                                  N_1,
                                                  N_2,
                                                  box,
                                                  rcutsq,
                                                  bin_width);

    return cudaSuccess;
    }
/*!
 * \param d_accum_rdf Accumulated RDF (output)
 * \param num_samples Number of samples accumulated in the RDF (output)
 * \param d_counts Number of counts to add in
 * \param N_1 Number of particles in group 1
 * \param N_2 Number of particles in group 2
 * \param box_volume Volume of the simulation box
 * \param num_duplicates Number of duplicates in last evaluation of RDF
 * \param num_bins Total number of bins in RDF
 * \param rcut RDF cutoff radius
 * \param bin_width RDF bin width
 * \param block_size Number of threads per block
 *
 * The necessary prefactor for the RDF normalization is precomputed before
 * the kernel launch since it is shared between all threads. The number of
 * samples (\a num_samples) is incremented after the accumulation.
 *
 * \sa azplugins::gpu::kernel::analyze_rdf_accumulate
 */
cudaError_t analyze_rdf_accumulate(double *d_accum_rdf,
                                   unsigned int &num_samples,
                                   const unsigned int *d_counts,
                                   const unsigned int N_1,
                                   const unsigned int N_2,
                                   const Scalar box_volume,
                                   const unsigned int num_duplicates,
                                   const unsigned int num_bins,
                                   const Scalar rcut,
                                   const Scalar bin_width,
                                   const unsigned int block_size)
    {
    // don't do anything if there are no bins
    if (num_bins == 0) return cudaSuccess;

    // ensure that accumulated RDF is zero if there are no samples
    if (num_samples == 0)
        {
        cudaMemset(d_accum_rdf, 0, sizeof(double) * num_bins);
        }

    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)kernel::analyze_rdf_accumulate);
        max_block_size = attr.maxThreadsPerBlock;
        }

    // preprocess a prefactor for the calculation
    const double prefactor = box_volume / static_cast<double>(N_1 * N_2 - num_duplicates);

    const unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(num_bins / run_block_size + 1);
    kernel::analyze_rdf_accumulate<<<grid, run_block_size>>>(d_accum_rdf,
                                                             d_counts,
                                                             num_bins,
                                                             rcut,
                                                             bin_width,
                                                             prefactor);

    ++num_samples;

    return cudaSuccess;
    }

} // end namespace gpu
} // end namespace azplugins
