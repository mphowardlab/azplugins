// Copyright (c) 2018-2019, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file RDFAnalyzerGPU.cuh
 * \brief Declaration of kernel drivers for RDFAnalyzerGPU
 */

#ifndef AZPLUGINS_RDF_ANALYZER_GPU_CUH_
#define AZPLUGINS_RDF_ANALYZER_GPU_CUH_

#include <cuda_runtime.h>
#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"

namespace azplugins
{
namespace gpu
{

//! Kernel driver to bin particles for RDF
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
                            const unsigned int block_size);

//! Kernel driver to accumulate particles for RDF
cudaError_t analyze_rdf_accumulate(double *d_accum_rdf,
                                   unsigned int& num_samples,
                                   const unsigned int *d_counts,
                                   const unsigned int N_1,
                                   const unsigned int N_2,
                                   const Scalar box_volume,
                                   const unsigned int num_duplicates,
                                   const unsigned int num_bins,
                                   const Scalar rcut,
                                   const Scalar bin_width,
                                   const unsigned int block_size);

} // end namespace gpu
} // end namespace azplugins

#endif // AZPLUGINS_RDF_ANALYZER_GPU_CUH_
