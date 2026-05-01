// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt

/*!
 * \file DynamicBondUpdaterGPU.cuh
 * \brief Declaration of kernel drivers for DynamicBondUpdaterGPU
 */

#ifndef AZPLUGINS_DYNAMIC_BOND_UPDATER_GPU_CUH_
#define AZPLUGINS_DYNAMIC_BOND_UPDATER_GPU_CUH_

#include <cuda_runtime.h>
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/BoxDim.h"
#include "hoomd/ParticleData.cuh"
#include <iostream>

#include "hoomd/md/NeighborListGPUTree.cuh"

#ifdef NVCC
#define DEVICE __device__ __forceinline__
#define HOSTDEVICE __host__ __device__ __forceinline__
#else
#define DEVICE
#define HOSTDEVICE
#endif


namespace hoomd
{
namespace azplugins
{
namespace gpu
{

cudaError_t sort_possible_bond_array(Scalar3 *d_all_possible_bonds,
                                     const unsigned int size);

cudaError_t filter_existing_bonds(Scalar3 *d_all_possible_bonds,
                                  unsigned int *d_n_existing_bonds,
                                  const unsigned int *d_existing_bonds_list,
                                  const Index2D& exli,
                                  const unsigned int size,
                                  const unsigned int block_size);

cudaError_t copy_possible_bonds(Scalar3 *d_all_possible_bonds,
                                const Scalar4 *d_postype,
                                const unsigned int *d_tag,
                                const unsigned int *d_sorted_indexes,
                                const unsigned int *d_n_neigh,
                                const unsigned int *d_nlist,
                                const BoxDim box,
                                const unsigned int max_bonds,
                                const  Scalar r_cut,
                                const bool groups_identical,
                                const unsigned int N,
                                const unsigned int block_size);

cudaError_t remove_zeros_and_sort_possible_bond_array(Scalar3 *d_all_possible_bonds,
                                                      const unsigned int size,
                                                      int *d_max_non_zero_bonds);



//! Sentinel for an invalid particle (e.g., ghost)
const unsigned int NeighborListTypeSentinel = 0xffffffff;


} // end namespace gpu
} // end namespace azplugins
} // end namespace hoomd


#undef DEVICE
#undef HOSTDEVICE


#endif // AZPLUGINS_DYNAMIC_BOND_UPDATER_GPU_CUH_
