// Copyright (c) 2018-2020, Michael P. Howard
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

cudaError_t sort_and_remove_zeros_possible_bond_array(Scalar3 *d_all_possible_bonds,
                                            const unsigned int size,
                                            int *d_max_non_zero_bonds);

cudaError_t remove_zeros_possible_bond_array(Scalar3 *d_all_possible_bonds,
                                             const unsigned int size,
                                             int *d_max_non_zero_bonds);

}
}

#endif // AZPLUGINS_DYNAMIC_BOND_UPDATER_GPU_CUH_
