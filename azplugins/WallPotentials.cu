// Copyright (c) 2016, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward / Everyone is free to add additional potentials

/*!
 * \file WallPotentials.cu
 * \brief Defines the driver functions for computing wall forces on the GPU
 *
 * The driver function must have an explicit template instantiation for each
 * evaluator. The template must be in the global namespace that is used by hoomd.
 */

#include "WallPotentials.h"
#include "hoomd/md/EvaluatorWalls.h"
#include "hoomd/md/PotentialExternalGPU.cuh"

//! Evaluator for colloid (integrated Lennard-Jones) wall potential
template cudaError_t gpu_cpef< EvaluatorWalls<azplugins::detail::WallEvaluatorColloid> >
    (const external_potential_args_t& external_potential_args,
     const typename EvaluatorWalls<azplugins::detail::WallEvaluatorColloid>::param_type *d_params,
     const typename EvaluatorWalls<azplugins::detail::WallEvaluatorColloid>::field_type *d_field);

//! Evaluator for Lennard-Jones 9-3 wall potential
template cudaError_t gpu_cpef< EvaluatorWalls<azplugins::detail::WallEvaluatorLJ93> >
    (const external_potential_args_t& external_potential_args,
     const typename EvaluatorWalls<azplugins::detail::WallEvaluatorLJ93>::param_type *d_params,
     const typename EvaluatorWalls<azplugins::detail::WallEvaluatorLJ93>::field_type *d_field);
