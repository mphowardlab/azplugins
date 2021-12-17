// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

#include "AnisoPairPotentials.cuh"

namespace azplugins
{
namespace gpu
{

//! Kernel driver for Two-patch Morse anisotropic pair potential
template cudaError_t compute_aniso_pair_potential<azplugins::detail::AnisoPairEvaluatorTwoPatchMorse>
    (const a_pair_args_t& pair_args,
     const typename azplugins::detail::AnisoPairEvaluatorTwoPatchMorse::param_type*
     #ifdef HOOMD_MD_ANISO_SHAPE_PARAM
     , const typename azplugins::detail::AnisoPairEvaluatorTwoPatchMorse::shape_param_type*
     #endif
    );

} // end namespace gpu
} // end namespace azplugins
