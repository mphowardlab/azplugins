// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

// Adapted from hoomd/md/export_PotentialPairGPU.cc.inc of HOOMD-blue.
// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// See CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off
#include "hoomd/md/PotentialPairGPU.h"
#include "PairEvaluator@_evaluator@.h"

#define EVALUATOR_CLASS PairEvaluator@_evaluator@
#define EXPORT_FUNCTION export_PotentialPair@_evaluator@GPU
// clang-format on

namespace hoomd
    {

// Use CPU class from another compilation unit to reduce compile time and compiler memory usage.
extern template class md::PotentialPair<azplugins::detail::EVALUATOR_CLASS>;

namespace azplugins
    {
namespace detail
    {
void EXPORT_FUNCTION(pybind11::module& m)
    {
    hoomd::md::detail::export_PotentialPairGPU<EVALUATOR_CLASS>(m, "PotentialPair@_evaluator@GPU");
    }
    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd
