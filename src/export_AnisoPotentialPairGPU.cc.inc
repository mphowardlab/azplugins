// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

// See CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off
#include "hoomd/md/AnisoPotentialPairGPU.h"
#include "AnisoPairEvaluator@_evaluator@.h"

#define EVALUATOR_CLASS AnisoPairEvaluator@_evaluator@
#define EXPORT_FUNCTION export_AnisoPotentialPair@_evaluator@GPU
// clang-format on

namespace hoomd
    {

// Use CPU class from another compilation unit to reduce compile time and compiler memory usage.
extern template class md::AnisoPotentialPair<azplugins::detail::EVALUATOR_CLASS>;

namespace azplugins
    {
namespace detail
    {
void EXPORT_FUNCTION(pybind11::module& m)
    {
    hoomd::md::detail::export_AnisoPotentialPairGPU<EVALUATOR_CLASS>(
        m,
        "AnisoPotentialPair@_evaluator@GPU");
    }
    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd
