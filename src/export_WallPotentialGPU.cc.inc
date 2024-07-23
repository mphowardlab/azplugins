// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// See md/CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off
#include "hoomd/md/PotentialExternalGPU.h"
#include "hoomd/md/EvaluatorWalls.h"

#include "WallEvaluator@_evaluator@.h"

#define EVALUATOR_CLASS hoomd::md::EvaluatorWalls<hoomd::azplugins::detail::WallEvaluator@_evaluator@>
#define EXPORT_FUNCTION export_PotentialWall@_evaluator@GPU

// clang-format on

namespace hoomd
    {

// Use CPU class from another compilation unit to reduce compile time and compiler memory usage.
// extern template class md::PotentialPair<azplugins::detail::EVALUATOR_CLASS>;
namespace azplugins
    {
namespace detail
    {

void EXPORT_FUNCTION(pybind11::module& m)
    {
    hoomd::md::detail::export_PotentialExternalGPU<EVALUATOR_CLASS>(m, "WallPotential@_evaluator@GPU");
    }


    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd
