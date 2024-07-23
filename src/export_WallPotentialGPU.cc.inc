// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

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
namespace azplugins
    {
namespace detail
    {

void EXPORT_FUNCTION(pybind11::module& m)
    {
    hoomd::md::detail::export_PotentialExternalGPU<EVALUATOR_CLASS>(m,
                                                                    "WallPotential@_evaluator@GPU");
    }

    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd
