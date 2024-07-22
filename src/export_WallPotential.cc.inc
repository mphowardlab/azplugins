// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// See md/CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off
#include "hoomd/md/PotentialExternal.h"
#include "WallEvaluatorLJ93.h"
#include "hoomd/md/EvaluatorWalls.h"

#define EVALUATOR_CLASS hoomd::md::EvaluatorWalls<hoomd::azplugins::detail::WallEvaluatorLJ93>
#define EXPORT_FUNCTION export_PotentialWallLJ93
// clang-format on

namespace hoomd
    {
namespace azplugins
    {
namespace detail
    {

void EXPORT_FUNCTION(pybind11::module& m)
    {
    hoomd::md::detail::export_PotentialExternal<EVALUATOR_CLASS>(m, "WallPotentialLJ93");
    }

    } // end namespace detail
    } // end namespace rusenplugins
    } // end namespace hoomd
