// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

// Adapted from hoomd/md/export_PotentialPair.cc.inc of HOOMD-blue.
// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// See CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off
#include "hoomd/md/PotentialPair.h"
#include "PairEvaluator@_evaluator@.h"

#define EVALUATOR_CLASS PairEvaluator@_evaluator@
#define EXPORT_FUNCTION export_PotentialPair@_evaluator@
// clang-format on

namespace hoomd
    {
namespace azplugins
    {
namespace detail
    {
void EXPORT_FUNCTION(pybind11::module& m)
    {
    hoomd::md::detail::export_PotentialPair<EVALUATOR_CLASS>(m, "PotentialPair@_evaluator@");
    }
    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd
