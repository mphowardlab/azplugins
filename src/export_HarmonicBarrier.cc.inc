// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

// See CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off
#include "HarmonicBarrier.h"
#include "@_geometry@BarrierEvaluator.h"

#define BARRIER_EVALUATOR @_geometry@BarrierEvaluator
#define EXPORT_FUNCTION export_@_geometry@HarmonicBarrier
// clang-format on

namespace hoomd
    {
namespace azplugins
    {

template class HarmonicBarrier<BARRIER_EVALUATOR>;

namespace detail
    {
void EXPORT_FUNCTION(pybind11::module& m)
    {
    export_HarmonicBarrier<BARRIER_EVALUATOR>(m, "@_geometry@HarmonicBarrier");
    }
    } // end namespace detail

    } // end namespace azplugins
    } // end namespace hoomd
