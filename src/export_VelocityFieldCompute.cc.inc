// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

// See CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off
#include "VelocityFieldCompute.h"
#include "@_geometry@BinningOperation.h"

#define BINNING_OPERATION @_geometry@BinningOperation
#define EXPORT_FUNCTION export_@_geometry@VelocityFieldCompute
// clang-format on

namespace hoomd
    {
namespace azplugins
    {

template class VelocityFieldCompute<BINNING_OPERATION>;

namespace detail
    {
void EXPORT_FUNCTION(pybind11::module& m)
    {
    export_VelocityFieldCompute<BINNING_OPERATION>(m, "@_geometry@VelocityFieldCompute");
    }
    } // end namespace detail

    } // end namespace azplugins
    } // end namespace hoomd
