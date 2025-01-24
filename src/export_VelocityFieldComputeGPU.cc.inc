// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

// See CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off
#include "VelocityFieldComputeGPU.h"
#include "@_geometry@BinningOperation.h"

#define BINNING_OPERATION @_geometry@BinningOperation
#define EXPORT_FUNCTION export_@_geometry@VelocityFieldComputeGPU
// clang-format on

namespace hoomd
    {
namespace azplugins
    {

template class VelocityFieldComputeGPU<BINNING_OPERATION>;

namespace detail
    {
void EXPORT_FUNCTION(pybind11::module& m)
    {
    export_VelocityFieldComputeGPU<BINNING_OPERATION>(m, "@_geometry@VelocityFieldComputeGPU");
    }
    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd
