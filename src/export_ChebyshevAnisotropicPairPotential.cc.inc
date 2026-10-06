// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

// Adapted from hoomd/md/export_PotentialPair.cc.inc of HOOMD-blue.
// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// See CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off
#include "ChebyshevAnisotropicPairPotential.h"

#define SYMMETRY_CLASS ShapeSymmetry@_symmetry@
#define EXPORT_FUNCTION export_ChebyshevAnisotropicPairPotential@_symmetry@
// clang-format on

namespace hoomd
    {
namespace azplugins
    {
namespace detail
    {
void EXPORT_FUNCTION(pybind11::module& m)
    {
    export_ChebyshevAnisotropicPairPotential<SYMMETRY_CLASS>(
        m,
        "ChebyshevAnisotropicPairPotential@_symmetry@");
    }
    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd
