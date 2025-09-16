// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#include "ImagePotentialBond.h"
#include "hoomd/md/EvaluatorBondHarmonic.h"
#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace azplugins
    {
namespace detail
    {
void export_ImageBondPotentialHarmonic(pybind11::module& m)
    {
    export_ImagePotentialBond<hoomd::md::EvaluatorBondHarmonic>(m, "ImageBondPotentialHarmonic");
    }
    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd
