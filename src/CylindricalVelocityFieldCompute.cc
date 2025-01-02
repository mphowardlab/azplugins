// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#include "CylindricalBinningOperation.h"
#include "VelocityFieldCompute.h"

namespace hoomd
    {
namespace azplugins
    {

template class VelocityFieldCompute<CylindricalBinningOperation>;

namespace detail
    {
void export_CylindricalVelocityFieldCompute(pybind11::module& m)
    {
    export_VelocityFieldCompute<CylindricalBinningOperation>(m, "CylindricalVelocityFieldCompute");
    }
    } // end namespace detail

    } // namespace azplugins
    } // namespace hoomd
