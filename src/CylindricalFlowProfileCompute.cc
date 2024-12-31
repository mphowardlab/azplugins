// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#include "CylindricalBinningOperation.h"
#include "FlowProfileCompute.h"

namespace hoomd
    {
namespace azplugins
    {

template class FlowProfileCompute<CylindricalBinningOperation>;

namespace detail
    {
void export_CylindricalFlowProfileCompute(pybind11::module& m)
    {
    export_FlowProfileCompute<CylindricalBinningOperation>(m, "CylindricalFlowProfileCompute");
    }
    } // end namespace detail

    } // namespace azplugins
    } // namespace hoomd
