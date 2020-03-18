// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file BounceBackGeometry.h
 * \brief Declaration of valid bounce back geometries.
 *
 * This file is empty because all used geometries are implemented in hoomd v2.6.0
 * Users can add custom geometries here.
 */


#ifndef AZPLUGINS_BOUNCE_BACK_GEOMETRY_H_
#define AZPLUGINS_BOUNCE_BACK_GEOMETRY_H_

#include "hoomd/mpcd/BoundaryCondition.h"
#include "hoomd/mpcd/SlitGeometry.h"
#include "hoomd/mpcd/BoundaryCondition.h"
#include "MPCDSymCosGeometry.h"

#ifndef NVCC
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

namespace azplugins
{
namespace detail
{

//! Export SymCosGeometry to python
void export_SymCosGeometry(pybind11::module& m);

} // end namespace detail
} // end namespace azplugins

#endif // NVCC
#endif // MPCD_STREAMING_GEOMETRY_H_
