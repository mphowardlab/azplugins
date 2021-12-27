// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

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
#include "SinusoidalChannelGeometry.h"
#include "SinusoidalExpansionConstrictionGeometry.h"
#ifndef NVCC
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

namespace azplugins
{
namespace detail
{

//! Export SinusoidalChannel to python
void export_SinusoidalChannel(pybind11::module& m);

//! Export SinusoidalExpansionConstriction to python
void export_SinusoidalExpansionConstriction(pybind11::module& m);

} // end namespace detail
} // end namespace azplugins

#endif // NVCC
#endif // MPCD_STREAMING_GEOMETRY_H_
