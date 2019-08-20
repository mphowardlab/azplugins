// Copyright (c) 2018-2019, Michael P. Howard
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

#if (HOOMD_VERSION_MAJOR >= 2) && (HOOMD_MINOR_VERSION >= 7)
#define AZPLUGINS_API_INTEGRATE_SLIT_PORE
#include "hoomd/mpcd/SlitPoreGeometry.h"
#endif

#ifndef NVCC
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

namespace azplugins
{
namespace detail
{


} // end namespace detail
} // end namespace azplugins

#endif // NVCC
#endif // MPCD_STREAMING_GEOMETRY_H_
