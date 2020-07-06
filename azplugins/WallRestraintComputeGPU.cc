// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

#include "WallRestraintComputeGPU.h"

namespace azplugins
{
template class WallRestraintComputeGPU<PlaneWall>;
template class WallRestraintComputeGPU<CylinderWall>;
template class WallRestraintComputeGPU<SphereWall>;
} // end namespace azplugins
