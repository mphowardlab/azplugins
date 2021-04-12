// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

#include "WallRestraintComputeGPU.cuh"

namespace azplugins
{
namespace gpu
{
// PlaneWall template
template cudaError_t compute_wall_restraint<PlaneWall>(Scalar4*,
                                                       Scalar*,
                                                       const unsigned int*,
                                                       const Scalar4*,
                                                       const int3*,
                                                       const BoxDim&,
                                                       const PlaneWall&,
                                                       Scalar,
                                                       unsigned int,
                                                       unsigned int,
                                                       unsigned int);
// CylinderWall template
template cudaError_t compute_wall_restraint<CylinderWall>(Scalar4*,
                                                          Scalar*,
                                                          const unsigned int*,
                                                          const Scalar4*,
                                                          const int3*,
                                                          const BoxDim&,
                                                          const CylinderWall&,
                                                          Scalar,
                                                          unsigned int,
                                                          unsigned int,
                                                          unsigned int);
// SphereWall template
template cudaError_t compute_wall_restraint<SphereWall>(Scalar4*,
                                                        Scalar*,
                                                        const unsigned int*,
                                                        const Scalar4*,
                                                        const int3*,
                                                        const BoxDim&,
                                                        const SphereWall&,
                                                        Scalar,
                                                        unsigned int,
                                                        unsigned int,
                                                        unsigned int);
} // end namespace gpu
} // end namespace azplugins
