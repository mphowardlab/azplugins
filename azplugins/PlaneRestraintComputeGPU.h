// Copyright (c) 2018-2019, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*! \file PlaneRestraintComputeGPU.h
 *  \brief Computes harmonic restraint forces relative to a plane, on the GPU
 */

#ifndef AZPLUGINS_PLANE_RESTRAINT_COMPUTE_GPU_H_
#define AZPLUGINS_PLANE_RESTRAINT_COMPUTE_GPU_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "PlaneRestraintCompute.h"

#include "hoomd/Autotuner.h"
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

namespace azplugins
{
//! Applies a harmonic force relative to a plane for a group of particles on the GPU
class PYBIND11_EXPORT PlaneRestraintComputeGPU : public PlaneRestraintCompute
    {
    public:
        //! Constructs the compute
        PlaneRestraintComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                                 std::shared_ptr<ParticleGroup> group,
                                 Scalar3 point,
                                 Scalar3 direction,
                                 Scalar k);

    protected:
        //! Actually compute the forces on the GPU
        virtual void computeForces(unsigned int timestep);

    private:
        std::shared_ptr<Autotuner> m_tuner; //!< Tuner for force kernel
    };

namespace detail
{
//! Exports the PlaneRestraintComputeGPU to python
void export_PlaneRestraintComputeGPU(pybind11::module& m);
} // end namespace detail
} // end namespace azplugins

#endif // AZPLUGINS_PLANE_RESTRAINT_COMPUTE_GPU_H_
