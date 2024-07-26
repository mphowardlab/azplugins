// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file ImplicitDropletEvaporatorGPU.h
 * \brief Declaration of ImplicitDropletEvaporatorGPU
 */

#ifndef AZPLUGINS_IMPLICIT_DROPLET_EVAPORATOR_GPU_H_
#define AZPLUGINS_IMPLICIT_DROPLET_EVAPORATOR_GPU_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "ImplicitEvaporatorGPU.h"

namespace azplugins
    {

//! Implicit solvent evaporator in a spherical (droplet) geometry (on the GPU)
class PYBIND11_EXPORT ImplicitDropletEvaporatorGPU : public ImplicitEvaporatorGPU
    {
    public:
    //! Constructor
    ImplicitDropletEvaporatorGPU(std::shared_ptr<SystemDefinition> sysdef,
                                 std::shared_ptr<Variant> interf);

    //! Destructor
    virtual ~ImplicitDropletEvaporatorGPU();

    protected:
    //! Implements the force calculation
    virtual void computeForces(unsigned int timestep);
    };

namespace detail
    {
//! Exports the ImplicitDropletEvaporatorGPU to python
void export_ImplicitDropletEvaporatorGPU(pybind11::module& m);
    } // end namespace detail

    } // end namespace azplugins

#endif // AZPLUGINS_IMPLICIT_DROPLET_EVAPORATOR_GPU_H_
