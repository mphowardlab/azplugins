// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

/*!
 * \file ImplicitPlaneEvaporatorGPU.h
 * \brief Declaration of ImplicitPlaneEvaporatorGPU
 */

#ifndef AZPLUGINS_IMPLICIT_PLANE_EVAPORATOR_GPU_H_
#define AZPLUGINS_IMPLICIT_PLANE_EVAPORATOR_GPU_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "ImplicitEvaporatorGPU.h"

namespace azplugins
{

//! Implicit solvent evaporator in a planar (thin film) geometry (on the GPU)
class PYBIND11_EXPORT ImplicitPlaneEvaporatorGPU : public ImplicitEvaporatorGPU
    {
    public:
        //! Constructor
        ImplicitPlaneEvaporatorGPU(std::shared_ptr<SystemDefinition> sysdef,
                                   std::shared_ptr<Variant> interf);

        //! Destructor
        virtual ~ImplicitPlaneEvaporatorGPU();

    protected:
        //! Implements the force calculation
        virtual void computeForces(unsigned int timestep);
    };

namespace detail
{
//! Exports the ImplicitPlaneEvaporatorGPU to python
void export_ImplicitPlaneEvaporatorGPU(pybind11::module& m);
} // end namespace detail

} // end namespace azplugins

#endif // AZPLUGINS_IMPLICIT_PLANE_EVAPORATOR_GPU_H_
