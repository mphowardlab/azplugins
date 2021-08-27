// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

/*!
 * \file ImplicitPlaneEvaporator.h
 * \brief Declaration of ImplicitPlaneEvaporator
 */

#ifndef AZPLUGINS_IMPLICIT_PLANE_EVAPORATOR_H_
#define AZPLUGINS_IMPLICIT_PLANE_EVAPORATOR_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "ImplicitEvaporator.h"

#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

namespace azplugins
{

//! Implicit solvent evaporator in a planar (thin film) geometry
/*!
 * The interface normal is defined along +z going from the liquid into the vapor phase,
 * and the origin is z = 0. This effectively models a drying thin film.
 */
class PYBIND11_EXPORT ImplicitPlaneEvaporator : public ImplicitEvaporator
    {
    public:
        //! Constructor
        ImplicitPlaneEvaporator(std::shared_ptr<SystemDefinition> sysdef,
                                std::shared_ptr<Variant> interf);

        virtual ~ImplicitPlaneEvaporator();

    protected:
        //! Implements the force calculation
        virtual void computeForces(unsigned int timestep);
    };

namespace detail
{
//! Exports the ImplicitPlaneEvaporator to python
void export_ImplicitPlaneEvaporator(pybind11::module& m);
} // end namespace detail

} // end namespace azplugins

#endif // AZPLUGINS_IMPLICIT_PLANE_EVAPORATOR_H_
