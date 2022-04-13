// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2022, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

/*!
 * \file ImplicitDropletEvaporator.h
 * \brief Declaration of ImplicitDropletEvaporator
 */

#ifndef AZPLUGINS_IMPLICIT_DROPLET_EVAPORATOR_H_
#define AZPLUGINS_IMPLICIT_DROPLET_EVAPORATOR_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "ImplicitEvaporator.h"

#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

namespace azplugins
{

//! Implicit solvent evaporator in a spherical (droplet) geometry
/*
 * The interface normal is that of a sphere, and its origin is (0,0,0).
 */
class PYBIND11_EXPORT ImplicitDropletEvaporator : public ImplicitEvaporator
    {
    public:
        //! Constructor
        ImplicitDropletEvaporator(std::shared_ptr<SystemDefinition> sysdef,
                                  std::shared_ptr<Variant> interf);

        virtual ~ImplicitDropletEvaporator();

    protected:
        //! Implements the force calculation
        virtual void computeForces(unsigned int timestep);
    };

namespace detail
{
//! Exports the ImplicitDropletEvaporator to python
void export_ImplicitDropletEvaporator(pybind11::module& m);
} // end namespace detail

} // end namespace azplugins

#endif // AZPLUGINS_IMPLICIT_DROPLET_EVAPORATOR_H_
