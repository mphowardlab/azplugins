// Copyright (c) 2016, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file ImplicitEvaporator.h
 * \brief Declaration of ImplicitEvaporator
 */

#ifndef AZPLUGINS_IMPLICIT_EVAPORATOR_H_
#define AZPLUGINS_IMPLICIT_EVAPORATOR_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "hoomd/ForceCompute.h"
#include "hoomd/Variant.h"

#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

namespace azplugins
{

//! Adds a force modeling a moving liquid-vapor interface along the z axis
/*! \ingroup computes
 * The moving interface compute acts on particles along the z direction, with the interface normal defined along +z
 * going from the liquid into the vapor phase. The interface potential is harmonic. It does not include an attractive
 * part (i.e., it is truncated at its minimum, and is zero for any negative displacements relative to the minimum.
 * The position of the minimum can be adjusted with an offset, which controls an effective contact angle.
 * The potential is cutoff at a certain distance above the interface, at which point it switches to a linear potential
 * with a fixed force constant. This models the gravitational force experienced by particles once the interface has moved
 * past the particles. In practice, this cutoff should be roughly the particle radius.
 */
class ImplicitEvaporator : public ForceCompute
    {
    public:
        //! Constructs the compute
        ImplicitEvaporator(std::shared_ptr<SystemDefinition> sysdef,
                           std::shared_ptr<Variant> interf);

        //! Destructor
        ~ImplicitEvaporator();

        //! Set the per-type potential parameters
        /*!
         * \param type Particle type id
         * \param k Spring constant
         * \param offset Distance to shift potential minimum from interface
         * \param g Linear potential force constant
         * \param cutoff Distance from potential minimum to cutoff harmonic potential and switch to linear
         */
        void setParams(unsigned int type, Scalar k, Scalar offset, Scalar g, Scalar cutoff)
            {
            ArrayHandle<Scalar4> h_params(m_params, access_location::host, access_mode::readwrite);
            h_params.data[type] = make_scalar4(k, offset, g, cutoff);
            }

    protected:
        //! Implements the force calculation
        virtual void computeForces(unsigned int timestep);

        std::shared_ptr<Variant> m_interf;      //!< Variant for computing the current location of the interface

        GPUArray<Scalar4> m_params;             //!< Per-type array of parameters for the potential

    private:
        //! Reallocate the per-type parameter arrays when the number of types changes
        void reallocateParams()
            {
            m_params.resize(m_pdata->getNTypes());
            }
    };

namespace detail
{
//! Exports the ImplicitEvaporator to python
void export_ImplicitEvaporator(pybind11::module& m);
} // end namespace detail

} // end namespace azplugins

#endif // AZPLUGINS_IMPLICIT_EVAPORATOR_H_
