// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*! \file PlaneRestraintCompute.h
 *  \brief Computes harmonic restraint forces relative to a plane
 */

#ifndef AZPLUGINS_PLANE_RESTRAINT_COMPUTE_H_
#define AZPLUGINS_PLANE_RESTRAINT_COMPUTE_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "hoomd/ForceCompute.h"
#include "hoomd/ParticleGroup.h"
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

namespace azplugins
{
//! Applies a harmonic force relative to a plane for a group of particles
/*
 * Particles are restrained to a plane defined by a point and a normal using
 * a harmonic potential that is a function of the distance \a d from this plane:
 *
 * \f[ V(d) = \frac{k}{2} d^2 \f]
 *
 * This restraint is implemented as a ForceCompute (and not an external potential)
 * because it acts on a ParticleGroup (and not all particles by type). The definition
 * of the plane could be simplified by using a PlaneWall object, but this seemed like
 * overkill for this easy implementation.
 */
class PYBIND11_EXPORT PlaneRestraintCompute : public ForceCompute
    {
    public:
        //! Constructs the compute
        PlaneRestraintCompute(std::shared_ptr<SystemDefinition> sysdef,
                              std::shared_ptr<ParticleGroup> group,
                              Scalar3 point,
                              Scalar3 direction,
                              Scalar k);

        //! Destructor
        virtual ~PlaneRestraintCompute();

        //! Get the point on the line
        Scalar3 getPoint() const
            {
            return m_o;
            }

        //! Set the point on the line
        void setPoint(const Scalar3& o)
            {
            m_o = o;
            }

        //! Get the plane normal
        Scalar3 getNormal() const
            {
            return m_n;
            }

        //! Set the plane normal
        void setNormal(const Scalar3& n)
            {
            // direction must be a unit vector
            m_n = n / slow::sqrt(dot(n,n));
            }

        //! Get the force constant
        Scalar getForceConstant() const
            {
            return m_k;
            }

        //! Set the force constant
        void setForceConstant(Scalar k)
            {
            m_k = k;
            }

    protected:
        std::shared_ptr<ParticleGroup> m_group; //!< Group to apply forces to

        Scalar3 m_o;    //!< Point in the plane
        Scalar3 m_n;    //!< Unit normal
        Scalar m_k;     //!< Spring constant

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

namespace detail
{
//! Exports the PlaneRestraintCompute to python
void export_PlaneRestraintCompute(pybind11::module& m);
} // end namespace detail
} // end namespace azplugins

#endif // AZPLUGINS_PLANE_RESTRAINT_COMPUTE_H_
