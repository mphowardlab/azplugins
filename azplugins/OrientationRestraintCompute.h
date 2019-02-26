// Copyright (c) 2018-2019, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: wes_reinhart

/*! \file OrientationRestraintCompute.h
    \brief Declares a class for computing orientation restraining forces
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef AZPLUGINS_ORIENTATION_RESTRAINT_COMPUTE_H_
#define AZPLUGINS_ORIENTATION_RESTRAINT_COMPUTE_H_

#include "hoomd/ForceCompute.h"

namespace azplugins
{
//! Adds a restraining force to groups of particles
/*!
 * OrientationRestraintCompute computes a symmetric restraining potential calculated based on the angle between the current and
 * initial orientations of particles. This effectively allows particles to have nearly fixed orientation while still retaining
 * their degrees of freedom.
 *
 * \f[ V(\theta) = k \sin^2(\theta) \f]
 *
 * where \f$\theta\f$ is the angle between the current and reference orientation.
 * The strength of the potential depends on a spring constant \f$k\f$ so that the particle orientation can be
 * restrained to varying degrees. If \f$k\f$ is very large, the particle orientation is
 * essentially constrained. However, because the particles retain their degrees of freedom, shorter integration timesteps
 * must be taken for large \f$k\f$ to maintain stability.
 *
 * The torque is computed as
 *
 * \f[ -\frac{dV}{d(\hat{n}_i \cdot \hat{n}_{i,\mathrm{ref}})} \left( \hat{n}_i \times \hat{n}_{i,\mathrm{ref}} \right) \f]
 * \f[ \mathbf{\tau}(\theta) = -k \left[ 1 - 2 \left( \hat{n}_i \cdot \hat{n}_{i,\mathrm{ref}} \right) \right] \left( \hat{n}_i \times \hat{n}_{i,\mathrm{ref}} \right) \f]
 *
 */
class PYBIND11_EXPORT OrientationRestraintCompute : public ForceCompute
    {
    public:
        //! Constructs the compute
        OrientationRestraintCompute(std::shared_ptr<SystemDefinition> sysdef,
                                    std::shared_ptr<ParticleGroup> group);

        //! Destructor
        ~OrientationRestraintCompute();

        //! Set the force constant to a new value
        /*!
         * \param k Force constant
         */
        void setForceConstant(Scalar k)
            {
            m_k = k;
            }

        //! Sets the reference orientations of particles to their current value
        void setInitialOrientations();

        //! Sets the reference orientation of a particles to a supplied value
        void setOrientation(unsigned int tag, Scalar4 &orient);

        //! Returns a list of log quantities this compute calculates
        virtual std::vector<std::string> getProvidedLogQuantities();

        //! Calculates the requested log value and returns it
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);

        //! Returns true if this ForceCompute requires anisotropic integration
        virtual bool isAnisotropic()
            {
            return true;
            }

    protected:
        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);

        Scalar m_k; //!< Force constant

        std::shared_ptr<ParticleGroup> m_group;    //!< Group of particles to apply force to
        GPUArray<Scalar4> m_ref_orient;            //!< Reference orientations of the particles stored by tag
        std::vector<std::string> m_logname_list; //!< Cache all generated logged quantities names
    };

namespace detail
{
//! Exports the OrientationRestraintCompute to python
void export_OrientationRestraintCompute(pybind11::module& m);
} // end namespace detail

} // end namespace azplugins

#endif // AZPLUGINS_ORIENTATION_RESTRAINT_COMPUTE_H_
