// Copyright (c) 2016-2017, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: wes_reinhart

/*!
 * \file PositionRestraintCompute.h
 * \brief Declares a class for computing position restraining forces
 */

#ifndef AZPLUGINS_POSITION_RESTRAINT_COMPUTE_H_
#define AZPLUGINS_POSITION_RESTRAINT_COMPUTE_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "hoomd/ForceCompute.h"
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

namespace azplugins
{
//! Adds a restraining force to groups of particles
/*!
 * PositionRestraintCompute computes a harmonic potential calculated based on the distance between the current and
 * initial positions of particles. This effectively allows particles to have nearly fixed position while still retaining
 * their degrees of freedom.
 *
 * \f[ V(\mathbf{r}) = \frac{1}{2} \mathbf{k} \mathbf{\Delta r} \mathbf{\Delta r}^T \f]
 *
 * The strength of the potential depends on a spring constant \f$\mathbf{k}\f$ so that the particle position can be
 * restrained in any of the three coordinate directions. If \f$\mathbf{k}\f$ is very large, the particle position is
 * essentially constrained. However, because the particles retain their degrees of freedom, shorter integration timesteps
 * must be taken for large \f$\mathbf{k}\f$ to maintain stability.
 *
 * The force is computed as
 *
 * \f[ F(\mathbf{r}) = - \mathbf{k} \mathbf{\Delta r}
 *
 */
class PositionRestraintCompute : public ForceCompute
    {
    public:
        //! Constructs the compute
        PositionRestraintCompute(std::shared_ptr<SystemDefinition> sysdef,
                                 std::shared_ptr<ParticleGroup> group);

        //! Destructor
        ~PositionRestraintCompute();

        //! Set the force constant to a new value
        /*!
         * \param kx Force constant in x-direction
         * \param ky Force constant in y-direction
         * \param kz Force constant in z-direction
         */
        void setForceConstant(Scalar kx, Scalar ky, Scalar kz)
            {
            m_k = make_scalar3(kx, ky, kz);
            }

        //! Sets the reference positions of particles to their current value
        void setInitialPositions();

        //! Sets the reference position of a particles to a supplied value
        void setPosition(unsigned int tag, Scalar4 &pos);

        //! Returns a list of log quantities this compute calculates
        virtual std::vector<std::string> getProvidedLogQuantities();

        //! Calculates the requested log value and returns it
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);

    protected:
        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);

        Scalar3 m_k; //!< Force constants

        std::shared_ptr<ParticleGroup> m_group;  //!< Group of particles to apply force to
        GPUArray<Scalar4> m_ref_pos;               //!< Reference positions of the particles stored by tag
        std::vector<std::string> m_logname_list;  //!< Cache all generated logged quantities names

        bool m_has_warned;  //!< Flag if a warning has been issued about the virial
    };

namespace detail
{
//! Exports the PositionRestraintCompute to python
void export_PositionRestraintCompute(pybind11::module& m);
} // end namespace detail
} // end namespace azplugins

#endif // AZPLUGINS_POSITION_RESTRAINT_COMPUTE_H_
