// Copyright (c) 2016-2017, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt

/*!
 * \file ReversePerturbationFlow.h
 * \brief Declaration of Reverse perturbation flow
 */

#ifndef AZPLUGINS_REVERSE_PERTURBATION_FLOW_H_
#define AZPLUGINS_REVERSE_PERTURBATION_FLOW_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "hoomd/Updater.h"
#include "hoomd/ParticleGroup.h"
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

namespace azplugins
{
//! Reverse perturbation flow updater
/*!
 * ReversePerturbationFlow implements reverse perturbation flow as
 * described in Müller-Plathe,F. Physical Review E 59.5 (1999): 4894.
 *
 * A flow is induced by swapping momentum in x direction based on particle position in z-direction.
 */
class ReversePerturbationFlow : public Updater
    {
    public:
        //! Constructor
        ReversePerturbationFlow(std::shared_ptr<SystemDefinition> sysdef,
                                std::shared_ptr<ParticleGroup> group,
                                unsigned int num_swap,
                                Scalar slab_width,
                                Scalar p_target);

        //! Destructor
        virtual ~ReversePerturbationFlow();

        //! Apply velocity swaps
        virtual void update(unsigned int timestep);

        //! Set the maximum number of swapped pairs
        void setNswap(unsigned int new_num_swap);

        //! Set the slab width
        void setSlabWidth(Scalar slab_width);

        //! Set group to operate on
        /*!
         * \param group Group for the updater to operate on
         */
        void setGroup(std::shared_ptr<ParticleGroup> group)
            {
            m_group = group;
            }

        //! Set target momentum
        /*!
         * \param p_target target momentum for selecting pair swaps
         */
        void setTargetMomentum(Scalar p_target)
            {
            m_p_target = p_target;
            }

        //! Get max number of swaps
        Scalar getNswap() const
            {
            return m_num_swap;
            }

        //! Get slab thickness
        Scalar getSlabWidth() const
            {
            return m_slab_width;
            }

        //! Get target momentum
        Scalar getTargetMomentum() const
            {
            return m_p_target;
            }

        //! Get group
        std::shared_ptr<ParticleGroup> getGroup() const
            {
            return m_group;
            }

        //! Computes the requested log value and returns it
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);

        //!Returns a list of log quantities this compute calculates
        virtual std::vector<std::string> getProvidedLogQuantities();

    protected:
        std::shared_ptr<ParticleGroup> m_group;//!< Group the updater is operating on
        unsigned int m_num_swap;     //!< maximum number of swaps
        Scalar m_slab_width;         //!< thickness of slabs
        Scalar m_p_target;           //!< target momentum to pick pairs closest to
        Scalar m_momentum_exchange;  //!< current momentum excange between slabs
        Scalar2 m_lo_pos;            //!< position of bottom slab in box
        Scalar2 m_hi_pos;            //!< position of top slab in box
        unsigned int m_num_lo;       //!< number of particles in bottom slab
        GPUArray<Scalar2> m_layer_lo;//!< List of all particles (indices,momentum) in bottom slab sorted by momentum
        unsigned int m_num_hi;       //!< number of particles in top slab
        GPUArray<Scalar2> m_layer_hi;//!< List of all particles (indices,momentum) in top slab sorted by momentum

        //! Find candidate particles for swapping in the slabs
        virtual void findSwapParticles();
        //! Swaps momentum between the slabs
        virtual void swapPairMomentum();

    private:
        bool m_update_slabs;    //!< If true, update the slab positions

        //! Request to check box on next update
        void requestUpdateSlabs()
            {
            m_update_slabs = true;
            }

        //! Sets the slab positions in the box and validates them
        void setSlabs();
    };

namespace detail
{
//! Export ReversePerturbationFlow to python
void export_ReversePerturbationFlow(pybind11::module& m);
} // end namespace detail
} // end namespace azplugins
#endif // AZPLUGINS_REVERSE_PERTURBATION_FLOW_H_
