// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2022, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

/*!
 * \file MPCDReversePerturbationFlow.h
 * \brief Declaration of Reverse perturbation flow
 */

#ifndef AZPLUGINS_MPCD_REVERSE_PERTURBATION_FLOW_H_
#define AZPLUGINS_MPCD_REVERSE_PERTURBATION_FLOW_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "hoomd/Updater.h"
#include "hoomd/ParticleGroup.h"
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"
#include "hoomd/mpcd/SystemData.h"

namespace azplugins
{
//! Reverse perturbation flow updater
/*!
 * MPCDReversePerturbationFlow implements reverse pertubation flow as
 * described in Mueller-Plathe,F. Physical Review E 59.5 (1999): 4894.
 *
 * A flow is induced by swapping velocities in x direction based on particle position in z-direction.
 */
class PYBIND11_EXPORT MPCDReversePerturbationFlow : public Updater
    {
    public:
        //! Constructor
        MPCDReversePerturbationFlow(std::shared_ptr<mpcd::SystemData> sysdata,
                                    unsigned int num_swap,
                                    Scalar slab_width,
                                    Scalar slab_distance,
                                    Scalar p_target);

        //! Destructor
        virtual ~MPCDReversePerturbationFlow();

        //! Apply velocity swaps
        virtual void update(unsigned int timestep);

        //! Set the maximum number of swapped pairs
        void setNswap(unsigned int new_num_swap);

        //! Set the slab width
        void setSlabWidth(Scalar slab_width);

        //! Set the slab width
        void setSlabDistance(Scalar slab_distance);

        //! Set the target velocity
        void setTargetMomentum(Scalar p_target);

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


        //! Get slab thickness
        Scalar getSlabDistance() const
            {
            return m_slab_distance;
            }


        //! Get target momentum
        Scalar getTargetMomentum() const
            {
            return m_p_target;
            }

        //! Computes the requested log value and returns it
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);

        //!Returns a list of log quantities this compute calculates
        virtual std::vector<std::string> getProvidedLogQuantities();

    protected:
        std::shared_ptr<mpcd::SystemData> m_mpcd_sys;     //!< MPCD system data
        std::shared_ptr<mpcd::ParticleData> m_mpcd_pdata; //!< MPCD particle data
        unsigned int m_num_swap;     //!< maximum number of swaps
        Scalar m_slab_width;         //!< thickness of slabs
        Scalar m_slab_distance;        //!< distance between slabs
        Scalar m_momentum_exchange;  //!< current momentum excange between slabs
        Scalar2 m_lo_pos;            //!< position of bottom slab in box
        Scalar2 m_hi_pos;            //!< position of top slab in box
        unsigned int m_num_lo;       //!< number of particles in bottom slab
        GPUArray<Scalar2> m_layer_lo;//!< List of all particles (indices,momentum) in bottom slab sorted by momentum closest to +m_p_target
        unsigned int m_num_hi;       //!< number of particles in top slab
        GPUArray<Scalar2> m_layer_hi;//!< List of all particles (indices,momentum) in top slab sorted by momentum closest to -m_p_target
        Scalar m_p_target;           //!< target momentum for particles in the slabs
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
void export_MPCDReversePerturbationFlow(pybind11::module& m);
} // end namespace detail
} // end namespace azplugins
#endif // AZPLUGINS_REVERSE_PERTURBATION_FLOW_H_
