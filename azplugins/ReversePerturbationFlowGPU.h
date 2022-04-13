// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2022, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

/*!
 * \file ReversePerturbationFlowGPU.h
 * \brief Declaration of ReversePerturbationFlowGPU
 */

#ifndef AZPLUGINS_REVERSE_PERTURBATION_FLOW_GPU_H_
#define AZPLUGINS_REVERSE_PERTURBATION_FLOW_GPU_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "ReversePerturbationFlow.h"
#include "hoomd/Autotuner.h"
#include "hoomd/GPUFlags.h"

namespace azplugins
{
//! Particle type updater on the GPU
/*!
 * See ReversePerturbationFlow for details. This class inherits and minimally implements
 * the CPU methods from ReversePerturbationFlow on the GPU.
 */
class PYBIND11_EXPORT ReversePerturbationFlowGPU : public ReversePerturbationFlow
    {
    public:
        //! Constructor with parameters
        ReversePerturbationFlowGPU(std::shared_ptr<SystemDefinition> sysdef,
                                   std::shared_ptr<ParticleGroup> group,
                                   unsigned int num_swap,
                                   Scalar slab_width,
                                   Scalar p_target);

        //! Destructor
        virtual ~ReversePerturbationFlowGPU() {};

        /*! Set autotuner parameters
         * \param enable Enable / disable autotuning
         * \param period period (approximate) in time steps when retuning occurs
         */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            ReversePerturbationFlow::setAutotunerParams(enable, period);

            m_swap_tuner->setPeriod(period);
            m_swap_tuner->setEnabled(enable);

            m_mark_tuner->setPeriod(period);
            m_mark_tuner->setEnabled(enable);

            m_fill_tuner->setPeriod(period);
            m_fill_tuner->setEnabled(enable);

            m_split_tuner->setPeriod(period);
            m_split_tuner->setEnabled(enable);
            }
    protected:
        //! Find all possible particles suitable for swapping
        virtual void findSwapParticles();
        //! Swap momentum of particles
        virtual void swapPairMomentum();

    private:
        std::unique_ptr<Autotuner> m_swap_tuner; //!< Tuner for swaping velocities
        std::unique_ptr<Autotuner> m_mark_tuner; //!< Tuner for marking  particles in slabs
        std::unique_ptr<Autotuner> m_fill_tuner; //!< Tuner for filling the m_pairs GPUArray
        std::unique_ptr<Autotuner> m_split_tuner; //!< Tuner for finding the split in the array

        GPUArray<Scalar2> m_slab_pairs;   //!< Flags for device selection of particles in slabs
        GPUFlags<unsigned int> m_num_mark;//!< GPU flags for the number of marked particles
        GPUFlags<unsigned int> m_split;   //!< GPU flag for finding split between top and bottom in m_slab_pairs
        GPUFlags<int> m_type;             //!< GPU flag for marking wether first particle is either in the top or bottom slab
    };

namespace detail
{
//! Export ReversePerturbationFlowGPU to python
void export_ReversePerturbationFlowGPU(pybind11::module& m);
} // end namespace detail
} // end namespace azplugins
#endif // AZPLUGINS_REVERSE_PERTUBATION_FLOW_GPU_H_
