// Copyright (c) 2016, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file ParticleEvaporatorGPU.h
 * \brief Declaration of ParticleEvaporatorGPU
 */

#ifndef AZPLUGINS_PARTICLE_EVAPORATOR_GPU_H_
#define AZPLUGINS_PARTICLE_EVAPORATOR_GPU_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "ParticleEvaporator.h"
#include "hoomd/Autotuner.h"
#include "hoomd/GPUFlags.h"

namespace azplugins
{

//! Solvent-particle evaporator on the GPU
/*!
 * See ParticleEvaporator for details. This class inherits and minimally implements
 * the CPU methods from ParticleEvaporator on the GPU.
 */
class ParticleEvaporatorGPU : public ParticleEvaporator
    {
    public:
        //! Simple constructor
        ParticleEvaporatorGPU(std::shared_ptr<SystemDefinition> sysdef, unsigned int seed);

        //! Constructor with parameters
        ParticleEvaporatorGPU(std::shared_ptr<SystemDefinition> sysdef,
                       unsigned int inside_type,
                       unsigned int outside_type,
                       Scalar z_lo,
                       Scalar z_hi,
                       unsigned int seed);

        //! Destructor
        virtual ~ParticleEvaporatorGPU() {};

        //! Set autotuner parameters
        /*!
         * \param enable Enable / disable autotuning
         * \param period period (approximate) in time steps when retuning occurs
         */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            ParticleEvaporator::setAutotunerParams(enable, period);

            m_mark_tuner->setPeriod(period);
            m_mark_tuner->setEnabled(enable);

            m_pick_tuner->setPeriod(period);
            m_pick_tuner->setEnabled(enable);
            }

    protected:
        //! Mark particles as candidates for evaporation on the GPU
        virtual unsigned int markParticles();

        //! Apply evaporation to picks on the GPU
        virtual void applyPicks();

    private:
        std::unique_ptr<Autotuner> m_mark_tuner;    //!< Tuner for marking particles
        std::unique_ptr<Autotuner> m_pick_tuner;    //!< Tuner for applying picks
        GPUVector<unsigned char> m_select_flags;    //!< Flags (1/0) for device selection of marked particles
        GPUFlags<unsigned int> m_num_mark;          //!< GPU flags for the number of marked particles
    };

namespace detail
{
//! Export ParticleEvaporatorGPU to python
void export_ParticleEvaporatorGPU(pybind11::module& m);
} // end namespace detail

} // end namespace azplugins

#endif // AZPLUGINS_PARTICLE_EVAPORATOR_GPU_H_
