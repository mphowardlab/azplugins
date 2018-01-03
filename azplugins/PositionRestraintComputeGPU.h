// Copyright (c) 2016-2018, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: wes_reinhart

/*!
 * \file PositionRestraintComputeGPU.h
 * \brief Declares a class for computing position restraining forces on the GPU
 */

#ifndef AZPLUGINS_POSITION_RESTRAINT_COMPUTE_GPU_H_
#define AZPLUGINS_POSITION_RESTRAINT_COMPUTE_GPU_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "PositionRestraintCompute.h"
#include "hoomd/Autotuner.h"

namespace azplugins
{
//! Adds a restraining force to groups of particles on the GPU
class PositionRestraintComputeGPU : public PositionRestraintCompute
    {
    public:
        //! Constructs the compute
        PositionRestraintComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                                    std::shared_ptr<ParticleGroup> group);

        //! Destructor
        ~PositionRestraintComputeGPU() {};

        //! Set autotuner parameters
        /*!
         * \param enable Enable/disable autotuning
         * \param period period (approximate) in time steps when returning occurs
         */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            PositionRestraintCompute::setAutotunerParams(enable, period);
            m_tuner->setPeriod(period);
            m_tuner->setEnabled(enable);
            }

    protected:
        //! Actually compute the forces on the GPU
        virtual void computeForces(unsigned int timestep);

    private:
        std::unique_ptr<Autotuner> m_tuner;   //!< Autotuner for block size
    };

namespace detail
{
//! Exports the PositionRestraintComputeGPU to python
void export_PositionRestraintComputeGPU(pybind11::module& m);
} // end namespace detail
} // end namespace azplugins

#endif // AZPLUGINS_POSITION_RESTRAINT_COMPUTE_GPU_H_
