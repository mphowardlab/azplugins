// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2022, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

/*!
 * \file OrientationRestraintComputeGPU.h
 * \brief Declares a class for computing orientation restraining forces on the GPU
 */

#ifndef AZPLUGINS_ORIENTATION_RESTRAINT_COMPUTE_GPU_H_
#define AZPLUGINS_ORIENTATION_RESTRAINT_COMPUTE_GPU_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "OrientationRestraintCompute.h"
#include "hoomd/Autotuner.h"

namespace azplugins
{
//! Adds a restraining force to groups of particles on the GPU
class PYBIND11_EXPORT OrientationRestraintComputeGPU : public OrientationRestraintCompute
    {
    public:
        //! Constructs the compute
        OrientationRestraintComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                                       std::shared_ptr<ParticleGroup> group);

        //! Destructor
        ~OrientationRestraintComputeGPU() {};

        //! Set autotuner parameters
        /*!
         * \param enable Enable/disable autotuning
         * \param period period (approximate) in time steps when returning occurs
         */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            OrientationRestraintCompute::setAutotunerParams(enable, period);
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
//! Exports the OrientationRestraintComputeGPU to python
void export_OrientationRestraintComputeGPU(pybind11::module& m);
} // end namespace detail
} // end namespace azplugins

#endif // AZPLUGINS_ORIENTATION_RESTRAINT_COMPUTE_GPU_H_
