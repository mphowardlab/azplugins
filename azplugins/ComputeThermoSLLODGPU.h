// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "hoomd/Compute.h"
#include "ComputeThermoSLLOD.h"

/*! \file ComputeThermoGPU.h
    \brief Declares a class for computing thermodynamic quantities on the GPU
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __AZPLUGINS_COMPUTE_THERMO_SLLOD_GPU_H__
#define __AZPLUGINS_COMPUTE_THERMO_SLLOD_GPU_H__

namespace azplugins
{
//! Computes thermodynamic properties of a group of particles on the GPU
/*! ComputeThermoGPU is a GPU accelerated implementation of ComputeThermo
    \ingroup computes
*/
class PYBIND11_EXPORT ComputeThermoSLLODGPU : public ComputeThermoSLLOD
    {
    public:
        //! Constructs the compute
        ComputeThermoSLLODGPU(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<ParticleGroup> group,
                         Scalar shear_rate,
                         const std::string& suffix = std::string(""));
        virtual ~ComputeThermoSLLODGPU();

    protected:
        GlobalVector<Scalar4> m_scratch;  //!< Scratch space for partial sums
        GlobalVector<Scalar> m_scratch_pressure_tensor; //!< Scratch space for pressure tensor partial sums
        GlobalVector<Scalar> m_scratch_rot; //!< Scratch space for rotational kinetic energy partial sums
        unsigned int m_block_size;   //!< Block size executed
        cudaEvent_t m_event;         //!< CUDA event for synchronization

#ifdef ENABLE_MPI
        //! Reduce properties over MPI
        virtual void reduceProperties();

#endif
        virtual void addFlowField();
        virtual void removeFlowField();

        //! Does the actual computation
        virtual void computeProperties();
    };

namespace detail
{
//! Exports the ComputeThermoGPU class to python
void export_ComputeThermoSLLODGPU(pybind11::module& m);

} // end namespace detail
} // end namespace azplugins
#endif
