// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file HarmonicBarrierGPU.h
 * \brief Declaration of HarmonicBarrierGPU
 */

#ifndef AZPLUGINS_HARMONIC_BARRIER_GPU_H_
#define AZPLUGINS_HARMONIC_BARRIER_GPU_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "HarmonicBarrier.h"

#include "hoomd/Autotuner.h"

namespace hoomd
    {

namespace azplugins
    {

//! Moving Harmonic potential on the GPU
/*!
 * This class does not implement any force evaluation on its own, as the geometry should be
 * implemented by deriving classes. It exists as a thin layer between HarmonicBarrier
 * to remove some boilerplate of setting up the autotuners.
 */
class PYBIND11_EXPORT HarmonicBarrierGPU : public HarmonicBarrier
    {
    public:
    //! Constructor
    HarmonicBarrierGPU(std::shared_ptr<SystemDefinition> sysdef,
                       std::shared_ptr<Variant> interf);

    protected:
    std::shared_ptr<Autotuner<1>> m_tuner; //!< Autotuner for block size
    };

    } // end namespace azplugins

    } // end namespace hoomd

#endif // AZPLUGINS_HARMONIC_BARRIER_GPU_H_
