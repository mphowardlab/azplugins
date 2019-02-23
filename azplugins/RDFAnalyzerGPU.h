// Copyright (c) 2016-2018, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file RDFAnalyzerGPU.h
 * \brief Declaration of RDFAnalyzerGPU
 */

#ifndef AZPLUGINS_RDF_ANALYZER_GPU_H_
#define AZPLUGINS_RDF_ANALYZER_GPU_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "RDFAnalyzer.h"
#include "hoomd/Autotuner.h"
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

namespace azplugins
{

//! Radial distribution function analyzer for the GPU
class PYBIND11_EXPORT RDFAnalyzerGPU : public RDFAnalyzer
    {
    public:
        //! Constructor
        RDFAnalyzerGPU(std::shared_ptr<SystemDefinition> sysdef,
                       std::shared_ptr<ParticleGroup> group_1,
                       std::shared_ptr<ParticleGroup> group_2,
                       Scalar rcut,
                       Scalar bin_width);

        //! Destructor
        virtual ~RDFAnalyzerGPU() {};

        //! Set autotuner parameters
        /*!
         * \param enable Enable/disable autotuning
         * \param period period (approximate) in time steps when returning occurs
         */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            RDFAnalyzer::setAutotunerParams(enable, period);
            m_bin_tuner->setPeriod(period);
            m_bin_tuner->setEnabled(enable);
            }

    protected:
        //! Bin particle counts
        virtual void binParticles();

        //! Accumulate the counts into the RDF
        virtual void accumulate();

    private:
        std::unique_ptr<Autotuner> m_bin_tuner;     //!< Tuner for binning particles
        std::unique_ptr<Autotuner> m_accum_tuner;   //!< Tuner for accumulating rdf
    };

namespace detail
{
//! Exports the RDFAnalyzerGPU to python
void export_RDFAnalyzerGPU(pybind11::module& m);
} // end namespace detail
} // end namespace azplugins

#endif // AZPLUGINS_RDF_ANALYZER_GPU_H_
