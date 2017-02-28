// Copyright (c) 2016-2017, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file RDFAnalyzer.h
 * \brief Declaration of RDFAnalyzer
 */

#ifndef AZPLUGINS_RDF_ANALYZER_H_
#define AZPLUGINS_RDF_ANALYZER_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "hoomd/Analyzer.h"
#include "hoomd/GPUFlags.h"
#include "hoomd/ParticleGroup.h"

#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

namespace azplugins
{
//! Radial distribution function analyzer
/*!
 * The RDFAnalyzer computes the radial distribution of two groups during the simulation.
 * The total RDF is accumulated internally over multiple frames so that it can be averaged.
 * No assumptions are made about the box size or number of particles, since it is the RDF
 * from each frame that is accumulated rather than the counts. It is up to the user to
 * determine if the accumulation makes physical sense.
 *
 * Currently, the analysis is implemented by an all-pairs search. This will fail
 * in systems with large numbers of particles when the group sizes are large.
 *
 * \todo Use a CellList to accelerate the calculation.
 */
class RDFAnalyzer : public Analyzer
    {
    public:
        //! Constructor
        RDFAnalyzer(std::shared_ptr<SystemDefinition> sysdef,
                    std::shared_ptr<ParticleGroup> group_1,
                    std::shared_ptr<ParticleGroup> group_2,
                    Scalar rcut,
                    Scalar bin_width);

        //! Destructor
        virtual ~RDFAnalyzer();

        //! Perform radial distribution function analysis
        virtual void analyze(unsigned int timestep);

        //! Get a copy of the bins
        std::vector<Scalar> getBins() const;

        //! Get a copy of the RDF data
        std::vector<Scalar> get() const;

        //! Reset the accumulated values
        void reset();

    protected:
        std::shared_ptr<ParticleGroup> m_group_1;   //!< First particle group
        std::shared_ptr<ParticleGroup> m_group_2;   //!< Second particle group

        GPUArray<unsigned int> m_counts;    //!< Array of bin counts from a single frame
        Scalar m_rcut;                      //!< Cutoff radius for calculation
        Scalar m_bin_width;                 //!< Bin width
        unsigned int m_num_bins;            //!< Number of bins
        GPUFlags<unsigned int> m_duplicates;    //!< Number of duplicated counts at current evaluation

        GPUArray<double> m_accum_rdf;   //!< Accumulated rdf
        unsigned int m_num_samples;     //!< Number of samples accumulated into the gr

        //! Bin particle counts
        virtual void binParticles();

        //! Accumulate the counts into the RDF
        virtual void accumulate();
    };

namespace detail
{
//! Exports the RDFAnalyzer to python
void export_RDFAnalyzer(pybind11::module& m);
} // end namespace detail
} // end namespace azplugins

#endif // AZPLUGINS_RDF_ANALYZER_H_
