// Copyright (c) 2018-2019, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file RDFAnalyzerGPU.cc
 * \brief Definition of RDFAnalyzerGPU
 */

#include "RDFAnalyzerGPU.h"
#include "RDFAnalyzerGPU.cuh"

namespace azplugins
{

/*!
 * \param sysdef System definition
 * \param group_1 Particle group
 * \param group_2 Particle
 * \param rcut Cutoff radius for calculation
 * \param bin_width Width for binning particle distances
 */
RDFAnalyzerGPU::RDFAnalyzerGPU(std::shared_ptr<SystemDefinition> sysdef,
                               std::shared_ptr<ParticleGroup> group_1,
                               std::shared_ptr<ParticleGroup> group_2,
                               Scalar rcut,
                               Scalar bin_width)
    : RDFAnalyzer(sysdef, group_1, group_2, rcut, bin_width)
    {
    m_bin_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "rdf_bin", m_exec_conf));
    m_accum_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "rdf_accum", m_exec_conf));
    }

void RDFAnalyzerGPU::binParticles()
    {
    if (m_prof) m_prof->push(m_exec_conf, "bin");

    ArrayHandle<unsigned int> d_counts(m_counts, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_group_1(m_group_1->getIndexArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_group_2(m_group_2->getIndexArray(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);

    m_duplicates.resetFlags(0);

    m_bin_tuner->begin();
    gpu::analyze_rdf_bin(d_counts.data,
                         m_duplicates.getDeviceFlags(),
                         d_group_1.data,
                         d_group_2.data,
                         d_pos.data,
                         m_group_1->getNumMembers(),
                         m_group_2->getNumMembers(),
                         m_pdata->getBox(),
                         m_num_bins,
                         m_rcut * m_rcut,
                         m_bin_width,
                         m_bin_tuner->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
    m_bin_tuner->end();

    if (m_prof) m_prof->pop(m_exec_conf);
    }

void RDFAnalyzerGPU::accumulate()
    {
    if (m_prof) m_prof->push(m_exec_conf, "accumulate");

    ArrayHandle<unsigned int> d_counts(m_counts, access_location::device, access_mode::read);
    ArrayHandle<double> d_accum_rdf(m_accum_rdf, access_location::device, access_mode::readwrite);

    m_accum_tuner->begin();
    gpu::analyze_rdf_accumulate(d_accum_rdf.data,
                                m_num_samples,
                                d_counts.data,
                                m_group_1->getNumMembersGlobal(),
                                m_group_2->getNumMembersGlobal(),
                                m_pdata->getGlobalBox().getVolume(),
                                m_duplicates.readFlags(),
                                m_num_bins,
                                m_rcut,
                                m_bin_width,
                                m_accum_tuner->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
    m_accum_tuner->end();

    if (m_prof) m_prof->pop(m_exec_conf);
    }

namespace detail
{
/*!
 * \param m Python module for export
 */
void export_RDFAnalyzerGPU(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_< RDFAnalyzerGPU, std::shared_ptr<RDFAnalyzerGPU> >(m, "RDFAnalyzerGPU", py::base<RDFAnalyzer>())
        .def(py::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, std::shared_ptr<ParticleGroup>, Scalar, Scalar>());
    }
} // end namespace detail
} // end namespace azplugins
