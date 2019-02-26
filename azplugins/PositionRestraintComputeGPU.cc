// Copyright (c) 2018-2019, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: wes_reinhart

/*!
 * \file PositionRestraintCompute.cc
 * \brief Definition of PositionRestraintCompute
 */

#include "PositionRestraintComputeGPU.h"
#include "PositionRestraintComputeGPU.cuh"

namespace azplugins
{
/*!
 * \param sysdef SystemDefinition containing the ParticleData to compute forces on
 * \param group A group of particles
 */
PositionRestraintComputeGPU::PositionRestraintComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                         std::shared_ptr<ParticleGroup> group)
        : PositionRestraintCompute(sysdef, group)
    {
    m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "position_restraint", m_exec_conf));
    }

/*!
 * \param timestep Current timestep
 */
void PositionRestraintComputeGPU::computeForces(unsigned int timestep)
    {
    ArrayHandle<Scalar4> d_force(m_force,access_location::device,access_mode::overwrite);
    ArrayHandle<unsigned int> d_member_idx(m_group->getIndexArray(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_ref_pos(m_ref_pos, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);

    m_tuner->begin();
    gpu::compute_position_restraint(d_force.data,
                                    d_member_idx.data,
                                    d_pos.data,
                                    d_ref_pos.data,
                                    d_tag.data,
                                    m_k,
                                    m_pdata->getBox(),
                                    m_pdata->getN(),
                                    m_group->getNumMembers(),
                                    m_tuner->getParam(),
                                    m_exec_conf->getComputeCapability()/10);
    if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
    m_tuner->end();
    }

namespace detail
{
/*!
 * \param m Python module to export to
 */
void export_PositionRestraintComputeGPU(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_< PositionRestraintComputeGPU, std::shared_ptr<PositionRestraintComputeGPU> >
        (m, "PositionRestraintComputeGPU", py::base<PositionRestraintCompute>() )
        .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup> >());
    }
} // end namespace detail
} // end namespace azplugins
