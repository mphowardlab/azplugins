// Copyright (c) 2016-2017, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: wes_reinhart

/*!
 * \file OrientationRestraintCompute.cc
 * \brief Definition of OrientationRestraintCompute
 */

#include "OrientationRestraintComputeGPU.h"
#include "OrientationRestraintComputeGPU.cuh"

namespace azplugins
{
/*!
 * \param sysdef SystemDefinition containing the ParticleData to compute forces on
 * \param group A group of particles
 */
OrientationRestraintComputeGPU::OrientationRestraintComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                               std::shared_ptr<ParticleGroup> group)
        : OrientationRestraintCompute(sysdef, group)
    {
    m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "orientation_restraint", m_exec_conf));
    }

/*!
 * \param timestep Current timestep
 */
void OrientationRestraintComputeGPU::computeForces(unsigned int timestep)
    {
    ArrayHandle<Scalar4> d_force(m_force,access_location::device,access_mode::overwrite);
    ArrayHandle<Scalar4> d_torque(m_torque,access_location::device,access_mode::overwrite);

    ArrayHandle<unsigned int> d_member_idx(m_group->getIndexArray(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_ref_orient(m_ref_orient, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_orient(m_pdata->getOrientationArray(), access_location::device, access_mode::read);

    m_tuner->begin();
    gpu::compute_orientation_restraint(d_force.data,
                                       d_torque.data,
                                       d_member_idx.data,
                                       d_orient.data,
                                       d_ref_orient.data,
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
void export_OrientationRestraintComputeGPU(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_< OrientationRestraintComputeGPU, std::shared_ptr<OrientationRestraintComputeGPU> >
        (m, "OrientationRestraintComputeGPU", py::base<OrientationRestraintCompute>() )
        .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup> >());
    }
} // end namespace detail
} // end namespace azplugins
