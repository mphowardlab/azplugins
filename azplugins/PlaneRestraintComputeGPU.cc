// Copyright (c) 2018-2019, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

#include "PlaneRestraintComputeGPU.h"
#include "PlaneRestraintComputeGPU.cuh"

namespace azplugins
{
/*!
 * \param sysdef HOOMD system definition.
 * \param group Particle group to compute on.
 * \param point Point in the plane.
 * \param normal Normal to the plane.
 * \param k Spring constant.
 */
PlaneRestraintComputeGPU::PlaneRestraintComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                           std::shared_ptr<ParticleGroup> group,
                                                           Scalar3 point,
                                                           Scalar3 normal,
                                                           Scalar k)
    : PlaneRestraintCompute(sysdef, group, point, normal, k)
    {
    m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "harmonic_plane", m_exec_conf));
    }

/*!
 * \param timestep Current timestep
 *
 * Harmonic forces are computed on all particles in group based on their distance from the plane on the GPU.
 */
void PlaneRestraintComputeGPU::computeForces(unsigned int timestep)
    {
    ArrayHandle<unsigned int> d_group(m_group->getIndexArray(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial, access_location::device, access_mode::overwrite);

    // zero the forces and virial before calling
    cudaMemset((void*)d_force.data, 0, sizeof(Scalar4)*m_force.getNumElements());
    cudaMemset((void*)d_virial.data, 0, sizeof(Scalar)*m_virial.getNumElements());

    m_tuner->begin();
    gpu::compute_plane_restraint(d_force.data,
                                 d_virial.data,
                                 d_group.data,
                                 d_pos.data,
                                 d_image.data,
                                 m_pdata->getGlobalBox(),
                                 m_o,
                                 m_n,
                                 m_k,
                                 m_group->getNumMembers(),
                                 m_virial_pitch,
                                 m_tuner->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
    m_tuner->end();
    }

namespace detail
{
void export_PlaneRestraintComputeGPU(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<PlaneRestraintComputeGPU,std::shared_ptr<PlaneRestraintComputeGPU>>(m,"PlaneRestraintComputeGPU",py::base<PlaneRestraintCompute>())
    .def(py::init<std::shared_ptr<SystemDefinition>,std::shared_ptr<ParticleGroup>,Scalar3,Scalar3,Scalar>())
    ;
    }
} // end namespace detail
} // end namespace azplugins
