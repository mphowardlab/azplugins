// Copyright (c) 2018-2019, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file ImplicitPlaneEvaporatorGPU.cc
 * \brief Definition of ImplicitPlaneEvaporatorGPU
 */

#include "ImplicitPlaneEvaporatorGPU.h"
#include "ImplicitPlaneEvaporatorGPU.cuh"

namespace azplugins
{

/*!
 * \param sysdef System definition
 * \param interf Position of the interface
 */
ImplicitPlaneEvaporatorGPU::ImplicitPlaneEvaporatorGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                       std::shared_ptr<Variant> interf)
        : ImplicitEvaporatorGPU(sysdef, interf)
    {
    m_exec_conf->msg->notice(5) << "Constructing ImplicitPlaneEvaporatorGPU" << std::endl;
    }

ImplicitPlaneEvaporatorGPU::~ImplicitPlaneEvaporatorGPU()
    {
    m_exec_conf->msg->notice(5) << "Destroying ImplicitPlaneEvaporatorGPU" << std::endl;
    }

/*!
 * \param timestep Current timestep
 */
void ImplicitPlaneEvaporatorGPU::computeForces(unsigned int timestep)
    {
    ImplicitEvaporatorGPU::computeForces(timestep);

    const BoxDim& box = m_pdata->getGlobalBox();
    const Scalar interf_origin = m_interf->getValue(timestep);
    if (interf_origin > box.getHi().z || interf_origin < box.getLo().z)
        {
        m_exec_conf->msg->error() << "ImplicitEvaporator interface must be inside the simulation box" << std::endl;
        throw std::runtime_error("ImplicitEvaporator interface must be inside the simulation box");
        }

    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_params(m_params, access_location::device, access_mode::read);

    m_tuner->begin();
    gpu::compute_implicit_evap_force(d_force.data,
                                       d_virial.data,
                                       d_pos.data,
                                       d_params.data,
                                       interf_origin,
                                       m_pdata->getN(),
                                       m_pdata->getNTypes(),
                                       m_tuner->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
    m_tuner->end();
    }

namespace detail
{
/*!
 * \param m Python module to export to
 */
void export_ImplicitPlaneEvaporatorGPU(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<ImplicitPlaneEvaporatorGPU,std::shared_ptr<ImplicitPlaneEvaporatorGPU> >(m, "ImplicitPlaneEvaporatorGPU", py::base<ImplicitEvaporatorGPU>())
        .def(py::init<std::shared_ptr<SystemDefinition>,std::shared_ptr<Variant>>())
    ;
    }
} // end namespace detail

} // end namespace azplugins
