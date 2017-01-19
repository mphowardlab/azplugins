// Copyright (c) 2016, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file ImplicitEvaporatorGPU.cc
 * \brief Definition of ImplicitEvaporatorGPU
 */

#include "ImplicitEvaporatorGPU.h"
#include "ImplicitEvaporatorGPU.cuh"

namespace azplugins
{

/*!
 * \param sysdef System definition
 * \param interf Position of the interface
 */
ImplicitEvaporatorGPU::ImplicitEvaporatorGPU(std::shared_ptr<SystemDefinition> sysdef,
                                             std::shared_ptr<Variant> interf)
        : ImplicitEvaporator(sysdef, interf)
    {
    m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "implicit_evap", m_exec_conf));
    }

/*!
 * \param timestep Current timestep
 */
void ImplicitEvaporatorGPU::computeForces(unsigned int timestep)
    {
    PDataFlags flags = this->m_pdata->getFlags();
    if (!m_has_warned && (flags[pdata_flag::pressure_tensor] || flags[pdata_flag::isotropic_virial]))
        {
        m_exec_conf->msg->warning() << "ImplicitEvaporator does not compute its virial contribution, pressure may be inaccurate" << std::endl;
        m_has_warned = true;
        }

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
void export_ImplicitEvaporatorGPU(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_< ImplicitEvaporatorGPU, std::shared_ptr<ImplicitEvaporatorGPU> >(m, "ImplicitEvaporatorGPU", py::base<ImplicitEvaporator>())
        .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<Variant> >())
    ;
    }
} // end namespace detail

} // end namespace azplugins
