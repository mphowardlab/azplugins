// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file ImplicitDropletEvaporatorGPU.cc
 * \brief Definition of ImplicitDropletEvaporatorGPU
 */

#include "ImplicitDropletEvaporatorGPU.h"
#include "ImplicitDropletEvaporatorGPU.cuh"

namespace azplugins
    {

/*!
 * \param sysdef System definition
 * \param interf Position of the interface
 */
ImplicitDropletEvaporatorGPU::ImplicitDropletEvaporatorGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                           std::shared_ptr<Variant> interf)
    : ImplicitEvaporatorGPU(sysdef, interf)
    {
    m_exec_conf->msg->notice(5) << "Constructing ImplicitDropletEvaporatorGPU" << std::endl;
    }

ImplicitDropletEvaporatorGPU::~ImplicitDropletEvaporatorGPU()
    {
    m_exec_conf->msg->notice(5) << "Destroying ImplicitDropletEvaporatorGPU" << std::endl;
    }

/*!
 * \param timestep Current timestep
 */
void ImplicitDropletEvaporatorGPU::computeForces(unsigned int timestep)
    {
    ImplicitEvaporatorGPU::computeForces(timestep);

    // check radius fits in box
    const Scalar interf_origin = m_interf->getValue(timestep);
        {
        const BoxDim& box = m_pdata->getGlobalBox();
        const Scalar3 hi = box.getHi();
        const Scalar3 lo = box.getLo();
        if (interf_origin > hi.x || interf_origin < lo.x || interf_origin > hi.y
            || interf_origin < lo.y || interf_origin > hi.z || interf_origin < lo.z)
            {
            m_exec_conf->msg->error()
                << "ImplicitDropletEvaporator interface must be inside the simulation box"
                << std::endl;
            throw std::runtime_error(
                "ImplicitDropletEvaporator interface must be inside the simulation box");
            }
        }

    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_params(m_params, access_location::device, access_mode::read);

    m_tuner->begin();
    gpu::compute_implicit_evap_droplet_force(d_force.data,
                                             d_virial.data,
                                             d_pos.data,
                                             d_params.data,
                                             interf_origin,
                                             m_pdata->getN(),
                                             m_pdata->getNTypes(),
                                             m_tuner->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();
    }

namespace detail
    {
/*!
 * \param m Python module to export to
 */
void export_ImplicitDropletEvaporatorGPU(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<ImplicitDropletEvaporatorGPU, std::shared_ptr<ImplicitDropletEvaporatorGPU>>(
        m,
        "ImplicitDropletEvaporatorGPU",
        py::base<ImplicitEvaporatorGPU>())
        .def(py::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<Variant>>());
    }
    } // end namespace detail

    } // end namespace azplugins
