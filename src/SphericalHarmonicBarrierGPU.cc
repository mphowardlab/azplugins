// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file SphericalHarmonicBarrierGPU.cc
 * \brief Definition of SphericalHarmonicBarrierGPU
 */

#include "SphericalHarmonicBarrierGPU.h"
#include "SphericalHarmonicBarrierGPU.cuh"

namespace hoomd
    {

namespace azplugins
    {

/*!
 * \param sysdef System definition
 * \param interf Position of the interface
 */
SphericalHarmonicBarrierGPU::SphericalHarmonicBarrierGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                         std::shared_ptr<Variant> interf)
    : HarmonicBarrierGPU(sysdef, interf)
    {
    m_exec_conf->msg->notice(5) << "Constructing SphericalHarmonicBarrierGPU" << std::endl;
    }

SphericalHarmonicBarrierGPU::~SphericalHarmonicBarrierGPU()
    {
    m_exec_conf->msg->notice(5) << "Destroying SphericalHarmonicBarrierGPU" << std::endl;
    }

/*!
 * \param timestep Current timestep
 */
void SphericalHarmonicBarrierGPU::computeForces(uint64_t timestep)
    {
    HarmonicBarrierGPU::computeForces(timestep);

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
                << "SphericalHarmonicBarrier interface must be inside the simulation box"
                << std::endl;
            throw std::runtime_error(
                "SphericalHarmonicBarrier interface must be inside the simulation box");
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
                                             m_tuner->getParam()[0]);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();
    }

namespace detail
    {
/*!
 * \param m Python module to export to
 */
void export_SphericalHarmonicBarrierGPU(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<SphericalHarmonicBarrierGPU, std::shared_ptr<SphericalHarmonicBarrierGPU>, HarmonicBarrierGPU>(
        m,
        "SphericalHarmonicBarrierGPU")
        .def(py::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<Variant>>());
    }
    } // end namespace detail

    } // end namespace azplugins

    } // end namespace hoomd
