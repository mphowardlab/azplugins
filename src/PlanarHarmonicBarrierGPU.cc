// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file PlanarHarmonicBarrierGPU.cc
 * \brief Definition of PlanarHarmonicBarrierGPU
 */

#include "PlanarHarmonicBarrierGPU.h"
#include "PlanarHarmonicBarrierGPU.cuh"

namespace hoomd
    {

namespace azplugins
    {

/*!
 * \param sysdef System definition
 * \param interf Position of the interface
 */
PlanarHarmonicBarrierGPU::PlanarHarmonicBarrierGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                   std::shared_ptr<Variant> interf)
    : HarmonicBarrierGPU(sysdef, interf)
    {
    m_exec_conf->msg->notice(5) << "Constructing PlanarHarmonicBarrierGPU" << std::endl;
    }

PlanarHarmonicBarrierGPU::~PlanarHarmonicBarrierGPU()
    {
    m_exec_conf->msg->notice(5) << "Destroying PlanarHarmonicBarrierGPU" << std::endl;
    }

/*!
 * \param timestep Current timestep
 */
void PlanarHarmonicBarrierGPU::computeForces(uint64_t timestep)
    {
    HarmonicBarrierGPU::computeForces(timestep);

    const BoxDim& box = m_pdata->getGlobalBox();
    const Scalar interf_origin = m_interf->operator()(timestep);
    if (interf_origin > box.getHi().z || interf_origin < box.getLo().z)
        {
        m_exec_conf->msg->error()
            << "HarmonicBarrier interface must be inside the simulation box" << std::endl;
        throw std::runtime_error("HarmonicBarrier interface must be inside the simulation box");
        }

    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar2> d_params(m_params, access_location::device, access_mode::read);

    m_tuner->begin();
    gpu::compute_force_planar_harmonic_barrier(d_force.data,
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
void export_PlanarHarmonicBarrierGPU(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<PlanarHarmonicBarrierGPU, std::shared_ptr<PlanarHarmonicBarrierGPU>, HarmonicBarrierGPU>(
        m,
        "PlanarHarmonicBarrierGPU")
        .def(py::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<Variant>>());
    }
    } // end namespace detail

    } // end namespace azplugins

    } // end namespace hoomd
