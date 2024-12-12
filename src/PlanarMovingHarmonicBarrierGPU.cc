// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file PlanarMovingHarmonicBarrierGPU.cc
 * \brief Definition of PlanarMovingHarmonicBarrierGPU
 */

#include "PlanarMovingHarmonicBarrierGPU.h"
#include "PlanarMovingHarmonicBarrierGPU.cuh"

namespace hoomd
    {

namespace azplugins
    {

/*!
 * \param sysdef System definition
 * \param interf Position of the interface
 */
PlanarMovingHarmonicBarrierGPU::PlanarMovingHarmonicBarrierGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                               std::shared_ptr<Variant> interf)
    : MovingHarmonicPotentialGPU(sysdef, interf)
    {
    m_exec_conf->msg->notice(5) << "Constructing PlanarMovingHarmonicBarrierGPU" << std::endl;
    }

PlanarMovingHarmonicBarrierGPU::~PlanarMovingHarmonicBarrierGPU()
    {
    m_exec_conf->msg->notice(5) << "Destroying PlanarMovingHarmonicBarrierGPU" << std::endl;
    }

/*!
 * \param timestep Current timestep
 */
void PlanarMovingHarmonicBarrierGPU::computeForces(unsigned int timestep)
    {
    MovingHarmonicPotentialGPU::computeForces(timestep);

    const BoxDim& box = m_pdata->getGlobalBox();
    const Scalar interf_origin = m_interf->getValue(timestep);
    if (interf_origin > box.getHi().z || interf_origin < box.getLo().z)
        {
        m_exec_conf->msg->error()
            << "MovingHarmonicPotential interface must be inside the simulation box" << std::endl;
        throw std::runtime_error("MovingHarmonicPotential interface must be inside the simulation box");
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
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();
    }

namespace detail
    {
/*!
 * \param m Python module to export to
 */
void export_PlanarMovingHarmonicBarrierGPU(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<PlanarMovingHarmonicBarrierGPU, std::shared_ptr<PlanarMovingHarmonicBarrierGPU>>(
        m,
        "PlanarMovingHarmonicBarrierGPU",
        py::base<MovingHarmonicPotentialGPU>())
        .def(py::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<Variant>>());
    }
    } // end namespace detail

    } // end namespace azplugins

    } // end namespace hoomd
