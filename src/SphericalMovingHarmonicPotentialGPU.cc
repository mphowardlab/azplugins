// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file SphericalMovingHarmonicPotentialGPU.cc
 * \brief Definition of SphericalMovingHarmonicPotentialGPU
 */

#include "SphericalMovingHarmonicPotentialGPU.h"
#include "SphericalMovingHarmonicPotentialGPU.cuh"

namespace hoomd
    {

namespace azplugins
    {

/*!
 * \param sysdef System definition
 * \param interf Position of the interface
 */
SphericalMovingHarmonicPotentialGPU::SphericalMovingHarmonicPotentialGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                                         std::shared_ptr<Variant> interf)
    : MovingHarmonicPotentialGPU(sysdef, interf)
    {
    m_exec_conf->msg->notice(5) << "Constructing SphericalMovingHarmonicPotentialGPU" << std::endl;
    }

SphericalMovingHarmonicPotentialGPU::~SphericalMovingHarmonicPotentialGPU()
    {
    m_exec_conf->msg->notice(5) << "Destroying SphericalMovingHarmonicPotentialGPU" << std::endl;
    }

/*!
 * \param timestep Current timestep
 */
void SphericalMovingHarmonicPotentialGPU::computeForces(unsigned int timestep)
    {
    MovingHarmonicPotentialGPU::computeForces(timestep);

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
                << "SphericalMovingHarmonicPotential interface must be inside the simulation box"
                << std::endl;
            throw std::runtime_error(
                "SphericalMovingHarmonicPotential interface must be inside the simulation box");
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
void export_SphericalMovingHarmonicPotentialGPU(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<SphericalMovingHarmonicPotentialGPU, std::shared_ptr<SphericalMovingHarmonicPotentialGPU>>(
        m,
        "SphericalMovingHarmonicPotentialGPU",
        py::base<MovingHarmonicPotentialGPU>())
        .def(py::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<Variant>>());
    }
    } // end namespace detail

    } // end namespace azplugins

    } // end namespace hoomd
