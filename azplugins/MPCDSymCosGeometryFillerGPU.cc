// Copyright (c) 2018-2019, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt


/*!
 * \file SymCosGeometryFillerGPU.cc
 * \brief Definition of SymCosGeometryFillerGPU
 */

#include "MPCDSymCosGeometryFillerGPU.h"
#include "MPCDSymCosGeometryFillerGPU.cuh"

namespace azplugins
{

SymCosGeometryFillerGPU::SymCosGeometryFillerGPU(std::shared_ptr<mpcd::SystemData> sysdata,
                                                   Scalar density,
                                                   unsigned int type,
                                                   std::shared_ptr<::Variant> T,
                                                   unsigned int seed,
                                                   std::shared_ptr<const detail::SymCosGeometry> geom)
    : SymCosGeometryFiller(sysdata, density, type, T, seed, geom)
    {
    m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "mpcd_sine_filler", m_exec_conf));
    }

/*!
 * \param timestep Current timestep
 */
void SymCosGeometryFillerGPU::drawParticles(unsigned int timestep)
    {
    ArrayHandle<Scalar4> d_pos(m_mpcd_pdata->getPositions(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_vel(m_mpcd_pdata->getVelocities(), access_location::device, access_mode::readwrite);
    ArrayHandle<unsigned int> d_tag(m_mpcd_pdata->getTags(), access_location::device, access_mode::readwrite);

    const unsigned int first_idx = m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual() - m_N_fill;

    m_tuner->begin();
    gpu::sym_cos_draw_particles(d_pos.data,
                                   d_vel.data,
                                   d_tag.data,
                                   *m_geom,
                                   m_pi_period_div_L,
                                   m_amplitude,
                                   m_H_narrow,
                                   m_thickness,
                                   m_pdata->getBox(),
                                   m_mpcd_pdata->getMass(),
                                   m_type,
                                   m_N_fill,
                                   m_first_tag,
                                   first_idx,
                                   m_T->getValue(timestep),
                                   timestep,
                                   m_seed,
                                   m_tuner->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
    m_tuner->end();
    }

namespace detail
{
/*!
 * \param m Python module to export to
 */
void export_SymCosGeometryFillerGPU(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<SymCosGeometryFillerGPU, std::shared_ptr<SymCosGeometryFillerGPU>>
        (m, "SymCosGeometryFillerGPU", py::base<SymCosGeometryFiller>())
        .def(py::init<std::shared_ptr<mpcd::SystemData>,
                      Scalar,
                      unsigned int,
                      std::shared_ptr<::Variant>,
                      unsigned int,
                      std::shared_ptr<const SymCosGeometry>>())
        ;
    }
} // end namespace detail

} // end namespace azplugins
