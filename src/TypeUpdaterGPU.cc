// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2022, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

/*!
 * \file TypeUpdaterGPU.cc
 * \brief Definition of TypeUpdaterGPU
 */

#include "TypeUpdaterGPU.h"
#include "TypeUpdaterGPU.cuh"

namespace azplugins
{

/*!
 * \param sysdef System definition
 *
 * The system is initialized in a configuration that will be invalid on the
 * first check of the types and region. This constructor requires that the user
 * properly initialize the system via setters.
 */
TypeUpdaterGPU::TypeUpdaterGPU(std::shared_ptr<SystemDefinition> sysdef)
        : TypeUpdater(sysdef)
    {
    m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "type_updater", m_exec_conf));
    }

/*!
 * \param sysdef System definition
 * \param inside_type Type id of particles inside region
 * \param outside_type Type id of particles outside region
 * \param z_lo Lower bound of region in z
 * \param z_hi Upper bound of region in z
 */
TypeUpdaterGPU::TypeUpdaterGPU(std::shared_ptr<SystemDefinition> sysdef,
                               unsigned int inside_type,
                               unsigned int outside_type,
                               Scalar z_lo,
                               Scalar z_hi)
        : TypeUpdater(sysdef, inside_type, outside_type, z_lo, z_hi)
    {
    m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "type_updater", m_exec_conf));
    }

/*!
 * \param timestep Timestep update is called
 */
void TypeUpdaterGPU::changeTypes(unsigned int timestep)
    {
    if (m_prof) m_prof->push(m_exec_conf, "type update");

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);

    m_tuner->begin();
    gpu::change_types_region(d_pos.data, m_inside_type, m_outside_type, m_z_lo, m_z_hi, m_pdata->getN(), m_tuner->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();

    if (m_prof) m_prof->pop();
    }

namespace detail
{
/*!
 * \param m Python module to export to
 */
void export_TypeUpdaterGPU(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_< TypeUpdaterGPU, std::shared_ptr<TypeUpdaterGPU> >(m, "TypeUpdaterGPU", py::base<TypeUpdater>())
        .def(py::init< std::shared_ptr<SystemDefinition> >())
        .def(py::init<std::shared_ptr<SystemDefinition>, unsigned int, unsigned int, Scalar, Scalar>());
    }
} // end namespace detail

} // end namespace azplugins
