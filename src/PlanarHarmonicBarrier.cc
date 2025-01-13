// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file PlanarHarmonicBarrier.cc
 * \brief Definition of PlanarHarmonicBarrier
 */

#include "PlanarHarmonicBarrier.h"

namespace hoomd
    {

namespace azplugins
    {
/*!
 * \param sysdef System definition
 * \param interf Position of the interface
 */
PlanarHarmonicBarrier::PlanarHarmonicBarrier(std::shared_ptr<SystemDefinition> sysdef,
                                             std::shared_ptr<Variant> interf)
    : HarmonicBarrier(sysdef, interf)
    {
    m_exec_conf->msg->notice(5) << "Constructing PlanarHarmonicBarrier" << std::endl;
    }

PlanarHarmonicBarrier::~PlanarHarmonicBarrier()
    {
    m_exec_conf->msg->notice(5) << "Destroying PlanarHarmonicBarrier" << std::endl;
    }

/*!
 * \param timestep Current timestep
 */
void PlanarHarmonicBarrier::computeForces(uint64_t timestep)
    {
    HarmonicBarrier::computeForces(timestep);

    const BoxDim& box = m_pdata->getGlobalBox();
    const Scalar interf_origin = m_interf->operator()(timestep);
    if (interf_origin > box.getHi().z || interf_origin < box.getLo().z)
        {
        m_exec_conf->msg->error()
            << "PlanarHarmonicBarrier interface must be inside the simulation box" << std::endl;
        throw std::runtime_error(
            "PlanarHarmonicBarrier interface must be inside the simulation box");
        }

    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);
    memset((void*)h_force.data, 0, sizeof(Scalar4) * m_force.getNumElements());
    memset((void*)h_virial.data, 0, sizeof(Scalar) * m_virial.getNumElements());

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar2> h_params(m_params, access_location::host, access_mode::read);
    for (unsigned int idx = 0; idx < m_pdata->getN(); ++idx)
        {
        const Scalar4 postype_i = h_pos.data[idx];
        const Scalar z_i = postype_i.z;
        const unsigned int type_i = __scalar_as_int(postype_i.w);

        const Scalar2 params = h_params.data[type_i];
        const Scalar k = params.x;
        const Scalar offset = params.y;

        const Scalar dz = z_i - (interf_origin + offset);
        if (dz < Scalar(0.0))
            continue;

        Scalar fz(0.0), e(0.0);
        // harmonic
        fz = -k * dz;
        e = Scalar(-0.5) * fz * dz; // (k/2) dz^2

        h_force.data[idx] = make_scalar4(0.0, 0.0, fz, e);
        }
    }

namespace detail
    {
/*!
 * \param m Python module to export to
 */
void export_PlanarHarmonicBarrier(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<PlanarHarmonicBarrier, std::shared_ptr<PlanarHarmonicBarrier>, HarmonicBarrier>(
        m,
        "PlanarHarmonicBarrier")
        .def(py::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<Variant>>());
    ;
    }
    } // end namespace detail

    } // end namespace azplugins

    } // end namespace hoomd
