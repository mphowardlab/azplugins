// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2022, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

/*!
 * \file ImplicitPlaneEvaporator.cc
 * \brief Definition of ImplicitPlaneEvaporator
 */

#include "ImplicitPlaneEvaporator.h"

namespace azplugins
{
/*!
 * \param sysdef System definition
 * \param interf Position of the interface
 */
ImplicitPlaneEvaporator::ImplicitPlaneEvaporator(std::shared_ptr<SystemDefinition> sysdef,
                                                 std::shared_ptr<Variant> interf)
        : ImplicitEvaporator(sysdef, interf)
    {
    m_exec_conf->msg->notice(5) << "Constructing ImplicitPlaneEvaporator" << std::endl;
    }

ImplicitPlaneEvaporator::~ImplicitPlaneEvaporator()
    {
    m_exec_conf->msg->notice(5) << "Destroying ImplicitPlaneEvaporator" << std::endl;
    }

/*!
 * \param timestep Current timestep
 */
void ImplicitPlaneEvaporator::computeForces(unsigned int timestep)
    {
    ImplicitEvaporator::computeForces(timestep);

    const BoxDim& box = m_pdata->getGlobalBox();
    const Scalar interf_origin = m_interf->getValue(timestep);
    if (interf_origin > box.getHi().z || interf_origin < box.getLo().z)
        {
        m_exec_conf->msg->error() << "ImplicitPlaneEvaporator interface must be inside the simulation box" << std::endl;
        throw std::runtime_error("ImplicitPlaneEvaporator interface must be inside the simulation box");
        }

    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);
    memset((void*)h_force.data, 0, sizeof(Scalar4) * m_force.getNumElements());
    memset((void*)h_virial.data, 0, sizeof(Scalar) * m_virial.getNumElements());

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_params(m_params, access_location::host, access_mode::read);
    for (unsigned int idx = 0; idx < m_pdata->getN(); ++idx)
        {
        const Scalar4 postype_i = h_pos.data[idx];
        const Scalar z_i = postype_i.z;
        const unsigned int type_i = __scalar_as_int(postype_i.w);

        const Scalar4 params = h_params.data[type_i];
        const Scalar k = params.x;
        const Scalar offset = params.y;
        const Scalar g = params.z;
        const Scalar cutoff = params.w;

        const Scalar dz = z_i - (interf_origin + offset);
        if (cutoff < Scalar(0.0) || dz < Scalar(0.0)) continue;

        Scalar fz(0.0), e(0.0);
        if (dz < cutoff) // harmonic
            {
            fz = -k * dz;
            e = Scalar(-0.5) * fz * dz; // (k/2) dz^2
            }
        else // linear
            {
            fz = -g;
            e = Scalar(0.5) * k * cutoff * cutoff + g * (dz - cutoff);
            }

        h_force.data[idx] = make_scalar4(0.0, 0.0, fz, e);
        }
    }

namespace detail
{
/*!
 * \param m Python module to export to
 */
void export_ImplicitPlaneEvaporator(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<ImplicitPlaneEvaporator,std::shared_ptr<ImplicitPlaneEvaporator>>(m, "ImplicitPlaneEvaporator", py::base<ImplicitEvaporator>())
        .def(py::init<std::shared_ptr<SystemDefinition>,std::shared_ptr<Variant>>());
    ;
    }
} // end namespace detail

} // end namespace azplugins
