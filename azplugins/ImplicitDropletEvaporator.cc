// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

/*!
 * \file ImplicitDropletEvaporator.cc
 * \brief Definition of ImplicitDropletEvaporator
 */

#include "ImplicitDropletEvaporator.h"

namespace azplugins
{
/*!
 * \param sysdef System definition
 * \param interf Position of the interface
 */
ImplicitDropletEvaporator::ImplicitDropletEvaporator(std::shared_ptr<SystemDefinition> sysdef,
                                                     std::shared_ptr<Variant> interf)
        : ImplicitEvaporator(sysdef, interf)
    {
    m_exec_conf->msg->notice(5) << "Constructing ImplicitDropletEvaporator" << std::endl;
    }

ImplicitDropletEvaporator::~ImplicitDropletEvaporator()
    {
    m_exec_conf->msg->notice(5) << "Destroying ImplicitDropletEvaporator" << std::endl;
    }

/*!
 * \param timestep Current timestep
 */
void ImplicitDropletEvaporator::computeForces(unsigned int timestep)
    {
    ImplicitEvaporator::computeForces(timestep);

    // check radius fits in box
    const Scalar interf_origin = m_interf->getValue(timestep);
        {
        const BoxDim& box = m_pdata->getGlobalBox();
        const Scalar3 hi = box.getHi();
        const Scalar3 lo = box.getLo();
        if (interf_origin > hi.x || interf_origin < lo.x ||
            interf_origin > hi.y || interf_origin < lo.y ||
            interf_origin > hi.z || interf_origin < lo.z)
            {
            m_exec_conf->msg->error() << "ImplicitDropletEvaporator interface must be inside the simulation box" << std::endl;
            throw std::runtime_error("ImplicitDropletEvaporator interface must be inside the simulation box");
            }
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
        const Scalar3 pos_i = make_scalar3(postype_i.x, postype_i.y, postype_i.z);
        const unsigned int type_i = __scalar_as_int(postype_i.w);

        const Scalar4 params = h_params.data[type_i];
        const Scalar k = params.x;
        const Scalar offset = params.y;
        const Scalar g = params.z;
        const Scalar cutoff = params.w;
        // continue if interaction is off
        if (cutoff < Scalar(0.0)) continue;

        // get distances and direction of force
        const Scalar r_i = fast::sqrt(dot(pos_i,pos_i));
        const Scalar dr = r_i - (interf_origin + offset);
        if (!(r_i > Scalar(0.0)) || dr < Scalar(0.0)) continue;
        const Scalar3 rhat = pos_i/r_i;

        Scalar3 f;
        Scalar e;
        if (dr < cutoff) // harmonic
            {
            f = -k * dr * rhat;
            e = Scalar(0.5) * k * (dr * dr); // (k/2) dr^2
            }
        else // linear
            {
            f = -g * rhat;
            e = Scalar(0.5) * k * cutoff * cutoff + g * (dr - cutoff);
            }

        h_force.data[idx] = make_scalar4(f.x, f.y, f.z, e);
        }
    }

namespace detail
{
/*!
 * \param m Python module to export to
 */
void export_ImplicitDropletEvaporator(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<ImplicitDropletEvaporator,std::shared_ptr<ImplicitDropletEvaporator>>(m, "ImplicitDropletEvaporator", py::base<ImplicitEvaporator>())
        .def(py::init<std::shared_ptr<SystemDefinition>,std::shared_ptr<Variant>>());
    ;
    }
} // end namespace detail

} // end namespace azplugins
