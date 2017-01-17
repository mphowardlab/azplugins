// Copyright (c) 2016-2017, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file ImplicitEvaporator.cc
 * \brief Definition of ImplicitEvaporator
 */

#include "ImplicitEvaporator.h"

namespace azplugins
{
/*!
 * \param sysdef System definition
 * \param interf Position of the interface
 */
ImplicitEvaporator::ImplicitEvaporator(std::shared_ptr<SystemDefinition> sysdef,
                                       std::shared_ptr<Variant> interf)
        : ForceCompute(sysdef), m_interf(interf), m_has_warned(false)
    {
    m_exec_conf->msg->notice(5) << "Constructing ImplicitEvaporator" << std::endl;

    // allocate memory per type for parameters
    GPUArray<Scalar4> params(m_pdata->getNTypes(), m_exec_conf);
    m_params.swap(params);

    // connect to type change to resize type data arrays
    m_pdata->getNumTypesChangeSignal().connect<ImplicitEvaporator, &ImplicitEvaporator::reallocateParams>(this);
    }

ImplicitEvaporator::~ImplicitEvaporator()
    {
    m_exec_conf->msg->notice(5) << "Destroying ImplicitEvaporator" << std::endl;

    m_pdata->getNumTypesChangeSignal().disconnect<ImplicitEvaporator, &ImplicitEvaporator::reallocateParams>(this);
    }

/*!
 * \param timestep Current timestep
 */
void ImplicitEvaporator::computeForces(unsigned int timestep)
    {
    PDataFlags flags = this->m_pdata->getFlags();
    if (!m_has_warned && (flags[pdata_flag::pressure_tensor] || flags[pdata_flag::isotropic_virial]))
        {
        m_exec_conf->msg->warning() << "ImplicitEvaporator does not compute its virial contribution, pressure may be inaccurate" << std::endl;
        m_has_warned = true;
        }

    const BoxDim& box = m_pdata->getGlobalBox();
    const Scalar interf_origin = m_interf->getValue(timestep);
    if (interf_origin > box.getHi().z || interf_origin < box.getLo().z)
        {
        m_exec_conf->msg->error() << "ImplicitEvaporator interface must be inside the simulation box" << std::endl;
        throw std::runtime_error("ImplicitEvaporator interface must be inside the simulation box");
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
void export_ImplicitEvaporator(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_< ImplicitEvaporator, std::shared_ptr<ImplicitEvaporator> >(m, "ImplicitEvaporator", py::base<ForceCompute>())
        .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<Variant> >())
        .def("setParams", &ImplicitEvaporator::setParams);
    ;
    }
} // end namespace detail

} // end namespace azplugins
