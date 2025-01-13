// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file SphericalHarmonicBarrier.cc
 * \brief Definition of SphericalHarmonicBarrier
 */

#include "SphericalHarmonicBarrier.h"

namespace hoomd
    {

namespace azplugins
    {
/*!
 * \param sysdef System definition
 * \param interf Position of the interface
 */
SphericalHarmonicBarrier::SphericalHarmonicBarrier(std::shared_ptr<SystemDefinition> sysdef,
                                                   std::shared_ptr<Variant> interf)
    : HarmonicBarrier(sysdef, interf)
    {
    m_exec_conf->msg->notice(5) << "Constructing SphericalHarmonicBarrier" << std::endl;
    }

SphericalHarmonicBarrier::~SphericalHarmonicBarrier()
    {
    m_exec_conf->msg->notice(5) << "Destroying SphericalHarmonicBarrier" << std::endl;
    }

/*!
 * \param timestep Current timestep
 */
void SphericalHarmonicBarrier::computeForces(uint64_t timestep)
    {
    HarmonicBarrier::computeForces(timestep);

    // check radius fits in box
    const Scalar interf_origin = m_interf->operator()(timestep);
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

    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);
    memset((void*)h_force.data, 0, sizeof(Scalar4) * m_force.getNumElements());
    memset((void*)h_virial.data, 0, sizeof(Scalar) * m_virial.getNumElements());

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar2> h_params(m_params, access_location::host, access_mode::read);
    for (unsigned int idx = 0; idx < m_pdata->getN(); ++idx)
        {
        const Scalar4 postype_i = h_pos.data[idx];
        const Scalar3 pos_i = make_scalar3(postype_i.x, postype_i.y, postype_i.z);
        const unsigned int type_i = __scalar_as_int(postype_i.w);

        const Scalar2 params = h_params.data[type_i];
        const Scalar k = params.x;
        const Scalar offset = params.y;

        // get distances and direction of force
        const Scalar r_i = fast::sqrt(dot(pos_i, pos_i));
        const Scalar dr = r_i - (interf_origin + offset);
        if (!(r_i > Scalar(0.0)) || dr < Scalar(0.0))
            continue;
        const Scalar3 rhat = pos_i / r_i;

        Scalar3 f;
        Scalar e;
        // harmonic
        f = -k * dr * rhat;
        e = Scalar(0.5) * k * (dr * dr); // (k/2) dr^2

        h_force.data[idx] = make_scalar4(f.x, f.y, f.z, e);
        }
    }

namespace detail
    {
/*!
 * \param m Python module to export to
 */
void export_SphericalHarmonicBarrier(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<SphericalHarmonicBarrier, std::shared_ptr<SphericalHarmonicBarrier>, HarmonicBarrier>(
        m,
        "SphericalHarmonicBarrier")
        .def(py::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<Variant>>());
    ;
    }
    } // end namespace detail

    } // end namespace azplugins

    } // end namespace hoomd
