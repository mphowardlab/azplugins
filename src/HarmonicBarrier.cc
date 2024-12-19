// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file HarmonicBarrier.cc
 * \brief Definition of HarmonicBarrier
 */

#include "HarmonicBarrier.h"

namespace hoomd
    {

namespace azplugins
    {
/*!
 * \param sysdef System definition
 * \param interf Position of the interface
 */
HarmonicBarrier::HarmonicBarrier(std::shared_ptr<SystemDefinition> sysdef,
                                 std::shared_ptr<Variant> interf)
    : ForceCompute(sysdef), m_interf(interf), m_has_warned(false)
    {
    // allocate memory per type for parameters
    GPUArray<Scalar4> params(m_pdata->getNTypes(), m_exec_conf);
    m_params.swap(params);
    }

HarmonicBarrier::~HarmonicBarrier() { }

/*!
 * \param timestep Current timestep
 *
 * This method only checks for warnings about the virial.
 * Deriving classes should implement the actual force compute method.
 */
void HarmonicBarrier::computeForces(uint64_t timestep)
    {
    PDataFlags flags = m_pdata->getFlags();
    if (!m_has_warned
        && flags[pdata_flag::pressure_tensor])
        {
        m_exec_conf->msg->warning() << "HarmonicBarrier does not compute its virial "
                                       "contribution, pressure may be inaccurate"
                                    << std::endl;
        m_has_warned = true;
        }
    }

namespace detail
    {
/*!
 * \param m Python module to export to
 */
void export_HarmonicBarrier(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<HarmonicBarrier, std::shared_ptr<HarmonicBarrier>, ForceCompute>(m,
                                                                                "HarmonicBarrier")
        .def(py::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<Variant>>())
        .def("setParams", &HarmonicBarrier::setParamsPython)
        .def("getParams", &HarmonicBarrier::getParams);
    }
    } // end namespace detail

    } // end namespace azplugins

    } // end namespace hoomd
