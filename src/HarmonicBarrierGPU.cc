// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file HarmonicBarrierGPU.cc
 * \brief Definition of HarmonicBarrierGPU
 */

#include "HarmonicBarrierGPU.h"

namespace hoomd
    {

namespace azplugins
    {

/*!
 * \param sysdef System definition
 * \param interf Position of the interface
 */
HarmonicBarrierGPU::HarmonicBarrierGPU(std::shared_ptr<SystemDefinition> sysdef,
                                       std::shared_ptr<Variant> interf)
    : HarmonicBarrier(sysdef, interf)
    {
    m_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                                   this->m_exec_conf,
                                   "harmonic_barrier"));
    this->m_autotuners.push_back(m_tuner);
    }

namespace detail
    {
/*!
 * \param m Python module to export to
 */
void export_HarmonicBarrierGPU(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<HarmonicBarrierGPU, std::shared_ptr<HarmonicBarrierGPU>, HarmonicBarrier>(
        m,
        "HarmonicBarrierGPU")
        .def(py::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<Variant>>());
    }
    } // end namespace detail

    } // end namespace azplugins

    } // end namespace hoomd
