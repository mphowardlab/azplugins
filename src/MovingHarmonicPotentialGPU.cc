// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file MovingHarmonicPotentialGPU.cc
 * \brief Definition of MovingHarmonicPotentialGPU
 */

#include "MovingHarmonicPotentialGPU.h"

namespace hoomd
    {

namespace azplugins
    {

/*!
 * \param sysdef System definition
 * \param interf Position of the interface
 */
MovingHarmonicPotentialGPU::MovingHarmonicPotentialGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                       std::shared_ptr<Variant> interf)
    : MovingHarmonicPotential(sysdef, interf)
    {
    m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "implicit_evap", m_exec_conf));
    }

namespace detail
    {
/*!
 * \param m Python module to export to
 */
void export_MovingHarmonicPotentialGPU(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<MovingHarmonicPotentialGPU, std::shared_ptr<MovingHarmonicPotentialGPU>>(
        m,
        "MovingHarmonicPotentialGPU",
        py::base<MovingHarmonicPotential>())
        .def(py::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<Variant>>());
    }
    } // end namespace detail

    } // end namespace azplugins

    } // end namespace hoomd
