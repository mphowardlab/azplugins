// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file MovingHarmonicPotential.cc
 * \brief Definition of MovingHarmonicPotential
 */

#include "MovingHarmonicPotential.h"

namespace hoomd
    {

namespace azplugins
    {
/*!
 * \param sysdef System definition
 * \param interf Position of the interface
 */
MovingHarmonicPotential::MovingHarmonicPotential(std::shared_ptr<SystemDefinition> sysdef,
                                                 std::shared_ptr<Variant> interf)
    : ForceCompute(sysdef), m_interf(interf), m_has_warned(false)
    {
    // allocate memory per type for parameters
    GPUArray<Scalar4> params(m_pdata->getNTypes(), m_exec_conf);
    m_params.swap(params);

    // connect to type change to resize type data arrays
//    m_pdata->getNumTypesChangeSignal()
//        .connect<MovingHarmonicPotential, &MovingHarmonicPotential::reallocateParams>(this);
    }

MovingHarmonicPotential::~MovingHarmonicPotential()
    {
//    m_pdata->getNumTypesChangeSignal()
//        .disconnect<MovingHarmonicPotential, &MovingHarmonicPotential::reallocateParams>(this);
    }

/*!
 * \param timestep Current timestep
 *
 * This method only checks for warnings about the virial.
 * Deriving classes should implement the actual force compute method.
 */
void MovingHarmonicPotential::computeForces(unsigned int timestep)
    {
    PDataFlags flags = m_pdata->getFlags();
    if (!m_has_warned
        && flags[pdata_flag::pressure_tensor])
        {
        m_exec_conf->msg->warning() << "MovingHarmonicPotential does not compute its virial "
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
void export_MovingHarmonicPotential(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<MovingHarmonicPotential, std::shared_ptr<MovingHarmonicPotential>>(m,
                                                                                  "MovingHarmonicPotential",
                                                                                  py::base<ForceCompute>())
        .def(py::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<Variant>>())
        .def("setParams", &MovingHarmonicPotential::setParams);
    }
    } // end namespace detail

    } // end namespace azplugins

    } // end namespace hoomd
