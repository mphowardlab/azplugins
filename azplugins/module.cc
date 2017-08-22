// Copyright (c) 2015-2017, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward / Everyone is free to add additional objects

/*!
 * \file module.cc
 * \brief Export classes to python
 */

#include "hoomd/extern/pybind/include/pybind11/pybind11.h"
namespace py = pybind11;

#include "AnisoPairPotentials.h"
#include "PairPotentials.h"
#include "WallPotentials.h"

/* Updaters */
#include "TypeUpdater.h"
#include "ParticleEvaporator.h"
#ifdef ENABLE_CUDA
#include "TypeUpdaterGPU.h"
#include "ParticleEvaporatorGPU.h"
#endif // ENABLE_CUDA

/* Force computes */
#include "ImplicitEvaporator.h"
#include "OrientationRestraintCompute.h"
#include "PositionRestraintCompute.h"
#ifdef ENABLE_CUDA
#include "ImplicitEvaporatorGPU.h"
#include "OrientationRestraintComputeGPU.h"
#include "PositionRestraintComputeGPU.h"
#endif // ENABLE_CUDA

/* Analyzers */
#include "RDFAnalyzer.h"
#ifdef ENABLE_CUDA
#include "RDFAnalyzerGPU.h"
#endif // ENABLE_CUDA

//! Plugins for soft matter
namespace azplugins
{
//! azplugins implementation details
/*!
 * Classes, functions, and data structures that internally implement parts of the
 * plugins. These details are not part of the public interface, and may change at
 * any time.
 */
namespace detail
{

//! Helper function export the Ashbaugh-Hatch pair potential parameters
/*!
* \sa ashbaugh_params
*/
void export_ashbaugh_params(py::module& m)
    {
    py::class_<ashbaugh_params>(m, "ashbaugh_params")
    .def(py::init<>())
    .def_readwrite("lj1", &ashbaugh_params::lj1)
    .def_readwrite("lj2", &ashbaugh_params::lj2)
    .def_readwrite("lam", &ashbaugh_params::lambda)
    .def_readwrite("rwcasq", &ashbaugh_params::rwcasq)
    .def_readwrite("wca_shift", &ashbaugh_params::wca_shift)
    ;
    m.def("make_ashbaugh_params", &make_ashbaugh_params);
    }

} // end namespace detail

// document other namespaces that may crop up in other parts of the package
//! azplugins gpu implementations
/*!
 * Driver functions for plugin kernels. These driver functions are
 * not part of the public interface, and may change at any time.
 */
namespace gpu
{

//! azplugins gpu kernels
/*!
 * CUDA kernels to implement GPU pathways. These kernels are not
 * part of the public interface, and may change at any time.
 */
namespace kernel
{
} // end namespace kernel

} // end namespace gpu
} // end namespace azplugins


PYBIND11_PLUGIN(_azplugins)
    {
    pybind11::module m("_azplugins");

    /* Pair potentials */
    azplugins::detail::export_pair_potential<azplugins::detail::PairEvaluatorAshbaugh>(m, "PairPotentialAshbaugh");
    azplugins::detail::export_ashbaugh_params(m);
    azplugins::detail::export_pair_potential<azplugins::detail::PairEvaluatorColloid>(m, "PairPotentialColloid");
    azplugins::detail::export_pair_potential<azplugins::detail::PairEvaluatorLJ124>(m, "PairPotentialLJ124");
    azplugins::detail::export_pair_potential<azplugins::detail::PairEvaluatorLJ96>(m,"PairPotentialLJ96");
    azplugins::detail::export_pair_potential<azplugins::detail::PairEvaluatorShiftedLJ>(m, "PairPotentialShiftedLJ");

    /* Anisotropic pair potentials */

    /* Updaters */
    azplugins::detail::export_TypeUpdater(m);
    azplugins::detail::export_ParticleEvaporator(m); // this must follow TypeUpdater because TypeUpdater is the python base class
    #ifdef ENABLE_CUDA
    azplugins::detail::export_TypeUpdaterGPU(m);
    azplugins::detail::export_ParticleEvaporatorGPU(m);
    #endif // ENABLE_CUDA

    /* Wall potentials */
    azplugins::detail::export_wall_potential<azplugins::detail::WallEvaluatorColloid>(m, "WallPotentialColloid");
    azplugins::detail::export_wall_potential<azplugins::detail::WallEvaluatorLJ93>(m, "WallPotentialLJ93");

    /* Force computes */
    azplugins::detail::export_ImplicitEvaporator(m);
    azplugins::detail::export_OrientationRestraintCompute(m);
    azplugins::detail::export_PositionRestraintCompute(m);
    #ifdef ENABLE_CUDA
    azplugins::detail::export_ImplicitEvaporatorGPU(m);
    azplugins::detail::export_OrientationRestraintComputeGPU(m);
    azplugins::detail::export_PositionRestraintComputeGPU(m);
    #endif // ENABLE_CUDA

    /* Analyzers */
    azplugins::detail::export_RDFAnalyzer(m);
    #ifdef ENABLE_CUDA
    azplugins::detail::export_RDFAnalyzerGPU(m);
    #endif // ENABLE_CUDA

    return m.ptr();
    }
