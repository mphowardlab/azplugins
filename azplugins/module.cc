// Copyright (c) 2015-2017, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward / Everyone is free to add additional objects

/*!
 * \file module.cc
 * \brief Export classes to python
 */

#include "hoomd/extern/pybind/include/pybind11/pybind11.h"
namespace py = pybind11;

/* Potentials */
#include "AnisoPairPotentials.h"
#include "BondPotentials.h"
#include "PairPotentials.h"
#include "SpecialPairPotentials.h"
#include "WallPotentials.h"

/* Updaters */
#include "ReversePerturbationFlow.h"
#include "TypeUpdater.h"
#include "ParticleEvaporator.h"
#ifdef ENABLE_CUDA
#include "ReversePerturbationFlowGPU.h"
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

/* Integrators */
#include "BounceBackGeometry.h"
#include "BounceBackNVE.h"
#ifdef ENABLE_CUDA
#include "BounceBackNVEGPU.h"
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
//! Helper function export the Ashbaugh-Hatch-24-48 bond potential parameters
/*!
* \sa ashbaugh_params
*/
void export_ashbaugh_bond_params(py::module& m)
    {
    py::class_<ashbaugh_bond_params>(m, "ashbaugh_bond_params")
    .def(py::init<>())
    .def_readwrite("lj1", &ashbaugh_bond_params::lj1)
    .def_readwrite("lj2", &ashbaugh_bond_params::lj2)
    .def_readwrite("lam", &ashbaugh_bond_params::lambda)
    .def_readwrite("rwcasq", &ashbaugh_bond_params::rwcasq)
    .def_readwrite("wca_shift", &ashbaugh_bond_params::wca_shift)
    .def_readwrite("K", &ashbaugh_bond_params::K)
    .def_readwrite("r0", &ashbaugh_bond_params::r_0)
    ;
    m.def("make_ashbaugh_bond_params", &make_ashbaugh_bond_params);
    }

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

//! Helper function export the Two-Patch Morse pair potential parameters
/*!
* \sa two_patch_morse_params
*/
void export_two_patch_morse_params(py::module& m)
    {
    py::class_<two_patch_morse_params>(m, "two_patch_morse_params")
    .def(py::init<>())
    .def_readwrite("Mdeps", &two_patch_morse_params::Mdeps)
    .def_readwrite("Mrinv", &two_patch_morse_params::Mrinv)
    .def_readwrite("req", &two_patch_morse_params::req)
    .def_readwrite("omega", &two_patch_morse_params::omega)
    .def_readwrite("alpha", &two_patch_morse_params::alpha)
    ;
    m.def("make_two_patch_morse_params", &make_two_patch_morse_params);
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
    azplugins::detail::export_pair_potential<azplugins::detail::PairEvaluatorAshbaugh24>(m, "PairPotentialAshbaugh24");
    azplugins::detail::export_ashbaugh_params(m);
    azplugins::detail::export_pair_potential<azplugins::detail::PairEvaluatorColloid>(m, "PairPotentialColloid");
    azplugins::detail::export_pair_potential<azplugins::detail::PairEvaluatorLJ124>(m, "PairPotentialLJ124");
    azplugins::detail::export_pair_potential<azplugins::detail::PairEvaluatorLJ96>(m,"PairPotentialLJ96");
    azplugins::detail::export_pair_potential<azplugins::detail::PairEvaluatorShiftedLJ>(m, "PairPotentialShiftedLJ");
    azplugins::detail::export_pair_potential<azplugins::detail::PairEvaluatorSpline>(m, "PairPotentialSpline");

    /* Anisotropic pair potentials */
    azplugins::detail::export_aniso_pair_potential<azplugins::detail::AnisoPairEvaluatorTwoPatchMorse>(m, "AnisoPairPotentialTwoPatchMorse");
    azplugins::detail::export_two_patch_morse_params(m);

    /* Bond potentials */
    azplugins::detail::export_bond_potential<azplugins::detail::BondEvaluatorFENE>(m, "BondPotentialFENE");
    azplugins::detail::export_bond_potential<azplugins::detail::BondEvaluatorFENEAsh24>(m, "BondPotentialFENEAsh24");
    azplugins::detail::export_ashbaugh_bond_params(m);
    /* Special pair potentials */
    azplugins::detail::export_special_pair_potential<azplugins::detail::PairEvaluatorLJ96>(m,"SpecialPairPotentialLJ96");

    /* Updaters */
    azplugins::detail::export_ReversePerturbationFlow(m);
    azplugins::detail::export_TypeUpdater(m);
    azplugins::detail::export_ParticleEvaporator(m); // this must follow TypeUpdater because TypeUpdater is the python base class
    #ifdef ENABLE_CUDA
    azplugins::detail::export_ReversePerturbationFlowGPU(m);
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

    /* Integrators */
    azplugins::detail::export_boundary(m);
    azplugins::detail::export_SlitGeometry(m);
    azplugins::detail::export_BounceBackNVE<mpcd::detail::SlitGeometry>(m);
    #ifdef ENABLE_CUDA
    azplugins::detail::export_BounceBackNVEGPU<mpcd::detail::SlitGeometry>(m);
    #endif // ENABLE_CUDA

    return m.ptr();
    }
