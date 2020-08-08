// Copyright (c) 2018-2020, Michael P. Howard
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
#include "DPDPotentials.h"
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

/* MPCD */
#ifdef ENABLE_MPCD
#include "MPCDReversePerturbationFlow.h"
#ifdef ENABLE_CUDA
#include "MPCDReversePerturbationFlowGPU.h"
#endif // ENABLE_CUDA
#endif // ENABLE_MPCD

/* Force computes */
#include "ImplicitEvaporator.h"
#include "ImplicitDropletEvaporator.h"
#include "ImplicitPlaneEvaporator.h"
#include "OrientationRestraintCompute.h"
#include "PositionRestraintCompute.h"
#include "WallRestraintCompute.h"
#ifdef ENABLE_CUDA
#include "ImplicitEvaporatorGPU.h"
#include "ImplicitDropletEvaporatorGPU.h"
#include "ImplicitPlaneEvaporatorGPU.h"
#include "OrientationRestraintComputeGPU.h"
#include "PositionRestraintComputeGPU.h"
#include "WallRestraintComputeGPU.h"
#endif // ENABLE_CUDA

/* Analyzers */
#include "RDFAnalyzer.h"
#ifdef ENABLE_CUDA
#include "RDFAnalyzerGPU.h"
#endif // ENABLE_CUDA

/* Integrators */
#include "BounceBackGeometry.h"
#include "BounceBackNVE.h"
#include "FlowFields.h"
#include "TwoStepBrownianFlow.h"
#include "TwoStepLangevinFlow.h"
#include "TwoStepSLLODCouette.h"
#ifdef ENABLE_CUDA
#include "BounceBackNVEGPU.h"
#include "TwoStepBrownianFlowGPU.h"
#include "TwoStepLangevinFlowGPU.h"
#endif // ENABLE_CUDA

/* Variants */
#include "VariantSphereArea.h"

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

PYBIND11_MODULE(_azplugins, m)
    {
    /* Pair potentials */
    azplugins::detail::export_pair_potential<azplugins::detail::PairEvaluatorAshbaugh>(m, "PairPotentialAshbaugh");
    azplugins::detail::export_pair_potential<azplugins::detail::PairEvaluatorAshbaugh24>(m, "PairPotentialAshbaugh24");
    azplugins::detail::export_ashbaugh_params(m);
    azplugins::detail::export_pair_potential<azplugins::detail::PairEvaluatorColloid>(m, "PairPotentialColloid");
    azplugins::detail::export_pair_potential<azplugins::detail::PairEvaluatorHertz>(m, "PairPotentialHertz");
    azplugins::detail::export_pair_potential<azplugins::detail::PairEvaluatorLJ124>(m, "PairPotentialLJ124");
    azplugins::detail::export_pair_potential<azplugins::detail::PairEvaluatorLJ96>(m,"PairPotentialLJ96");
    azplugins::detail::export_pair_potential<azplugins::detail::PairEvaluatorShiftedLJ>(m, "PairPotentialShiftedLJ");
    azplugins::detail::export_pair_potential<azplugins::detail::PairEvaluatorSpline>(m, "PairPotentialSpline");

    /* Anisotropic pair potentials */
    azplugins::detail::export_aniso_pair_potential<azplugins::detail::AnisoPairEvaluatorTwoPatchMorse>(m, "AnisoPairPotentialTwoPatchMorse");
    azplugins::detail::export_two_patch_morse_params(m);

    /* DPD potentials */
    azplugins::detail::export_dpd_potential<azplugins::detail::DPDEvaluatorGeneralWeight>(m, "DPDPotentialGeneralWeight");

    /* Bond potentials */
    azplugins::detail::export_bond_potential<azplugins::detail::BondEvaluatorDoubleWell>(m, "BondPotentialDoubleWell");
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
    azplugins::detail::export_PlaneWall(m);
    azplugins::detail::export_CylinderWall(m);
    azplugins::detail::export_SphereWall(m);

    /* Force computes */
    azplugins::detail::export_ImplicitEvaporator(m);
    azplugins::detail::export_ImplicitDropletEvaporator(m);
    azplugins::detail::export_ImplicitPlaneEvaporator(m);
    azplugins::detail::export_OrientationRestraintCompute(m);
    azplugins::detail::export_WallRestraintCompute<PlaneWall>(m,"PlaneRestraintCompute");
    azplugins::detail::export_WallRestraintCompute<CylinderWall>(m,"CylinderRestraintCompute");
    azplugins::detail::export_WallRestraintCompute<SphereWall>(m,"SphereRestraintCompute");
    azplugins::detail::export_PositionRestraintCompute(m);
    #ifdef ENABLE_CUDA
    azplugins::detail::export_ImplicitEvaporatorGPU(m);
    azplugins::detail::export_ImplicitDropletEvaporatorGPU(m);
    azplugins::detail::export_ImplicitPlaneEvaporatorGPU(m);
    azplugins::detail::export_OrientationRestraintComputeGPU(m);
    azplugins::detail::export_WallRestraintComputeGPU<PlaneWall>(m,"PlaneRestraintComputeGPU");
    azplugins::detail::export_WallRestraintComputeGPU<CylinderWall>(m,"CylinderRestraintComputeGPU");
    azplugins::detail::export_WallRestraintComputeGPU<SphereWall>(m,"SphereRestraintComputeGPU");
    azplugins::detail::export_PositionRestraintComputeGPU(m);
    #endif // ENABLE_CUDA

    /* MPCD */
    #ifdef ENABLE_MPCD
    azplugins::detail::export_MPCDReversePerturbationFlow(m);
    #ifdef ENABLE_CUDA
    azplugins::detail::export_MPCDReversePerturbationFlowGPU(m);
    #endif // ENABLE_CUDA
    #endif // ENABLE_MPCD

    /* Analyzers */
    azplugins::detail::export_RDFAnalyzer(m);
    #ifdef ENABLE_CUDA
    azplugins::detail::export_RDFAnalyzerGPU(m);
    #endif // ENABLE_CUDA

    /* Integrators */
    azplugins::detail::export_ConstantFlow(m);
    azplugins::detail::export_ParabolicFlow(m);
    azplugins::detail::export_QuiescentFluid(m);
    azplugins::detail::export_TwoStepBrownianFlow<azplugins::ConstantFlow>(m, "BrownianConstantFlow");
    azplugins::detail::export_TwoStepBrownianFlow<azplugins::ParabolicFlow>(m, "BrownianParabolicFlow");
    azplugins::detail::export_TwoStepBrownianFlow<azplugins::QuiescentFluid>(m, "BrownianQuiescentFluid");
    azplugins::detail::export_TwoStepLangevinFlow<azplugins::ConstantFlow>(m, "LangevinConstantFlow");
    azplugins::detail::export_TwoStepLangevinFlow<azplugins::ParabolicFlow>(m, "LangevinParabolicFlow");
    azplugins::detail::export_TwoStepLangevinFlow<azplugins::QuiescentFluid>(m, "LangevinQuiescentFluid");
    azplugins::detail::export_TwoStepSLLODCouette(m);
    #ifdef ENABLE_CUDA
    azplugins::detail::export_TwoStepBrownianFlowGPU<azplugins::ConstantFlow>(m, "BrownianConstantFlowGPU");
    azplugins::detail::export_TwoStepBrownianFlowGPU<azplugins::ParabolicFlow>(m, "BrownianParabolicFlowGPU");
    azplugins::detail::export_TwoStepBrownianFlowGPU<azplugins::QuiescentFluid>(m, "BrownianQuiescentFluidGPU");
    azplugins::detail::export_TwoStepLangevinFlowGPU<azplugins::ConstantFlow>(m, "LangevinConstantFlowGPU");
    azplugins::detail::export_TwoStepLangevinFlowGPU<azplugins::ParabolicFlow>(m, "LangevinParabolicFlowGPU");
    azplugins::detail::export_TwoStepLangevinFlowGPU<azplugins::QuiescentFluid>(m, "LangevinQuiescentFluidGPU");
    #endif // ENABLE_CUDA
    azplugins::detail::export_BounceBackNVE<mpcd::detail::SlitGeometry>(m);
    #ifdef ENABLE_CUDA
    azplugins::detail::export_BounceBackNVEGPU<mpcd::detail::SlitGeometry>(m);
    #endif // ENABLE_CUDA

    /* Variants */
    azplugins::detail::export_VariantSphereArea(m);
    }
