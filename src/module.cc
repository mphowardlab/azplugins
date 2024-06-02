// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

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
#include "WallPotentials.h"

/* Updaters */
#include "ParticleEvaporator.h"
#include "TypeUpdater.h"
#ifdef ENABLE_CUDA
#include "ParticleEvaporatorGPU.h"
#include "TypeUpdaterGPU.h"
#endif // ENABLE_CUDA

/* MPCD */
#ifdef ENABLE_MPCD
#include "MPCDVelocityCompute.h"
#endif // ENABLE_MPCD

/* Force computes */
#include "ImplicitDropletEvaporator.h"
#include "ImplicitEvaporator.h"
#include "ImplicitPlaneEvaporator.h"
#ifdef ENABLE_CUDA
#include "ImplicitDropletEvaporatorGPU.h"
#include "ImplicitEvaporatorGPU.h"
#include "ImplicitPlaneEvaporatorGPU.h"
#endif // ENABLE_CUDA

/* Analyzers */
#include "GroupVelocityCompute.h"

/* Integrators */
#include "FlowFields.h"
#include "TwoStepBrownianFlow.h"
#include "TwoStepLangevinFlow.h"
#ifdef ENABLE_CUDA
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
        .def_readwrite("wca_shift", &ashbaugh_params::wca_shift);
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
        .def_readwrite("alpha", &two_patch_morse_params::alpha);
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
    azplugins::detail::export_pair_potential<azplugins::detail::PairEvaluatorAshbaugh>(
        m,
        "PairPotentialAshbaugh");
    azplugins::detail::export_ashbaugh_params(m);
    azplugins::detail::export_pair_potential<azplugins::detail::PairEvaluatorColloid>(
        m,
        "PairPotentialColloid");
    azplugins::detail::export_pair_potential<azplugins::detail::PairEvaluatorHertz>(
        m,
        "PairPotentialHertz");

    /* Anisotropic pair potentials */
    azplugins::detail::export_aniso_pair_potential<
        azplugins::detail::AnisoPairEvaluatorTwoPatchMorse>(m, "AnisoPairPotentialTwoPatchMorse");
    azplugins::detail::export_two_patch_morse_params(m);

    /* DPD potentials */
    azplugins::detail::export_dpd_potential<azplugins::detail::DPDEvaluatorGeneralWeight>(
        m,
        "DPDPotentialGeneralWeight");

    /* Bond potentials */
    azplugins::detail::export_bond_potential<azplugins::detail::BondEvaluatorDoubleWell>(
        m,
        "BondPotentialDoubleWell");

    /* Updaters */
    azplugins::detail::export_TypeUpdater(m);
    azplugins::detail::export_ParticleEvaporator(
        m); // this must follow TypeUpdater because TypeUpdater is the python base class
#ifdef ENABLE_CUDA
    azplugins::detail::export_TypeUpdaterGPU(m);
    azplugins::detail::export_ParticleEvaporatorGPU(m);
#endif // ENABLE_CUDA

    /* Wall potentials */
    azplugins::detail::export_wall_potential<azplugins::detail::WallEvaluatorColloid>(
        m,
        "WallPotentialColloid");
    azplugins::detail::export_wall_potential<azplugins::detail::WallEvaluatorLJ93>(
        m,
        "WallPotentialLJ93");
    azplugins::detail::export_PlaneWall(m);
    azplugins::detail::export_CylinderWall(m);
    azplugins::detail::export_SphereWall(m);

    /* Force computes */
    azplugins::detail::export_ImplicitEvaporator(m);
    azplugins::detail::export_ImplicitDropletEvaporator(m);
    azplugins::detail::export_ImplicitPlaneEvaporator(m);
#ifdef ENABLE_CUDA
    azplugins::detail::export_ImplicitEvaporatorGPU(m);
    azplugins::detail::export_ImplicitDropletEvaporatorGPU(m);
    azplugins::detail::export_ImplicitPlaneEvaporatorGPU(m);
#endif // ENABLE_CUDA

/* MPCD components */
#ifdef ENABLE_MPCD
    azplugins::detail::export_MPCDVelocityCompute(m);
#endif // ENABLE_MPCD

    /* Analyzers */
    azplugins::detail::export_GroupVelocityCompute(m);

    /* Integrators */
    azplugins::detail::export_ConstantFlow(m);
    azplugins::detail::export_ParabolicFlow(m);
    azplugins::detail::export_QuiescentFluid(m);
    azplugins::detail::export_TwoStepBrownianFlow<azplugins::ConstantFlow>(m,
                                                                           "BrownianConstantFlow");
    azplugins::detail::export_TwoStepBrownianFlow<azplugins::ParabolicFlow>(
        m,
        "BrownianParabolicFlow");
    azplugins::detail::export_TwoStepBrownianFlow<azplugins::QuiescentFluid>(
        m,
        "BrownianQuiescentFluid");
    azplugins::detail::export_TwoStepLangevinFlow<azplugins::ConstantFlow>(m,
                                                                           "LangevinConstantFlow");
    azplugins::detail::export_TwoStepLangevinFlow<azplugins::ParabolicFlow>(
        m,
        "LangevinParabolicFlow");
    azplugins::detail::export_TwoStepLangevinFlow<azplugins::QuiescentFluid>(
        m,
        "LangevinQuiescentFluid");
#ifdef ENABLE_CUDA
    azplugins::detail::export_TwoStepBrownianFlowGPU<azplugins::ConstantFlow>(
        m,
        "BrownianConstantFlowGPU");
    azplugins::detail::export_TwoStepBrownianFlowGPU<azplugins::ParabolicFlow>(
        m,
        "BrownianParabolicFlowGPU");
    azplugins::detail::export_TwoStepBrownianFlowGPU<azplugins::QuiescentFluid>(
        m,
        "BrownianQuiescentFluidGPU");
    azplugins::detail::export_TwoStepLangevinFlowGPU<azplugins::ConstantFlow>(
        m,
        "LangevinConstantFlowGPU");
    azplugins::detail::export_TwoStepLangevinFlowGPU<azplugins::ParabolicFlow>(
        m,
        "LangevinParabolicFlowGPU");
    azplugins::detail::export_TwoStepLangevinFlowGPU<azplugins::QuiescentFluid>(
        m,
        "LangevinQuiescentFluidGPU");
#endif // ENABLE_CUDA

    /* Variants */
    azplugins::detail::export_VariantSphereArea(m);
    }
