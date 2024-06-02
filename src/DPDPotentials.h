// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file DPDPotentials.h
 * \brief Convenience inclusion of all DPD potential evaluators.
 *
 * In HOOMD-blue, DPD potentials are templated on a base class ForceCompute called
 * PotentialPairDPDThermo, which uses an evaluator functor to compute the actual details of the DPD
 * potential. This avoids code duplication of calling the neighbor list, computing pair distances,
 * etc.
 *
 * To add a new DPD potential, take the following steps:
 *  1. Create an evaluator functor for your potential, for example DPDEvaluatorMyGreatPotential.h.
 *     This file should be included below. You can follow one of the other evaluator functors as
 *     an example for the details.
 *
 *  2. Explicitly instantiate a template for a CUDA driver for your potential in DPDPotentials.cu.
 *
 *  3. Expose the DPD pair potential on the python level in module.cc using export_dpd_potential and
 *     add the mirror python object to dpd.py.
 *
 *  4. Write a unit test for the potential in test-py.
 */

#ifndef AZPLUGINS_DPD_POTENTIALS_H_
#define AZPLUGINS_DPD_POTENTIALS_H_

// All DPD potential evaluators must be included here
#include "DPDEvaluatorGeneralWeight.h"

/*
 * The code below handles python exports using a templated function, and so
 * should not be compiled in NVCC.
 */
#ifndef NVCC
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"
namespace py = pybind11;

#include "hoomd/md/PotentialPairDPDThermo.h"
#ifdef ENABLE_CUDA
#include "DPDPotentials.cuh"
#include "hoomd/md/PotentialPairDPDThermoGPU.h"
#endif

namespace azplugins
    {
namespace detail
    {
//! Exports the DPD potential to the python module
template<class evaluator> void export_dpd_potential(py::module& m, const std::string& name)
    {
    typedef ::PotentialPair<evaluator> base_cpu;
    export_PotentialPair<base_cpu>(m, name + "Base");

    typedef ::PotentialPairDPDThermo<evaluator> dpd_cpu;
    export_PotentialPairDPDThermo<dpd_cpu, base_cpu>(m, name);

#ifdef ENABLE_CUDA
    typedef ::PotentialPairDPDThermoGPU<evaluator, azplugins::gpu::compute_dpd_potential<evaluator>>
        dpd_gpu;
    export_PotentialPairDPDThermoGPU<dpd_gpu, dpd_cpu>(m, name + "GPU");
#endif // ENABLE_CUDA
    }
    }  // end namespace detail
    }  // end namespace azplugins
#endif // NVCC

#endif // AZPLUGINS_DPD_POTENTIALS_H_
