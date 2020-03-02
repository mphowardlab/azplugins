// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward / Everyone is free to add additional potentials

/*!
 * \file AnisoPairPotentials.h
 * \brief Convenience inclusion of all anisotropic pair potential evaluators.
 *
 * In HOOMD-blue, anisotropic pair potentials are templated on a base class ForceCompute called
 * AnisoPotentialPair, which uses an evaluator functor to compute the actual details of the pair potential.
 * This avoids code duplication of calling the neighbor list, computing pair distances, etc.
 *
 * To add a new anisotropic pair potential, take the following steps:
 *  1. Create an evaluator functor for your potential, for example AnisoPairEvaluatorMyGreatPotential.h.
 *     This file should be included below. You can follow one of the other evaluator functors as
 *     an example for the details.
 *
 *  2. Explicitly instantiate a template for a CUDA driver for your potential in AnisoPairPotentials.cu.
 *
 *  3. Expose the anisotropic pair potential on the python level in module.cc using export_aniso_pair_potential
 *     and add the mirror python object to pair.py.
 *
 *  4. Write a unit test for the potential in test-py. Two types of tests should be conducted: one that
 *     checks that all methods work on the python object, and one that validates the force and energy between
 *     particle pairs at fixed distances.
 */

#ifndef AZPLUGINS_ANISO_PAIR_POTENTIALS_H_
#define AZPLUGINS_ANISO_PAIR_POTENTIALS_H_

// All anisotropic pair potential evaluators must be included here
#include "AnisoPairEvaluatorTwoPatchMorse.h"

/*
 * The code below handles python exports using a templated function, and so
 * should not be compiled in NVCC.
 */
#ifndef NVCC
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"
namespace py = pybind11;

#include "hoomd/md/AnisoPotentialPair.h"
#ifdef ENABLE_CUDA
#include "hoomd/md/AnisoPotentialPairGPU.h"
#include "AnisoPairPotentials.cuh"
#endif

namespace azplugins
{
namespace detail
{
//! Exports the anisotropic pair potential to the python module
template<class evaluator>
void export_aniso_pair_potential(py::module& m, const std::string& name)
    {
    typedef ::AnisoPotentialPair<evaluator> pair_potential_cpu;
    export_AnisoPotentialPair<pair_potential_cpu>(m, name);

    #ifdef ENABLE_CUDA
    typedef ::AnisoPotentialPairGPU<evaluator, azplugins::gpu::compute_aniso_pair_potential<evaluator> > pair_potential_gpu;
    export_AnisoPotentialPairGPU<pair_potential_gpu, pair_potential_cpu>(m, name + "GPU");
    #endif // ENABLE_CUDA
    }
} // end namespace detail
} // end namespace azplugins
#endif // NVCC

#endif // AZPLUGINS_ANISO_PAIR_POTENTIALS_H_
