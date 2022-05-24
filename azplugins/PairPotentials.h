// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2022, Auburn University
// This file is part of the azplugins project, released under the Modified BSD License.

/*!
 * \file PairPotentials.h
 * \brief Convenience inclusion of all pair potential evaluators.
 *
 * In HOOMD-blue, pair potentials are templated on a base class ForceCompute called PotentialPair,
 * which uses an evaluator functor to compute the actual details of the pair potential.
 * This avoids code duplication of calling the neighbor list, computing pair distances, etc.
 *
 * To add a new pair potential, take the following steps:
 *  1. Create an evaluator functor for your potential, for example PairEvaluatorMyGreatPotential.h.
 *     This file should be included below. You can follow one of the other evaluator functors as
 *     an example for the details.
 *
 *  2. Explicitly instantiate a template for a CUDA driver for your potential in its own .cu file. Add this
 *     file to CMakeLists.txt.
 *
 *  3. Expose the pair potential on the python level in module.cc using export_pair_potential and
 *     add the mirror python object to pair.py.
 *
 *  4. Write a unit test for the potential in test-py. Two types of tests should be conducted: one that
 *     checks that all methods work on the python object, and one that validates the force and energy between
 *     particle pairs at fixed distances.
 */

#ifndef AZPLUGINS_PAIR_POTENTIALS_H_
#define AZPLUGINS_PAIR_POTENTIALS_H_

// All pair potential evaluators must be included here
#include "PairEvaluatorAshbaugh.h"
#include "PairEvaluatorAshbaugh24.h"
#include "PairEvaluatorColloid.h"
#include "PairEvaluatorHertz.h"
#include "PairEvaluatorLJ124.h"
#include "PairEvaluatorLJ96.h"
#include "PairEvaluatorShiftedLJ.h"
#include "PairEvaluatorSpline.h"

/*
 * The code below handles python exports using a templated function, and so
 * should not be compiled in NVCC.
 */
#ifndef NVCC
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

#include "hoomd/md/PotentialPair.h"
#ifdef ENABLE_CUDA
#include "hoomd/md/PotentialPairGPU.h"
#include "PairPotentials.cuh"
#endif

namespace azplugins
{
namespace detail
{
//! Exports the pair potential to the python module
template<class evaluator>
void export_pair_potential(pybind11::module& m, const std::string& name)
    {
    typedef ::PotentialPair<evaluator> pair_potential_cpu;
    export_PotentialPair<pair_potential_cpu>(m, name);

    #ifdef ENABLE_CUDA
    typedef ::PotentialPairGPU<evaluator, azplugins::gpu::compute_pair_potential<evaluator> > pair_potential_gpu;
    export_PotentialPairGPU<pair_potential_gpu, pair_potential_cpu>(m, name + "GPU");
    #endif // ENABLE_CUDA
    }
} // end namespace detail
} // end namespace azplugins
#endif // NVCC

#endif // AZPLUGINS_PAIR_POTENTIALS_H_
