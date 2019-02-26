// Copyright (c) 2018-2019, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward / Everyone is free to add additional potentials

/*!
 * \file SpecialPairPotentials.h
 * \brief Convenience inclusion of all special pair potential evaluators.
 *
 * In HOOMD-blue, special pair potentials are templated on a base class ForceCompute called PotentialSpecialPair,
 * which uses an evaluator functor to compute the actual details of the special pair potential.
 *
 * All special pair potentials must be templated on an existing pair potential evaluator. See PairPotentials.h
 * for more details on writing such an evaluator.
 *
 * To add a new special pair potential, take the following steps:
 *  1. Explicitly instantiate a template for a CUDA driver for your potential in SpecialPairPotentials.cu.
 *
 *  2. Expose the special pair potential on the python level in module.cc using export_special_pair_potential and
 *     add the mirror python object to special_pair.py.
 *
 *  3. Write a unit test for the potential in test-py. Two types of tests should be conducted: one that
 *     checks that all methods work on the python object, and one that validates the force and energy between
 *     particle pairs at fixed distances.
 */

#ifndef AZPLUGINS_SPECIAL_PAIR_POTENTIALS_H_
#define AZPLUGINS_SPECIAL_PAIR_POTENTIALS_H_

#include "SpecialPairEvaluator.h"
#include "PairPotentials.h"

/*
 * The code below handles python exports using a templated function, and so
 * should not be compiled in NVCC.
 */
#ifndef NVCC
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"
namespace py = pybind11;

#include "hoomd/md/PotentialSpecialPair.h"
#ifdef ENABLE_CUDA
#include "hoomd/md/PotentialSpecialPairGPU.h"
#include "SpecialPairPotentials.cuh"
#endif

namespace azplugins
{
namespace detail
{
//! Helper function export the special pair potential parameters
/*!
* \sa special_pair_params
*/
template<class evaluator>
void export_special_pair_params(py::module& m)
{
    const std::string name = "special_pair_params_" + evaluator::getName();

    py::class_<typename SpecialPairEvaluator<evaluator>::param_type>(m, name.c_str())
    .def(py::init<>())
    .def_readwrite("params", &SpecialPairEvaluator<evaluator>::param_type::params)
    .def_readwrite("rcutsq", &SpecialPairEvaluator<evaluator>::param_type::rcutsq)
    .def_readwrite("energy_shift", &SpecialPairEvaluator<evaluator>::param_type::energy_shift)
    ;
    m.def(("make_" + name).c_str(), &make_special_pair_params<evaluator>);
}

//! Exports the special pair potential to the python module
template<class evaluator>
void export_special_pair_potential(py::module& m, const std::string& name)
    {
    typedef SpecialPairEvaluator<evaluator> special_evaluator;
    typedef ::PotentialSpecialPair<special_evaluator> pair_potential_cpu;
    export_PotentialSpecialPair<pair_potential_cpu>(m, name);

    #ifdef ENABLE_CUDA
    typedef ::PotentialSpecialPairGPU<special_evaluator, azplugins::gpu::compute_special_pair_potential<special_evaluator>> pair_potential_gpu;
    export_PotentialSpecialPairGPU<pair_potential_gpu, pair_potential_cpu>(m, name + "GPU");
    #endif // ENABLE_CUDA

    export_special_pair_params<evaluator>(m);
    }

} // end namespace detail
} // end namespace azplugins
#endif // NVCC

#endif // AZPLUGINS_SPECIAL_PAIR_POTENTIALS_H_
