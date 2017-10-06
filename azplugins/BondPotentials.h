// Copyright (c) 2016-2017, Panagiotopoulos Group, Princeton University
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: astatt / Everyone is free to add additional potentials

/*!
 * \file BondPotentials.h
 * \brief Convenience inclusion of all bonded potential evaluators.
 *
 * In HOOMD-blue, bond potentials are templated on a base class ForceCompute called PotentialBond,
 * which uses an evaluator functor to compute the actual details of the bond potential.
 *
 * To add a new bond potential, take the following steps:
 *  1. Create an evaluator functor for your potential, for example BondPotentialEvaluatorMyGreatPotential.h.
 *     This file should be included below. You can follow one of the other evaluator functors as
 *     an example for the details.
 *
 *  2. Explicitly instantiate a template for a CUDA driver for your potential in BondPotentials.cu.
 *
 *  3. Expose the bond potential on the python level in module.cc using export_bond_potential and
 *     add the mirror python object to bond.py.
 *
 *  4. Write a unit test for the potential in test-py. Two types of tests should be conducted: one that
 *     checks that all methods work on the python object, and one that validates the force and energy between
 *     particle pairs at fixed distances.
 */

#ifndef AZPLUGINS_BOND_POTENTIALS_H_
#define AZPLUGINS_BOND_POTENTIALS_H_

// All bonded potential evaluators must be included here
#include "BondEvaluatorFENE.h"

/*
 * The code below handles python exports using a templated function, and so
 * should not be compiled in NVCC.
 */
#ifndef NVCC
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"
namespace py = pybind11;

#include "hoomd/md/PotentialBond.h"
#ifdef ENABLE_CUDA
#include "hoomd/md/PotentialBondGPU.h"
#include "BondPotentials.cuh"
#endif

namespace azplugins
{
namespace detail
{
//! Exports the bonded potential to the python module
template<class evaluator>
void export_bond_potential(py::module& m, const std::string& name)
    {
    typedef ::PotentialBond<evaluator> bond_potential_cpu;
    export_PotentialBond<bond_potential_cpu>(m, name);

    #ifdef ENABLE_CUDA
    typedef ::PotentialBondGPU<evaluator, azplugins::gpu::compute_bond_potential<evaluator> > bond_potential_gpu;
    export_PotentialBondGPU<bond_potential_gpu, bond_potential_cpu>(m, name + "GPU");
    #endif // ENABLE_CUDA
    }
} // end namespace detail
} // end namespace azplugins
#endif // NVCC

#endif // AZPLUGINS_BOND_POTENTIALS_H_
