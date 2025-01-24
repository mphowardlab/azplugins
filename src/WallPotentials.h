// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file WallPotentials.h
 * \brief Convenience inclusion of all wall potential evaluators.
 *
 * In HOOMD-blue, wall potentials are templated on a base class ForceCompute called
 * PotentialExternal (via a templated EvaluatorWall functor). This avoids code duplication of basic
 * routines.
 *
 * To add a new wall potential, take the following steps:
 *  1. Create an evaluator functor for your potential, for example WallEvaluatorMyGreatPotential.h.
 *     This file should be added to the list of includes below. You can follow one of the other
 *     evaluator functors as an example for the details.
 *
 *  2. Explicitly instantiate a template for a CUDA driver for your potential in WallPotentials.cu.
 *
 *  3. Expose the wall potential on the python level in module.cc with export_wall_potential and add
 *     the mirror python object to wall.py.
 *
 *  4. Write a unit test for the potential in test-py. Two types of tests should be conducted: one
 * that checks that all methods work on the python object, and one that validates the force and
 * energy for the particle at a fixed distance from the wall.
 */

#ifndef AZPLUGINS_WALL_POTENTIALS_H_
#define AZPLUGINS_WALL_POTENTIALS_H_

// All wall potential evaluators should be included here
#include "WallEvaluatorColloid.h"
#include "WallEvaluatorLJ93.h"

/*
 * The code below handles python exports using a templated function, and so should
 * not be compiled in NVCC.
 */
#ifndef NVCC
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

#include "hoomd/md/EvaluatorWalls.h"
#include "hoomd/md/PotentialExternal.h"
#ifdef ENABLE_CUDA
#include "hoomd/md/PotentialExternalGPU.h"
#endif

namespace azplugins
    {
namespace detail
    {
//! Convenience function to export wall potential to python
/*!
 * \param m Python module
 * \param name Name of CPU class
 * \tparam evaluator Wall potential evaluator functor
 *
 * A CPU object called \a name is exported to python, along with an
 * object for the wall parameters (name_params) and convenience method
 * for creating the parameters (make_name_params()). If CUDA is enabled,
 * a corresponding GPU class called nameGPU is exported.
 */
template<class evaluator> void export_wall_potential(pybind11::module& m, const std::string& name)
    {
    namespace py = pybind11;
    typedef ::EvaluatorWalls<evaluator> wall_evaluator;
    typedef ::PotentialExternal<wall_evaluator> wall_potential_cpu;
    export_PotentialExternal<wall_potential_cpu>(m, name);

#ifdef ENABLE_CUDA
    typedef ::PotentialExternalGPU<wall_evaluator> wall_potential_gpu;
    export_PotentialExternalGPU<wall_potential_gpu, wall_potential_cpu>(m, name + "GPU");
#endif // ENABLE_CUDA

    py::class_<typename wall_evaluator::param_type,
               std::shared_ptr<typename wall_evaluator::param_type>>(
        m,
        (wall_evaluator::getName() + "_params").c_str())
        .def(py::init<>())
        .def_readwrite("params", &wall_evaluator::param_type::params)
        .def_readwrite("rextrap", &wall_evaluator::param_type::rextrap)
        .def_readwrite("rcutsq", &wall_evaluator::param_type::rcutsq);
    m.def(std::string("make_" + wall_evaluator::getName() + "_params").c_str(),
          &make_wall_params<evaluator>);
    }

    } // end namespace detail
    } // end namespace azplugins
#endif // NVCC

#endif // AZPLUGINS_WALL_POTENTIALS_H_
