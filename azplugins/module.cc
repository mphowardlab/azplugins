// Maintainer: mphoward / Everyone is free to add additional objects

/*!
 * \file module.cc
 * \brief Export classes to python
 */

#include "hoomd/extern/pybind/include/pybind11/pybind11.h"
namespace py = pybind11;

/* Pair potential includes */
#include "PairPotentials.h"
#include "hoomd/md/PotentialPair.h"
#ifdef ENABLE_CUDA
#include "hoomd/md/PotentialPairGPU.h"
#endif // ENABLE_CUDA

/* Wall potential includes */
#include "WallPotentials.h"

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
    export_PotentialPair<azplugins::PairPotentialAshbaugh>(m, "PairPotentialAshbaugh");
    export_PotentialPair<azplugins::PairPotentialColloid>(m, "PairPotentialColloid");
    #ifdef ENABLE_CUDA
    export_PotentialPairGPU<azplugins::PairPotentialAshbaughGPU, azplugins::PairPotentialAshbaugh>(m, "PairPotentialAshbaughGPU");
    export_PotentialPairGPU<azplugins::PairPotentialColloidGPU, azplugins::PairPotentialColloid>(m, "PairPotentialColloidGPU");
    #endif // ENABLE_CUDA
    azplugins::detail::export_ashbaugh_params(m);

    /* Wall potentials */
    azplugins::detail::export_wall_potential<azplugins::detail::WallEvaluatorColloid>(m, "WallPotentialColloid");
    azplugins::detail::export_wall_potential<azplugins::detail::WallEvaluatorLJ93>(m, "WallPotentialLJ93");

    return m.ptr();
    }
