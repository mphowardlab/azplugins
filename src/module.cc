// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#include <pybind11/pybind11.h>

namespace hoomd
    {
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
    } // end namespace detail

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
    } // end namespace hoomd

// forward declaration of all export methods
namespace hoomd
    {
namespace azplugins
    {
namespace detail
    {
void export_PotentialBondDoubleWell(pybind11::module&);

#ifdef ENABLE_HIP
void export_PotentialBondDoubleWellGPU(pybind11::module&);
#endif // ENABLE_HIP
    }  // namespace detail
    }  // namespace azplugins
    }  // namespace hoomd

// python module
PYBIND11_MODULE(_azplugins, m)
    {
    using namespace hoomd::azplugins::detail;

    export_PotentialBondDoubleWell(m);

#ifdef ENABLE_HIP
    export_PotentialBondDoubleWellGPU(m);
#endif // ENABLE_HIP
    }
