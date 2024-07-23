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
// bond
void export_PotentialBondDoubleWell(pybind11::module&);

// flow
void export_ConstantFlow(pybind11::module&);
void export_ParabolicFlow(pybind11::module&);

// pair
void export_AnisoPotentialPairTwoPatchMorse(pybind11::module&);
void export_PotentialPairColloid(pybind11::module&);
void export_PotentialPairHertz(pybind11::module&);
void export_PotentialPairPerturbedLennardJones(pybind11::module&);

// dpd
void export_PotentialPairDPDThermoGeneralWeight(pybind11::module&);

// wall
void export_PotentialWallLJ93(pybind11::module& m);
void export_PotentialWallColloid(pybind11::module& m);

#ifdef ENABLE_HIP
// bond
void export_PotentialBondDoubleWellGPU(pybind11::module&);

// pair
void export_AnisoPotentialPairTwoPatchMorseGPU(pybind11::module&);
void export_PotentialPairColloidGPU(pybind11::module&);
void export_PotentialPairHertzGPU(pybind11::module&);
void export_PotentialPairPerturbedLennardJonesGPU(pybind11::module&);

// dpd
void export_PotentialPairDPDThermoGeneralWeightGPU(pybind11::module&);

#endif // ENABLE_HIP

    } // namespace detail
    } // namespace azplugins
    } // namespace hoomd

// python module
PYBIND11_MODULE(_azplugins, m)
    {
    using namespace hoomd::azplugins::detail;

    // bond
    export_PotentialBondDoubleWell(m);

    // flow
    export_ConstantFlow(m);
    export_ParabolicFlow(m);

    // pair
    export_AnisoPotentialPairTwoPatchMorse(m);
    export_PotentialPairColloid(m);
    export_PotentialPairHertz(m);
    export_PotentialPairPerturbedLennardJones(m);

    // dpd pair
    export_PotentialPairDPDThermoGeneralWeight(m);

    // wall
    export_PotentialWallLJ93(m);
    export_PotentialWallColloid(m);

#ifdef ENABLE_HIP
    // bond
    export_PotentialBondDoubleWellGPU(m);

    // pair
    export_AnisoPotentialPairTwoPatchMorseGPU(m);
    export_PotentialPairColloidGPU(m);
    export_PotentialPairHertzGPU(m);
    export_PotentialPairPerturbedLennardJonesGPU(m);

    // dpd pair
    export_PotentialPairDPDThermoGeneralWeightGPU(m);

#endif // ENABLE_HIP
    }
