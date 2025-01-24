// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file RNGIdentifiers.h
 * \brief Identifiers for random123 generator.
 */

#ifndef AZPLUGINS_RNG_IDENTIFIERS_H_
#define AZPLUGINS_RNG_IDENTIFIERS_H_

namespace hoomd
    {
namespace azplugins
    {

namespace detail
    {
struct RNGIdentifier
    {
    // hoomd's identifiers, changed by +/- 1
    static const uint8_t DPDEvaluatorGeneralWeight = 200;
    static const uint8_t TwoStepBrownianFlow = 201;
    static const uint8_t TwoStepLangevinFlow = 202;
    static const uint8_t ParticleEvaporator = 203;
    };

    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd

#endif // AZPLUGINS_RNG_IDENTIFIERS_H_
