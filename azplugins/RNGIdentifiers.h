// Copyright (c) 2018-2019, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file RNGIdentifier.h
 * \brief Identifiers for random123 generator.
 */

#ifndef AZPLUGINS_RNG_IDENTIFIERS_H_
#define AZPLUGINS_RNG_IDENTIFIERS_H_

namespace azplugins
{

struct RNGIdentifier
    {
    // hoomd's identifiers, changed by +/- 1
    static const uint32_t DPDEvaluatorGeneralWeight = 0x4a84f5d1;
    static const uint32_t TwoStepBrownianFlow = 0x431287fe;
    static const uint32_t TwoStepLangevinFlow = 0x89abcdee;
    };

} // end namespace azplugins

#endif // AZPLUGINS_RNG_IDENTIFIERS_H_
