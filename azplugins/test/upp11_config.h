// Copyright (c) 2018-2020, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file upp11_config.h
 * \brief Defines macros for setting up and running unit tests in the upp11 unit testing framework.
 */

#ifndef AZPLUGINS_TEST_UPP11_CONFIG_H_
#define AZPLUGINS_TEST_UPP11_CONFIG_H_

#include "hoomd/HOOMDMPI.h"
#include "hoomd/extern/upp11/upp11.h"

#include <cmath>
#include <string>

//! Macro to test if the difference between two floating-point values is within a tolerance
/*!
 * \param a First value to test
 * \param b Second value to test
 * \param eps Difference allowed between the two
 *
 * This assertion will pass if the difference between \a a and \a b is within a tolerance,
 * defined by \a eps times the smaller of the magnitude of \a a and \a b.
 *
 * \warning This assertion should not be used when one of the values should be zero. In that
 *          case, use UP_ASSERT_SMALL instead.
 */
#define UP_ASSERT_CLOSE(a,b,eps) \
upp11::TestCollection::getInstance().checkpoint(LOCATION, "UP_ASSERT_CLOSE"), \
upp11::TestAssert(LOCATION).assertTrue(std::abs((a)-(b)) <= (eps) * std::min(std::abs(a), std::abs(b)), \
                                       #a " (" + std::to_string(a) + ") close to " #b " (" + std::to_string(b) + ")")

//! Macro to test if a floating-point value is close to zero
/*!
 * \param a Value to test
 * \param eps Difference allowed from zero
 *
 * This assertion will pass if the absolute value of \a a is less than \a eps.
 */
#define UP_ASSERT_SMALL(a,eps) \
upp11::TestCollection::getInstance().checkpoint(LOCATION, "UP_ASSERT_SMALL"), \
upp11::TestAssert(LOCATION).assertTrue(std::abs(a) < (eps), #a " (" + std::to_string(a) + ") close to 0")

#ifdef ENABLE_MPI
#define HOOMD_UP_MAIN() \
int main(int argc, char **argv) \
    { \
    MPI_Init(&argc, &argv); \
    int val = upp11::TestMain().main(argc, argv); \
    MPI_Finalize(); \
    return val; \
    }
#else
#define HOOMD_UP_MAIN() \
int main(int argc, char **argv) { \
    return upp11::TestMain().main(argc, argv); \
}
#endif

#endif // AZPLUGINS_TEST_UPP11_CONFIG_H_
