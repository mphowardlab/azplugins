// Copyright (c) 2018-2019, Michael P. Howard
// This file is part of the azplugins project, released under the Modified BSD License.

// Maintainer: mphoward

/*!
 * \file example_test.cc
 * \brief An example of running a simple unit test with upp11.
 *
 * This test demonstrates how to write a unit test for upp11, and also serves to validate
 * the testing macros defined in upp11_config.h.
 *
 * \b Creating new tests:
 *
 * A compiled unit test should be put in a file with a name that ends in _test.cc. Tests are
 * named with underscores after the object or idea they are testing. For example, if the name
 * of the object being tested is MyGreatClass, then the test would be called my_great_class,
 * and the file containing the test would be my_great_class_test.cc. my_great_class must then be
 * added into the TEST_LIST in test/CMakeLists.txt
 *
 * \b Tests with MPI:
 *
 * Most compiled unit tests are easier to write for a single rank. However, it is often necessary
 * to also test MPI-specific features. In this case, it is usually best to write an additional test
 * file that runs in MPI to check these features. The test name can be the same for an MPI test,
 * but the file name should end in _mpi_test.cc. For example, for our previous example, the MPI
 * test file would be my_great_class_mpi_test.cc. Then, this test must be added to the MPI test
 * list along with a designated number of processors to execute on. Usually, this would be 8 so
 * that full decomposition is possible. (You can always utilize fewer processors.) The test
 * should be added in test/CMakeLists.txt with ADD_TO_MPI_TESTS(my_great_test 8) in the section
 * that follows the TEST_LIST.
 */

// Include the testing harness
#include "upp11_config.h"

// All compiled unit tests should call HOOMD_UP_MAIN() exactly once
HOOMD_UP_MAIN()

//! An example unit test
UP_TEST( example_test )
    {
    // UP_ASSERT can be used to validate true / false statements
    UP_ASSERT(true);

    // UP_ASSERT_EQUAL is useful for checking integer values
    UP_ASSERT_EQUAL(2, 2);

    // UP_ASSERT_CLOSE is useful for checking if two floating-point values are close
    UP_ASSERT_CLOSE(2.02,2.0,0.1);
    UP_ASSERT_CLOSE(2.0,2.02,0.1);

    // UP_ASSERT_SMALL should be used for floating point values close to zero.
    UP_ASSERT_SMALL(1.e-5, 0.001);
    }
