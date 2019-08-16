# Copyright (c) 2018-2019, Michael P. Howard
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: mphoward

project(azplugins)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.0 FATAL_ERROR)
cmake_policy(SET CMP0074 OLD)

# set the cmake policy to enable finding LOCATION in testing, per hoomd usage
if (CMAKE_MAJOR_VERSION VERSION_GREATER 2)
if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
    cmake_policy(SET CMP0042 NEW)
endif(COMMAND cmake_policy)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake ${CMAKE_MODULE_PATH})
find_package(HOOMD 2.6.0 REQUIRED)
include_directories(${HOOMD_INCLUDE_DIRS})
set(HOOMD_EXTERNAL_BUILD ON)

# plugins must be built as shared libraries
if (ENABLE_STATIC)
    message(SEND_ERROR "Plugins cannot be built against a statically compiled hoomd")
endif (ENABLE_STATIC)
set(BUILD_SHARED_LIBS on)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${HOOMD_ROOT} CACHE PATH "Installation prefix" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
set(PYTHON_MODULE_BASE_DIR ${CMAKE_INSTALL_PREFIX})

# setup unit tests
enable_testing()
option(BUILD_TESTING "Build unit tests" ON)
if (BUILD_TESTING)
    add_custom_target(test_all ALL)
    add_test(NAME azplugins-format
             COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/check_format.sh ${CMAKE_CURRENT_SOURCE_DIR}/azplugins)
endif()

# add subdirectories
add_subdirectory(${PROJECT_NAME})
