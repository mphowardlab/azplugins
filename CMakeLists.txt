# Copyright (c) 2016-2017, Panagiotopoulos Group, Princeton University
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: mphoward

project(azplugins)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6.2 FATAL_ERROR)

# set the cmake policy to enable finding LOCATION in testing, per hoomd usage
if (CMAKE_MAJOR_VERSION VERSION_GREATER 2)
if(COMMAND cmake_policy)
    cmake_policy(SET CMP0026 OLD)
endif(COMMAND cmake_policy)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})
SET(CMAKE_INSTALL_RPATH "$ORIGIN/..:$ORIGIN")

include(FindHOOMD.cmake)

# plugins must be built as shared libraries
if (ENABLE_STATIC)
    message(SEND_ERROR "Plugins cannot be built against a statically compiled hoomd")
endif (ENABLE_STATIC)

set(BUILD_SHARED_LIBS on)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${HOOMD_ROOT} CACHE PATH "Installation prefix" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

set(PYTHON_MODULE_BASE_DIR ${CMAKE_INSTALL_PREFIX})
message(STATUS "Install plugin to: " ${PYTHON_MODULE_BASE_DIR})

# setup unit tests
enable_testing()
option(BUILD_TESTING "Build unit tests" ON)
if (BUILD_TESTING)
    add_custom_target(test_all ALL)
endif()

# add subdirectories
add_subdirectory(${PROJECT_NAME})
