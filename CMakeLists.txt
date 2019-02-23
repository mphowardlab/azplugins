# Copyright (c) 2016-2018, Panagiotopoulos Group, Princeton University
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: mphoward

project(azplugins)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.0 FATAL_ERROR)

# set the cmake policy to enable finding LOCATION in testing, per hoomd usage
if (CMAKE_MAJOR_VERSION VERSION_GREATER 2)
if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
    cmake_policy(SET CMP0042 NEW)
endif(COMMAND cmake_policy)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})
SET(CMAKE_INSTALL_RPATH "$ORIGIN/..:$ORIGIN")

include(FindHOOMD.cmake)
# try to deduce the hoomd version, since FindHOOMD does not support versioning now.
set(find_hoomd_version "
from __future__ import print_function
import sys,os
sys.stdout = open(os.devnull, 'w')
import hoomd
print(hoomd.__version__, file=sys.stderr, end='')")
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "${find_hoomd_version}" ERROR_VARIABLE hoomd_version)
if(hoomd_version MATCHES "^Traceback*")
    message(STATUS "HOOMD version: not deduced, ensure that it is >= 2.2.2.")
else()
    message(STATUS "HOOMD version: " ${hoomd_version})
    if(hoomd_version VERSION_LESS "2.2.2")
    message(FATAL_ERROR "HOOMD version must be >= 2.2.2")
    endif(hoomd_version VERSION_LESS "2.2.2")
endif(hoomd_version MATCHES "^Traceback*")

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
    add_test(NAME azplugins-format
             COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/check_format.sh ${CMAKE_CURRENT_SOURCE_DIR}/azplugins)
endif()

# add subdirectories
add_subdirectory(${PROJECT_NAME})
