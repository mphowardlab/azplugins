# Copyright (c) 2016-2018, Panagiotopoulos Group, Princeton University
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: mphoward

# try to find hoomd from python
find_package(PythonInterp)
if (NOT HOOMD_ROOT)
    execute_process(COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/CMake/find_hoomd.py
                    OUTPUT_VARIABLE HOOMD_GUESS
                    OUTPUT_STRIP_TRAILING_WHITESPACE)

    find_path(HOOMD_ROOT
              NAMES HOOMDCommonLibsSetup.cmake
              HINTS ${HOOMD_GUESS}
              DOCS "HOOMD package"
              NO_DEFAULT_PATH)
else (NOT HOOMD_ROOT)
    set(HOOMD_ROOT "" CACHE PATH "HOOMD package")
endif (NOT HOOMD_ROOT)

# find include directory
find_path(HOOMD_INCLUDE_DIR
          NAMES HOOMDVersion.h
          HINTS ${HOOMD_ROOT}/include
          NO_DEFAULT_PATH
          )

# find version from includes with python
execute_process(COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/CMake/parse_version.py "${HOOMD_INCLUDE_DIR}/HOOMDVersion.h"
                OUTPUT_VARIABLE HOOMD_VERSION
                OUTPUT_STRIP_TRAILING_WHITESPACE)

# handle package at this point, in case something was not found, before proceeding
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HOOMD
                                  REQUIRED_VARS HOOMD_ROOT HOOMD_INCLUDE_DIR
                                  VERSION_VAR HOOMD_VERSION)

# hoomd core libraries and configuration
set(CMAKE_MODULE_PATH ${HOOMD_ROOT}
                      ${HOOMD_ROOT}/CMake/hoomd
                      ${HOOMD_ROOT}/CMake/thrust
                      ${CMAKE_MODULE_PATH}
                      )
include(hoomd_cache)
include(CMake_build_options)
include(CMake_preprocessor_flags)
include(CMake_install_options)
include(HOOMDMPISetup)
include(HOOMDPythonSetup)
include(HOOMDCUDASetup)
include(HOOMDCFlagsSetup)
include(HOOMDOSSpecificSetup)
include(HOOMDCommonLibsSetup)
include(HOOMDMacros)

# component libraries, with conditional compilation
set(HOOMD_LIBRARY ${HOOMD_ROOT}/_hoomd${PYTHON_MODULE_EXTENSION})
# cgcmm
if (BUILD_CGCMM)
    set(HOOMD_CGCMM_LIBRARY ${HOOMD_ROOT}/cgcmm/_cgcmm${PYTHON_MODULE_EXTENSION})
else (BUILD_CGCMM)
    set(HOOMD_CGCMM_LIBRARY "")
endif (BUILD_CGCMM)
# dem
if (BUILD_DEM)
    set(HOOMD_DEM_LIBRARY ${HOOMD_ROOT}/dem/_dem${PYTHON_MODULE_EXTENSION})
else (BUILD_DEM)
    set(HOOMD_DEM_LIBRARY "")
endif (BUILD_DEM)
# deprecated
if (BUILD_DEPRECATED)
    set(HOOMD_DEPRECATED_LIBRARY ${HOOMD_ROOT}/deprecated/_deprecated${PYTHON_MODULE_EXTENSION})
else (BUILD_DEPRECATED)
    set(HOOMD_DEPRECATED_LIBRARY "")
endif (BUILD_DEPRECATED)
# HPMC
if (BUILD_HPMC)
    set(HOOMD_HPMC_LIBRARY ${HOOMD_ROOT}/hpmc/_hpmc${PYTHON_MODULE_EXTENSION})
else (BUILD_HPMC)
    set(HOOMD_HPMC_LIBRARY "")
endif (BUILD_HPMC)
# jit
if (BUILD_JIT)
    set(HOOMD_JIT_LIBRARY ${HOOMD_ROOT}/jit/_jit${PYTHON_MODULE_EXTENSION})
else (BUILD_JIT)
    set(HOOMD_JIT_LIBRARY "")
endif (BUILD_JIT)
# md
if (BUILD_MD)
    set(HOOMD_MD_LIBRARY ${HOOMD_ROOT}/md/_md${PYTHON_MODULE_EXTENSION})
else (BUILD_MD)
    set(HOOMD_MD_LIBRARY "")
endif (BUILD_MD)
# metal
if (BUILD_METAL)
    set(HOOMD_METAL_LIBRARY ${HOOMD_ROOT}/metal/_metal${PYTHON_MODULE_EXTENSION})
else (BUILD_METAL)
    set(HOOMD_METAL_LIBRARY "")
endif (BUILD_METAL)
# mpcd
if (BUILD_MPCD)
    set(HOOMD_MPCD_LIBRARY ${HOOMD_ROOT}/mpcd/_mpcd${PYTHON_MODULE_EXTENSION})
else (BUILD_MPCD)
    set(HOOMD_MPCD_LIBRARY "")
endif (BUILD_MPCD)

# core libraries and include directories
set(HOOMD_LIBRARIES ${HOOMD_LIBRARY} ${HOOMD_COMMON_LIBS})
set(HOOMD_INCLUDE_DIRS ${HOOMD_INCLUDE_DIR})
