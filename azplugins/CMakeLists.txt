# Copyright (c) 2016-2017, Panagiotopoulos Group, Princeton University
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: mphoward / All developers are free to add new source files as needed

set(COMPONENT_NAME azplugins)

###############################
# Configure for internal build if required libraries are not set
if(NOT DEFINED HOOMD_EXTERNAL_BUILD)
    set(HOOMD_EXTERNAL_BUILD OFF)
endif(NOT DEFINED HOOMD_EXTERNAL_BUILD)

if(HOOMD_EXTERNAL_BUILD)
    # the md component is always required
    if (NOT EXISTS ${HOOMD_MD_LIB})
        message(FATAL_ERROR "azplugins requires hoomd installation with MD component.")
    endif (NOT EXISTS ${HOOMD_MD_LIB})

    # the mpcd library is optional
    if(EXISTS ${HOOMD_MPCD_LIB})
        set(BUILD_MPCD ON)
        message(STATUS "mpcd library detected in hoomd installation, building MPCD components.")
    else(EXISTS ${HOOMD_MPCD_LIB})
        set(BUILD_MPCD OFF)
        set(HOOMD_MPCD_LIB "")
        message(STATUS "mpcd library not detected in hoomd installation, skipping MPCD components.")
    endif(EXISTS ${HOOMD_MPCD_LIB})
else(HOOMD_EXTERNAL_BUILD)
    # the md component is always required
    if (NOT BUILD_MD)
        message(FATAL_ERROR "azplugins requires hoomd installation with MD component.")
    endif (NOT BUILD_MD)
    set(HOOMD_MD_LIB _md)

    # mpcd is optional
    if (BUILD_MPCD)
        set(HOOMD_MPCD_LIB _mpcd)
    else (BUILD_MPCD)
        set(HOOMD_MPCD_LIB "")
    endif (BUILD_MPCD)
endif(HOOMD_EXTERNAL_BUILD)
##############################

##############################
# List of all source files
# base / cpu c++ source files
set(_${COMPONENT_NAME}_sources
    module.cc
    BounceBackGeometry.cc
    ImplicitEvaporator.cc
    OrientationRestraintCompute.cc
    ParticleEvaporator.cc
    PositionRestraintCompute.cc
    RDFAnalyzer.cc
    ReversePerturbationFlow.cc
    TypeUpdater.cc
    )

# cuda-enabled c++ source files
if(ENABLE_CUDA)
list(APPEND _${COMPONENT_NAME}_sources
    ImplicitEvaporatorGPU.cc
    OrientationRestraintComputeGPU.cc
    ParticleEvaporatorGPU.cc
    PositionRestraintComputeGPU.cc
    RDFAnalyzerGPU.cc
    ReversePerturbationFlowGPU.cc
    TypeUpdaterGPU.cc
    )
endif(ENABLE_CUDA)

# cuda cu source files
set(_${COMPONENT_NAME}_cu_sources
    AnisoPairPotentials.cu
    BondPotentials.cu
    BounceBackNVEGPU.cu
    ImplicitEvaporatorGPU.cu
    OrientationRestraintComputeGPU.cu
    PairPotentials.cu
    ParticleEvaporatorGPU.cu
    PositionRestraintComputeGPU.cu
    RDFAnalyzerGPU.cu
    ReversePerturbationFlowGPU.cu
    SpecialPairPotentials.cu
    TypeUpdaterGPU.cu
    WallPotentials.cu
    )

# python files
set(py_files
    __init__.py
    analyze.py
    bond.py
    evaporate.py
    flow.py
    integrate.py
    pair.py
    restrain.py
    special_pair.py
    update.py
    wall.py
    )

###
## MPCD component source files
###
if(BUILD_MPCD)
# mpcd python files
list(APPEND py_files
    )
# mpcd c++ files
list(APPEND _${COMPONENT_NAME}_sources
    )
# mpcd c++/cu cuda files
if(ENABLE_CUDA)
list(APPEND _${COMPONENT_NAME}_sources
    )
list(APPEND _${COMPONENT_NAME}_cu_sources
    )
endif(ENABLE_CUDA)
endif(BUILD_MPCD)
##############################

# Need to define NO_IMPORT_ARRAY in every file but module.cc
set_source_files_properties(${_${COMPONENT_NAME}_sources} ${_${COMPONENT_NAME}_cu_sources} PROPERTIES COMPILE_DEFINITIONS NO_IMPORT_ARRAY)

# if MPCD component is enabled, need to define with a macro for source code
if(BUILD_MPCD)
add_definitions(-DENABLE_MPCD)
endif(BUILD_MPCD)

if (ENABLE_CUDA)
CUDA_COMPILE(_CUDA_GENERATED_FILES ${_${COMPONENT_NAME}_cu_sources} OPTIONS ${CUDA_ADDITIONAL_OPTIONS} SHARED)
endif (ENABLE_CUDA)

add_library (_${COMPONENT_NAME} SHARED ${_${COMPONENT_NAME}_sources} ${_CUDA_GENERATED_FILES})

# link the library to its dependencies
target_link_libraries(_${COMPONENT_NAME} ${HOOMD_LIBRARIES} ${HOOMD_MD_LIB} ${HOOMD_MPCD_LIB})

# if we are compiling with MPI support built in, set appropriate
# compiler/linker flags
if (ENABLE_MPI)
   if(MPI_CXX_COMPILE_FLAGS)
       set_target_properties(_${COMPONENT_NAME} PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
   endif(MPI_CXX_COMPILE_FLAGS)
   if(MPI_CXX_LINK_FLAGS)
       set_target_properties(_${COMPONENT_NAME} PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
   endif(MPI_CXX_LINK_FLAGS)
endif(ENABLE_MPI)

##################################
# change the name of the library to be a valid python module
# tweak the properties of the output to make a functional python module
set_target_properties(_${COMPONENT_NAME} PROPERTIES PREFIX "" OUTPUT_NAME "_${COMPONENT_NAME}")

# .dylib is not recognized as a python module by python on Mac OS X
if(APPLE)
    set_target_properties(_${COMPONENT_NAME} PROPERTIES SUFFIX ".so")
endif(APPLE)
fix_cudart_rpath(_${COMPONENT_NAME})

# install the library
install(TARGETS _${COMPONENT_NAME}
        LIBRARY DESTINATION ${PYTHON_MODULE_BASE_DIR}/${COMPONENT_NAME}
        )

################ Python only modules
# copy python modules to the build directory to make it a working python package
MACRO(copy_file file)
    add_custom_command (
        OUTPUT ${file}
        DEPENDS ${file}
        POST_BUILD
        COMMAND    ${CMAKE_COMMAND}
        ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/${file}
        COMMENT    "Copy hoomd/${COMPONENT_NAME}/${file}"
    )
ENDMACRO(copy_file)

install(FILES ${py_files}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}/${COMPONENT_NAME}
       )

foreach(file ${py_files})
    copy_file(${file})
endforeach()

add_custom_target(copy_${COMPONENT_NAME} ALL DEPENDS ${py_files})

if (BUILD_TESTING)
    add_subdirectory(test-py)
    add_subdirectory(test)
endif()
