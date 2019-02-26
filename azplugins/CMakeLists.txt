# Copyright (c) 2018-2019, Michael P. Howard
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: mphoward / All developers are free to add new source files as needed

set(COMPONENT_NAME azplugins)

###############################
# Configure for internal build if required libraries are not set
if(NOT DEFINED HOOMD_EXTERNAL_BUILD)
    set(HOOMD_EXTERNAL_BUILD OFF)
endif(NOT DEFINED HOOMD_EXTERNAL_BUILD)

# the md component is always required
if (NOT BUILD_MD)
    message(FATAL_ERROR "azplugins requires hoomd installation with MD component.")
endif (NOT BUILD_MD)
# configure the library names if this is an internal build
if(NOT HOOMD_EXTERNAL_BUILD)
    set(HOOMD_MD_LIBRARY _md)
    # mpcd is optional
    if (BUILD_MPCD)
        set(HOOMD_MPCD_LIBRARY _mpcd)
    endif (BUILD_MPCD)
endif(NOT HOOMD_EXTERNAL_BUILD)
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
    DPDPotentials.cu
    ImplicitEvaporatorGPU.cu
    OrientationRestraintComputeGPU.cu
    PairPotentialAshbaugh.cu
    PairPotentialAshbaugh24.cu
    PairPotentialColloid.cu
    PairPotentialLJ96.cu
    PairPotentialLJ124.cu
    PairPotentialShiftedLJ.cu
    PairPotentialSpline.cu
    ParticleEvaporatorGPU.cu
    PositionRestraintComputeGPU.cu
    RDFAnalyzerGPU.cu
    ReversePerturbationFlowGPU.cu
    SpecialPairPotentials.cu
    TwoStepBrownianFlowGPU.cu
    TwoStepLangevinFlowGPU.cu
    TypeUpdaterGPU.cu
    WallPotentials.cu
    )

# python files
set(py_files
    __init__.py
    analyze.py
    bond.py
    dpd.py
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
    mpcd.py
    )
# mpcd c++ files
list(APPEND _${COMPONENT_NAME}_sources
    MPCDReversePerturbationFlow.cc
    )
# mpcd c++/cu cuda files
if(ENABLE_CUDA)
list(APPEND _${COMPONENT_NAME}_sources
    MPCDReversePerturbationFlowGPU.cc
    )
list(APPEND _${COMPONENT_NAME}_cu_sources
    MPCDReversePerturbationFlowGPU.cu
    )
endif(ENABLE_CUDA)
endif(BUILD_MPCD)
##############################

# if MPCD component is enabled, need to define with a macro for source code
if(BUILD_MPCD)
add_definitions(-DENABLE_MPCD)
endif(BUILD_MPCD)

if (ENABLE_CUDA)
CUDA_COMPILE(_CUDA_GENERATED_FILES ${_${COMPONENT_NAME}_cu_sources} OPTIONS ${CUDA_ADDITIONAL_OPTIONS} SHARED)
endif (ENABLE_CUDA)

pybind11_add_module(_${COMPONENT_NAME} SHARED ${_${COMPONENT_NAME}_sources} ${_CUDA_GENERATED_FILES} NO_EXTRAS)
if (NOT HOOMD_EXTERNAL_BUILD)
if (APPLE)
if (BUILD_MPCD)
    set_target_properties(_${COMPONENT_NAME} PROPERTIES INSTALL_RPATH "@loader_path/..;@loader_path/../md;@loader_path/../mpcd;@loader_path")
else (BUILD_MPCD)
    set_target_properties(_${COMPONENT_NAME} PROPERTIES INSTALL_RPATH "@loader_path/..;@loader_path/../md;@loader_path")
endif (BUILD_MPCD)
else (APPLE)
if (BUILD_MPCD)
    set_target_properties(_${COMPONENT_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN/..;$ORIGIN/../md;$ORIGIN/../mpcd;$ORIGIN")
else (BUILD_MPCD)
    set_target_properties(_${COMPONENT_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN/..;$ORIGIN/../md;$ORIGIN")
endif (BUILD_MPCD)
endif (APPLE)
endif (NOT HOOMD_EXTERNAL_BUILD)

# link the library to its dependencies
target_link_libraries(_${COMPONENT_NAME} PRIVATE ${HOOMD_MD_LIBRARY} ${HOOMD_MPCD_LIBRARY} ${HOOMD_LIBRARIES})

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
