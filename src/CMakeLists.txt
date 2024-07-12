# TODO: Set COMPONENT_NAME to the name of your Python package.
set(COMPONENT_NAME azplugins)

# TODO: List all host C++ source code files in _${COMPONENT_NAME}_sources.
set(_${COMPONENT_NAME}_sources
    ConstantFlow.cc
    GroupVelocityCompute.cc
    module.cc
    ParabolicFlow.cc
    )

# TODO: List all GPU C++ source code files in _${COMPONENT_NAME}_cu_sources.
set(_${COMPONENT_NAME}_cu_sources
    )

# TODO: List all Python modules in python_files.
set(python_files
    __init__.py
    conftest.py
    bond.py
    flow.py
    pair.py
    )

# TODO: Add names of all bond evaluators
set(_bond_evaluators
    DoubleWell
    )

# TODO: Add names of all pair evaluators
set(_pair_evaluators
    Colloid
    Hertz
    PerturbedLennardJones
    )

# TODO: Add names of all dpd evaluators
set(_dpd_evaluators
    GeneralWeight
    )

# TODO: Add names of all anisotropic pair evaluators
set(_aniso_pair_evaluators
    TwoPatchMorse
    )

# process bond potentials
foreach(_evaluator ${_bond_evaluators})
    configure_file(export_PotentialBond.cc.inc
                   export_PotentialBond${_evaluator}.cc
                   @ONLY)
    set(_${COMPONENT_NAME}_sources ${_${COMPONENT_NAME}_sources} export_PotentialBond${_evaluator}.cc)

    if (ENABLE_HIP)
        configure_file(export_PotentialBondGPU.cc.inc
                       export_PotentialBond${_evaluator}GPU.cc
                       @ONLY)
        configure_file(PotentialBondGPUKernel.cu.inc
                       PotentialBond${_evaluator}GPUKernel.cu
                       @ONLY)
        set(_${COMPONENT_NAME}_sources ${_${COMPONENT_NAME}_sources} export_PotentialBond${_evaluator}GPU.cc)
        set(_${COMPONENT_NAME}_cu_sources ${_${COMPONENT_NAME}_cu_sources} PotentialBond${_evaluator}GPUKernel.cu)
    endif()
endforeach()

# process pair potentials
foreach(_evaluator ${_pair_evaluators})
    configure_file(export_PotentialPair.cc.inc
                   export_PotentialPair${_evaluator}.cc
                   @ONLY)
    set(_${COMPONENT_NAME}_sources ${_${COMPONENT_NAME}_sources} export_PotentialPair${_evaluator}.cc)

    if (ENABLE_HIP)
        configure_file(export_PotentialPairGPU.cc.inc
                       export_PotentialPair${_evaluator}GPU.cc
                       @ONLY)
        configure_file(PotentialPairGPUKernel.cu.inc
                       PotentialPair${_evaluator}GPUKernel.cu
                       @ONLY)
        set(_${COMPONENT_NAME}_sources ${_${COMPONENT_NAME}_sources} export_PotentialPair${_evaluator}GPU.cc)
        set(_${COMPONENT_NAME}_cu_sources ${_${COMPONENT_NAME}_cu_sources} PotentialPair${_evaluator}GPUKernel.cu)
    endif()
endforeach()

# process DPD potentials
foreach(_evaluator ${_dpd_evaluators})
    configure_file(export_PotentialPairDPDThermo.cc.inc
                   export_PotentialPairDPDThermo${_evaluator}.cc
                   @ONLY)
    set(_${COMPONENT_NAME}_sources ${_${COMPONENT_NAME}_sources}  export_PotentialPairDPDThermo${_evaluator}.cc)

    if (ENABLE_HIP)
        configure_file(export_PotentialPairDPDThermoGPU.cc.inc
                       export_PotentialPairDPDThermo${_evaluator}GPU.cc
                       @ONLY)
        configure_file(PotentialPairDPDThermoGPUKernel.cu.inc
                       PotentialPairDPDThermo${_evaluator}GPUKernel.cu
                       @ONLY)
        set(_${COMPONENT_NAME}_sources ${_${COMPONENT_NAME}_sources} export_PotentialPairDPDThermo${_evaluator}GPU.cc)
        set(_${COMPONENT_NAME}_cu_sources ${_${COMPONENT_NAME}_cu_sources} PotentialPairDPDThermo${_evaluator}GPUKernel.cu)
    endif()
endforeach()

# process anisotropic pair potentials
foreach(_evaluator ${_aniso_pair_evaluators})
    configure_file(export_AnisoPotentialPair.cc.inc
                   export_AnisoPotentialPair${_evaluator}.cc
                   @ONLY)
    set(_${COMPONENT_NAME}_sources ${_${COMPONENT_NAME}_sources} export_AnisoPotentialPair${_evaluator}.cc)

    if (ENABLE_HIP)
        configure_file(export_AnisoPotentialPairGPU.cc.inc
                       export_AnisoPotentialPair${_evaluator}GPU.cc
                       @ONLY)
        configure_file(AnisoPotentialPairGPUKernel.cu.inc
                       AnisoPotentialPair${_evaluator}GPUKernel.cu
                       @ONLY)
        set(_${COMPONENT_NAME}_sources ${_${COMPONENT_NAME}_sources} export_AnisoPotentialPair${_evaluator}GPU.cc)
        set(_${COMPONENT_NAME}_cu_sources ${_${COMPONENT_NAME}_cu_sources} AnisoPotentialPair${_evaluator}GPUKernel.cu)
    endif()
endforeach()

if (ENABLE_HIP)
set(_cuda_sources ${_${COMPONENT_NAME}_cu_sources})
endif (ENABLE_HIP)

hoomd_add_module(_${COMPONENT_NAME} SHARED ${_${COMPONENT_NAME}_sources} ${_cuda_sources} NO_EXTRAS)
# Alias into the HOOMD namespace so that external and symlinked components both work.
add_library(HOOMD::_${COMPONENT_NAME} ALIAS _${COMPONENT_NAME})

if (APPLE)
set_target_properties(_${COMPONENT_NAME} PROPERTIES INSTALL_RPATH "@loader_path/..;@loader_path")
else()
set_target_properties(_${COMPONENT_NAME} PROPERTIES INSTALL_RPATH "\$ORIGIN/..;\$ORIGIN")
endif()

# Link the library to its dependencies. Add or remove HOOMD extension modules (and/or external C++
# libraries) as needed.
target_link_libraries(_${COMPONENT_NAME}
                      PUBLIC HOOMD::_hoomd
                      PUBLIC HOOMD::_md
                      )

# Explicitly include this directory during builds. This seems to be needed for
# the autogenerated source files that are in the build directory.
target_include_directories(_${COMPONENT_NAME} PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

# Install the library.
install(TARGETS _${COMPONENT_NAME}
        LIBRARY DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${COMPONENT_NAME}
        )

# Install the Python package.
install(FILES ${python_files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${COMPONENT_NAME}
       )

# Copy the Python package to the build directory.
copy_files_to_build("${python_files}" "hoomd-component-${COMPONENT_NAME}" "*.py")

# Python tests.
add_subdirectory(pytest)
