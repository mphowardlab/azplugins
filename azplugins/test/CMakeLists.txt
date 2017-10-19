# Copyright (c) 2016-2017, Panagiotopoulos Group, Princeton University
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: mphoward / All developers are free to add unit tests as needed

###################################
## Setup all of the test executables
#
# Example for foo_bar_test.cc:
#   foo_bar
set(TEST_LIST
    example
    )

if(ENABLE_MPI)
    MACRO(ADD_TO_MPI_TESTS _KEY _VALUE)
    SET("NProc_${_KEY}" "${_VALUE}")
    SET(MPI_TEST_LIST ${MPI_TEST_LIST} ${_KEY})
    ENDMACRO(ADD_TO_MPI_TESTS)

    # Example for foo_bar_mpi_test.cc:
    #   ADD_TO_MPI_TESTS(foo_bar 8)
endif()
###################################

macro(compile_test TEST_EXE TEST_SRC)
    set_source_files_properties(${TEST_SRC} PROPERTIES COMPILE_DEFINITIONS NO_IMPORT_ARRAY)

    # add and link the unit test executable
    add_executable(${TEST_EXE} EXCLUDE_FROM_ALL ${TEST_SRC})
    add_dependencies(test_all ${TEST_EXE})
    target_link_libraries(${TEST_EXE} ${HOOMD_LIBRARIES} ${HOOMD_MD_LIB} _${COMPONENT_NAME})
    fix_cudart_rpath(${TEST_EXE})

    if (ENABLE_MPI)
        # set appropriate compiler/linker flags
        if(MPI_CXX_COMPILE_FLAGS)
            set_target_properties(${TEST_EXE} PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
        endif(MPI_CXX_COMPILE_FLAGS)
        if(MPI_CXX_LINK_FLAGS)
            set_target_properties(${TEST_EXE} PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
        endif(MPI_CXX_LINK_FLAGS)
    endif (ENABLE_MPI)
endmacro(compile_test)

# add non-MPI tests to test list first
foreach (CUR_TEST ${TEST_LIST})
    set(CUR_TEST_SRC ${CUR_TEST}_test.cc)
    set(CUR_TEST_EXE ${CUR_TEST}_test)
    set(CUR_TEST_NAME azplugins-core-${CUR_TEST})

    compile_test(${CUR_TEST_EXE} ${CUR_TEST_SRC})
    get_target_property(CUR_EXE_FULL ${CUR_TEST_EXE} LOCATION)
    if (ENABLE_MPI)
        add_test(${CUR_TEST_NAME} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_POSTFLAGS} ${CUR_EXE_FULL})
    else()
        add_test(${CUR_TEST_NAME} ${CUR_EXE_FULL})
    endif()
endforeach(CUR_TEST)

# add MPI tests
foreach (CUR_TEST ${MPI_TEST_LIST})
    set(CUR_TEST_SRC ${CUR_TEST}_mpi_test.cc)
    set(CUR_TEST_EXE ${CUR_TEST}_mpi_test)
    set(CUR_TEST_NAME azplugins-core-${CUR_TEST}-mpi)

    compile_test(${CUR_TEST_EXE} ${CUR_TEST_SRC})
    get_target_property(CUR_EXE_FULL ${CUR_TEST_EXE} LOCATION)
    add_test(${CUR_TEST_NAME}
             ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG}
             ${NProc_${CUR_TEST}} ${MPIEXEC_POSTFLAGS}
             ${CUR_EXE_FULL})
endforeach(CUR_TEST)
