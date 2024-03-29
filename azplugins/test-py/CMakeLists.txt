# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2022, Auburn University
# This file is part of the azplugins project, released under the Modified BSD License.

set(TEST_LIST_CPU
    test_analyze_group_velocity
    test_analyze_rdf
    test_bond_double_well
    test_bond_fene
    test_bond_fene24
    test_dpd_general
    test_evaporate_implicit
    test_evaporate_particles
    test_flow_brownian
    test_flow_constant
    test_flow_flow_profiler
    test_flow_langevin
    test_flow_parabolic
    test_flow_quiescent
    test_integrate_slit
    test_pair_ashbaugh
    test_pair_ashbaugh24
    test_pair_lj124
    test_pair_hertz
    test_pair_lj96
    test_pair_colloid
    test_pair_slj
    test_pair_spline
    test_pair_two_patch_morse
    test_restrain_cylinder
    test_restrain_orientation
    test_restrain_plane
    test_restrain_position
    test_restrain_sphere
    test_flow_reverse_perturbation
    test_special_pair_lj96
    test_update_type
    test_variant_sphere_area
    test_wall_colloid
    test_wall_lj93
    )

set(TEST_LIST_GPU
    test_analyze_rdf
    test_bond_double_well
    test_bond_fene
    test_bond_fene24
    test_dpd_general
    test_evaporate_implicit
    test_evaporate_particles
    test_flow_brownian
    test_flow_constant
    test_flow_flow_profiler
    test_flow_langevin
    test_integrate_slit
    test_pair_ashbaugh
    test_pair_ashbaugh24
    test_pair_lj124
    test_pair_hertz
    test_pair_lj96
    test_pair_colloid
    test_pair_slj
    test_pair_spline
    test_pair_two_patch_morse
    test_restrain_cylinder
    test_restrain_orientation
    test_restrain_plane
    test_restrain_position
    test_restrain_sphere
    test_flow_reverse_perturbation
    test_special_pair_lj96
    test_update_type
    test_variant_sphere_area
    test_wall_colloid
    test_wall_lj93
    )

set(MPCD_ONLY
    test_mpcd_reverse_perturbation
    test_mpcd_sinusoidal_channel
    test_mpcd_sinusoidal_expansion_constriction
    test_mpcd_velocity_compute
    )

set(MPI_ONLY
    )

set(EXCLUDE_FROM_MPI
    test_analyze_rdf
    test_restrain_orientation
    test_restrain_position
    test_flow_reverse_perturbation
    test_mpcd_reverse_perturbation
    )

# these macros are duplicates from other places in the hoomd code
# eventually hoomd will hopefully adopt a single macro definition
macro(add_script_test_cpu_mpi script)
    # execute on two processors
    SET(nproc 2)
    if (ENABLE_MPI)
        if(NOT "${EXCLUDE_FROM_MPI}" MATCHES ${script})
            add_test(NAME azplugins-${script}-mpi-cpu
                COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${nproc}
                ${MPIEXEC_POSTFLAGS} ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${script}.py "--mode=cpu")
            set_tests_properties(azplugins-${script}-mpi-cpu PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}")
        endif()
    endif(ENABLE_MPI)
endmacro()

macro(add_script_test_gpu_mpi script)
    # execute on two processors
    SET(nproc 2)
    if (ENABLE_MPI)
        if(NOT "${EXCLUDE_FROM_MPI}" MATCHES ${script})
            add_test(NAME azplugins-${script}-mpi-gpu
                COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${nproc}
                ${MPIEXEC_POSTFLAGS} ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${script}.py "--mode=gpu" "--gpu_error_checking")
            set_tests_properties(azplugins-${script}-mpi-gpu PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}")
        endif()
    endif(ENABLE_MPI)
endmacro()


macro(add_script_test_cpu script)
    add_test(azplugins-${script}-cpu ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${script}.py "--mode=cpu")
    set_tests_properties(azplugins-${script}-cpu PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}")

    add_script_test_cpu_mpi(${script})
endmacro()

macro(add_script_test_gpu script)
    add_test(azplugins-${script}-gpu ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${script}.py "--mode=gpu" "--gpu_error_checking")
    set_tests_properties(azplugins-${script}-gpu PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}")

    add_script_test_gpu_mpi(${script})
endmacro()

foreach (CUR_TEST ${TEST_LIST_CPU})
    add_script_test_cpu(${CUR_TEST})
endforeach (CUR_TEST)
if (BUILD_MPCD)
foreach (CUR_TEST ${MPCD_ONLY})
    add_script_test_cpu(${CUR_TEST})
endforeach (CUR_TEST)
endif (BUILD_MPCD)
if (ENABLE_MPI)
foreach (CUR_TEST ${MPI_ONLY})
    add_script_test_cpu_mpi(${CUR_TEST})
endforeach (CUR_TEST)
endif (ENABLE_MPI)

if (ENABLE_CUDA)
foreach (CUR_TEST ${TEST_LIST_GPU})
    add_script_test_gpu(${CUR_TEST})
endforeach (CUR_TEST)
if (BUILD_MPCD)
foreach (CUR_TEST ${MPCD_ONLY})
    add_script_test_gpu(${CUR_TEST})
endforeach (CUR_TEST)
endif (BUILD_MPCD)
if (ENABLE_MPI)
foreach (CUR_TEST ${MPI_ONLY})
    add_script_test_gpu_mpi(${CUR_TEST})
endforeach (CUR_TEST)
endif (ENABLE_MPI)
endif (ENABLE_CUDA)
