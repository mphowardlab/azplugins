# TODO: List all pytest files in test_files.
set(test_files
    __init__.py
    test_bond.py
    test_compute.py
    test_external.py
    test_flow.py
    test_pair.py
    test_pair_aniso.py
    test_pair_dpd.py
    test_wall.py
    )

# Copy tests to the install directory.
install(FILES ${test_files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${COMPONENT_NAME}/pytest
       )

# Copy tests to the build directory for testing prior to installation.
copy_files_to_build("${test_files}" "hoomd-component-${COMPONENT_NAME}-pytest" "*.py")
