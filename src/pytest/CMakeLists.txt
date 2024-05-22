# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2022, Auburn University
# This file is part of the azplugins project, released under the Modified BSD License.

# TODO: List all pytest files in test_files.
set(test_files __init__.py
    )

# Copy tests to the install directory.
install(FILES ${test_files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${COMPONENT_NAME}/pytest
       )

# Copy tests to the build directory for testing prior to installation.
copy_files_to_build("${test_files}" "hoomd-component-${COMPONENT_NAME}-pytest" "*.py")
