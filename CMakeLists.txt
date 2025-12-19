CMAKE_MINIMUM_REQUIRED(VERSION 3.15...3.27 FATAL_ERROR)

# Name the plugin project.
# TODO: Set the project title to the name of your Python package.
project(azplugins LANGUAGES C CXX)

# Find the installed HOOMD.
set(HOOMD_REQUIRED_VERSION "5.0.0")
find_package(HOOMD REQUIRED)
message(STATUS "Found HOOMD ${HOOMD_VERSION}: ${HOOMD_INSTALL_PREFIX}/${PYTHON_SITE_INSTALL_DIR}")
if(NOT HOOMD_VERSION VERSION_GREATER_EQUAL HOOMD_REQUIRED_VERSION)
    message(FATAL_ERROR "HOOMD >=${HOOMD_REQUIRED_VERSION} required (found ${HOOMD_VERSION}).")
endif()

# Force installation to the HOOMD installation location.
set(CMAKE_INSTALL_PREFIX ${HOOMD_INSTALL_PREFIX} CACHE PATH "Installation prefix" FORCE)

# Enable compiler warnings on gcc and clang.
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wconversion -Wno-sign-conversion -Wno-unknown-pragmas -Wno-deprecated-declarations -Wno-unused-result")
endif()

# Add the component's source directory.
add_subdirectory(src)
