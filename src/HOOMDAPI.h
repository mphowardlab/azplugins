// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file HOOMDAPI.h
 * \brief Defines information about the HOOMD API that is version dependent
 */

#ifndef AZPLUGINS_HOOMD_API_H_
#define AZPLUGINS_HOOMD_API_H_

#include "HOOMDVersion.h"

#if (HOOMD_VERSION_MAJOR >= 2)

#if (HOOMD_VERSION_MINOR >= 8)
// anisotropic pair potential evaluators require an additional shape parameter
#define HOOMD_MD_ANISO_SHAPE_PARAM
#endif // 2.8

#endif // 2.x

#endif // AZPLUGINS_HOOMD_API_H_
