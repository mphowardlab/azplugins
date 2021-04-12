# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021, Auburn University
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: mphoward

from __future__ import print_function
import os

try:
    import hoomd
    print(os.path.dirname(hoomd.__file__), end='')
except ImportError:
    print('HOOMD_ROOT-NOTFOUND', end='')
