# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2024, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""Mock installation of azplugins into a real hoomd installation."""

import os
import shutil
from pathlib import Path

import hoomd

src_dir = Path(__file__).parent.parent / 'src'
install_dir = Path(hoomd.__file__).parent / 'azplugins'

os.makedirs(install_dir, exist_ok=True)
for file in src_dir.glob('*.py'):
    shutil.copy(file, install_dir)
