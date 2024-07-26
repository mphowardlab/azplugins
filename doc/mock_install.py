# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2024, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""Mock installation of azplugins into a real hoomd installation."""

import os
import shutil
import site
import sys
from pathlib import Path

install_dir = Path(site.getsitepackages()[0])

hoomd_dir = Path(sys.argv[1])
for file in hoomd_dir.rglob('*.py'):
    relative_file = file.relative_to(hoomd_dir.parent)
    os.makedirs(install_dir / relative_file.parent, exist_ok=True)
    shutil.copy(file, install_dir / relative_file.parent)

azplugins_dir = Path(sys.argv[2])
for file in azplugins_dir.rglob('*.py'):
    relative_file = Path('hoomd') / 'azplugins' / file.relative_to(azplugins_dir)
    os.makedirs(install_dir / relative_file.parent, exist_ok=True)
    shutil.copy(file, install_dir / relative_file.parent)
