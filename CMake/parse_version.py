# Copyright (c) 2018-2019, Michael P. Howard
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: mphoward

from __future__ import print_function
import sys
import re

pattern = re.compile('#define HOOMD_VERSION "([0-9\.]+)"')

version="HOOMD_VERSION-NOTFOUND"
try:
    with open(sys.argv[1]) as f:
        for line in f:
            match = pattern.match(line)
            if match:
                version = match.group(1)
                break
except:
    pass

print(version, end="")
