from __future__ import print_function
import os

try:
    import hoomd
    print(os.path.dirname(hoomd.__file__), end='')
except ImportError:
    print('HOOMD_ROOT-NOTFOUND', end='')
