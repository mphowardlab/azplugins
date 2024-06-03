# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2024, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""Bond potentials."""

from hoomd.azplugins import _azplugins
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.md import bond


class _Bond(bond.Bond):
    _ext_module = _azplugins


class DoubleWell(_Bond):
    r"""Double-well bond potential."""

    _cpp_class_name = 'PotentialBondDoubleWell'

    def __init__(self):
        super().__init__()
        params = TypeParameter(
            'params',
            'bond_types',
            TypeParameterDict(V_max=float, a=float, b=float, c=float, len_keys=1),
        )
        self._add_typeparam(params)
