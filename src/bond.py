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
    r"""Double-well bond potential.

    `DoubleWell` specifies a double well potential between the two particles in
    the simulation state with:

    .. math::

        U(r)  =  \frac{V_{max}-c/2}{b^4} \left[
            \left( r - a/2 \right)^2 - b^2 \right]^2
            +\frac{c}{2b}(r - a/2) + c/2

    Attributes:
        params (TypeParameter[``bond type``, dict]):
            The parameter of the double-well bonds for each particle type.
            The dictionary has the following keys:

            * ``V_max`` (`float`, **required**) - Potential maximum energy
                barrier between the two minima at ``a/2`` for c=0 (in energy
                units)

            * ``a`` (`float`, **required**) - twice the location of the
                potential maximum, maximum is at ``a/2`` for c=0 ( in distance
                units)

            * ``b`` (`float`, **required**) - tunes the distance between the
                potential minima at ``(a/2 +/- b)`` for c=0 (in distance units)

            * ``c`` (`float`, **required**) -tunes the energy offset between the
                two potential minima values, i.e. it tilts the potential (in
                energy units). The default value of c is zero.

    Examples::

        dw = azplugins.bond.DoubleWell()
        dw.params['A-A'] = dict(V_max=2.0, a=2.5, b=0.5)
        dw.params['A-A'] = dict(V_max=2.0, a=2.5, b=0.2, c=1.0)
    """

    _cpp_class_name = 'PotentialBondDoubleWell'

    def __init__(self):
        super().__init__()
        params = TypeParameter(
            'params',
            'bond_types',
            TypeParameterDict(
                r_0=float, r_1=float, U_1=float, U_tilt=float, len_keys=1
            ),
        )
        self._add_typeparam(params)
