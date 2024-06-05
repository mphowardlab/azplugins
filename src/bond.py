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

        U(r)  =  U_1\left[\frac{\left((r-r_1)^2-(r_1-r_0)^2\right)^2}
                    {\left(r_1-r_0\right)^4}\right]
                + U_{\rm{tilt}}\left[1+\frac{r-r_1}{r_1-r_0}-
                    \frac{\left((r-r_1)^2-(r_1-r_0)^2\right)^2}
                        {\left(r_1-r_0\right)^4}\right]

    Attributes:
        params (TypeParameter[``bond type``, dict]):
            The parameter of the double-well bonds for each particle type.
            The dictionary has the following keys:

            * ``r_0`` (`float`, **required**) - The location of the
                first potential minimum( in distance units)

            * ``r_1`` (`float`, **required**) - The location of the
                local potential maximum( in distance units)

            * ``U_1`` (`float`, **required**) - Potential maximum energy
                barrier between the two minima at ``r_1`` for ``U_tilt=0`` (in
                energy units)

            * ``U_tilt`` (`float`, **required**) -tunes the energy offset
                between the two potential minima values, i.e. it tilts the
                potential (in energy units). The default value of c is zero.

    Examples::

        dw = azplugins.bond.DoubleWell()
        dw.params['A-A'] = dict(r_0=0.5, r_1=2.5, U_1=5.0, U_tilt=0.0)
        dw.params['A-A'] = dict(r_0=1.0, r_1=2.0, U_1=1.0, U_tilt=0.5)
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
