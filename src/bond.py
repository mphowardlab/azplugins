# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2024, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""Bond potentials."""

from hoomd.azplugins import _azplugins
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.md import bond


class DoubleWell(bond.Bond):
    r"""Double-well bond potential.

    `DoubleWell` specifies a double well potential between the two particles in
    the simulation state with:

    .. math::

        U(r)  &= U_1 \left[1 - \left(\frac{r_1-r}{r_1-r_0}\right)^2 \right]^2 \\
              &+ U_{\rm{tilt}}\left(1 - \frac{r_1-r}{r_1-r_0}
                    -\left[1 - \left(\frac{r_1-r}{r_1-r_0}\right)^2 \right]^2 \right)

    Attributes:
        params (TypeParameter[``bond type``, dict]):
            The parameter of the double-well bonds for each particle type.
            The dictionary has the following keys:

            * ``r_0`` (`float`, **required**) - Location of the first potential
              minimum :math:`r_0` when :math:`U_{\rm tilt} = 0`
              :math:`[\mathrm{length}]`

            * ``r_1`` (`float`, **required**) - Location of the potential local
              maximum :math:`r_1` when :math:`U_{\rm tilt} = 0`
              :math:`[\mathrm{length}]`

            * ``U_1`` (`float`, **required**) - Potential energy
              :math:`U_1 = U(r_1)`
              :math:`[\mathrm{energy}]`

            * ``U_tilt`` (`float`, **required**) - Tunes the energy offset
              :math:`U_{\rm tilt}` between the two potential minima values,
              i.e. it tilts the potential :math:`[\mathrm{energy}]`

    Examples::

        dw = azplugins.bond.DoubleWell()
        dw.params['A-A'] = dict(r_0=0.5, r_1=2.5, U_1=5.0, U_tilt=0.0)
        dw.params['A-A'] = dict(r_0=1.0, r_1=2.0, U_1=1.0, U_tilt=0.5)
    """

    _ext_module = _azplugins
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


class Quartic(bond.Bond):
    r"""Quartic bond potential.

    `Quartic` specifies a quartic potential between the two particles in
    the simulation state with:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        U(r) &= k (r - \Delta - r_0 - b_1)(r - \Delta - r_0 - b_2)(r - \Delta - r_0)^2
                  + U_0 + U_{\rm WCA}(r), & r < r_0 + \Delta \\
             &= U_0, & r \ge r_0 + \Delta
        \end{eqnarray*}

    where :math:`\vec{r}` is the vector pointing from one particle to the other in the
    bond. The potential :math:`U_{\rm WCA}(r)` is given by:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        U_{\rm WCA}(r) &= 4 \varepsilon \left[ \left( \frac{\sigma}{r-\Delta} \\
                                                                            \right)^{12}
                             - \left( \frac{\sigma}{r-\Delta} \right)^{6} \right]
                             + \varepsilon,
                                          & r < 2^{\frac{1}{6}}\sigma + \Delta  \\
                           &= 0,          & r \ge 2^{\frac{1}{6}}\sigma + \Delta
        \end{eqnarray*}


    Attributes:
        params (TypeParameter[``bond type``, dict]):
            The parameters of the quartic bonds for each particle type.
            The dictionary has the following keys:

            * ``k`` (`float`, **required**) - quartic attractive force strength
              :math:`[\mathrm{energy}/\mathrm{length}^4]`.

            * ``r_0`` (`float`, **required**) - Location of the quartic potential
              cutoff :math:`r_0`. Intended to be larger than the WCA cutoff,
              :math:`2^{1/6}\sigma`. When true,
              :math:`U(r_0) = U_{0} + U_{\rm WCA}(r_0)`
              :math:`[\mathrm{length}]`.

            * ``b_1`` (`float`, **required**) - First quartic potential fitting
              parameter :math:`b_1` :math:`[\mathrm{length}]`.

            * ``b_2`` (`float`, **required**) - Second quartic potential fitting
              parameter :math:`b_2` :math:`[\mathrm{length}]`.

            * ``U_0`` (`float`, **required**) - Quartic potential energy barrier height
              :math:`U_0` at :math:`r_0` when :math:`r_0 > 2^{1/6}\sigma`
              :math:`[\mathrm{energy}]`.

            * ``epsilon`` (`float`, **required**) - Repulsive WCA interaction energy
              :math:`\varepsilon` :math:`[\mathrm{energy}]`.

            * ``sigma`` (`float`, **required**) - Repulsive WCA interaction size
              :math:`\sigma` :math:`[\mathrm{length}]`.

            * ``Delta`` (`float`, **optional**) - Shift :math:`\Delta`,
              defaults to zero :math:`[\mathrm{length}]`.

    Examples::

        quartic = azplugins.bond.Quartic()
        quartic.params['A-A'] = dict(k=1434.3, r_0=1.5, b_1=-0.7589, b_2=0.0,
                                    U_0=67.2234, sigma=1, epsilon=1)
        quartic.params['B-B'] = dict(k=1434.3, r_0=1.5, b_1=-0.7589, b_2=0.0,
                                    U_0=67.2234, sigma=1, epsilon=1, delta=0.2)

    The parameters for the examples provided above were derived by Tsige and Stevens in
    a 2004 paper, which used this quartic bond potential with fitted parameters in place
    of FENE bond potentials. This paper can be found here:
    https://pubs.acs.org/doi/10.1021/ma034970t
    """

    _ext_module = _azplugins
    _cpp_class_name = 'PotentialBondQuartic'

    def __init__(self):
        super().__init__()
        params = TypeParameter(
            'params',
            'bond_types',
            TypeParameterDict(
                k=float,
                r_0=float,
                b_1=float,
                b_2=float,
                U_0=float,
                sigma=float,
                epsilon=float,
                delta=0.0,
                len_keys=1,
            ),
        )
        self._add_typeparam(params)
