# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2025, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""Wall potentials."""

from hoomd.azplugins import _azplugins
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.md.external import wall


class Colloid(wall.WallPotential):
    r"""Colloid wall potential.

    Args:
        walls (`list` [`hoomd.wall.WallGeometry`]) : A list of wall definitions.

    `Colloid` is Lennard-Jones colloid wall potential, which is derived from integrating
    the standard Lennard-Jones potential between a spherical particle of radius
    :math:`a` and a half plane:

    .. math::

        U(r) = \frac{A \sigma^6}{7560} \left[\frac{7a-r}{(r-a)^7} +
            \frac{7a+r}{(r+a)^7} \right] - \frac{A}{6} \left[\frac{2ar}{r^2-a^2}
            + \ln\left(\frac{z-a}{z+a}\right) \right]

    where:

    * :math:`\sigma` - diameter of Lennard-Jones particles in the sphere and the wall
    * :math:`A` - Hamaker constant, related to the Lennard-Jones parameters as
        :math:`A = 4 \pi^2 \varepsilon \sigma^6 \rho_{\rm w} \rho_{\rm c}`, where
        :math:`\rho_{\rm w}` is the number density of particles in the wall,
        :math:`\rho_{\rm c}` is the number density of particles in the sphere, and
        :math:`\varepsilon` is the Lennard-Jones energy parameter.

    Example::

        walls = [hoomd.wall.Plane(origin=(0, 0, 0), normal=(0, 1, 0))]
        colloid = azplugins.wall.Colloid(walls=walls)
        colloid.params["A"] = dict(
            A=4 * numpy.pi ** 2,  # rho = 1, epsilon = 1, sigma = 1
            a=1.0
            sigma=1.0,
            r_cut=3.0,
        )

    .. py:attribute:: params

        The `Colloid` potential parameters. The dictionary has the following
        keys:

        * ``A`` (`float`, **required**) - Hamaker constant
          :math:`A` :math:`[\mathrm{energy}]`
        * ``a`` (`float`, **required**) - Radius of particle
          :math:`a` :math:`[\mathrm{length}]`
        * ``sigma`` (`float`, **required**) - Lennard-Jones particle size
          :math:`\sigma` :math:`[\mathrm{length}]`
        * ``r_cut`` (`float`, **required**) - The cut off distance for
          the wall potential :math:`[\mathrm{length}]`
        * ``r_extrap`` (`float`, **optional**) - The distance to
          extrapolate the potential, defaults to 0.
          :math:`[\mathrm{length}]`.

        Type: :class:`~hoomd.data.typeparam.TypeParameter` [``particle_type``], `dict`]
    """

    _ext_module = _azplugins
    _cpp_class_name = "WallsPotentialColloid"

    def __init__(self, walls):
        super().__init__(walls)
        params = TypeParameter(
            "params",
            "particle_types",
            TypeParameterDict(
                A=float, a=float, sigma=float, r_cut=float, r_extrap=0.0, len_keys=1
            ),
        )
        self._add_typeparam(params)


class LJ93(wall.WallPotential):
    r"""Lennard-Jones 9-3 wall potential.

    Args:
        walls (`list` [`hoomd.wall.WallGeometry`]) : A list of wall definitions.

    `LJ93` is Lennard-Jones 9-3 wall potential, which is derived from integrating
    the standard Lennard-Jones potential between a point particle and a half plane:

    .. math::

        U(r) = A \left[\frac{2}{15} \left(\frac{\sigma}{r} \right)^9
            - \left(\frac{\sigma}{r}\right)^3 \right]

    where:

    * :math:`\sigma` - diameter of Lennard-Jones particles in the wall
    * :math:`A` - Hamaker constant, related to the Lennard-Jones parameters as
        :math:`A = \frac{2}{3} \pi \varepsilon \sigma^3 \rho`, where :math:`\rho`
        is the number density of particles in the wall and
        :math:`\varepsilon` is the Lennard-Jones energy parameter.

    Example::

        walls = [hoomd.wall.Plane(origin=(0, 0, 0), normal=(0, 1, 0))]
        lj93 = azplugins.wall.LJ93(walls=walls)
        lj93.params["A"] = dict(
            sigma=1.0,
            A=2 * numpy.pi / 3,  # rho = 1, epsilon = 1, sigma = 1
            r_cut=3.0,
        )

    .. py:attribute:: params

        The `LJ93` potential parameters. The dictionary has the following
        keys:

        * ``A`` (`float`, **required**) - Hamaker constant
          :math:`A` :math:`[\mathrm{energy}]`
        * ``sigma`` (`float`, **required**) - Lennard-Jones particle size
          :math:`\sigma` :math:`[\mathrm{length}]`
        * ``r_cut`` (`float`, **required**) - The cut off distance for
          the wall potential :math:`[\mathrm{length}]`
        * ``r_extrap`` (`float`, **optional**) - The distance to
          extrapolate the potential, defaults to 0.
          :math:`[\mathrm{length}]`.

        Type: :class:`~hoomd.data.typeparam.TypeParameter` [``particle_type``], `dict`]
    """

    _ext_module = _azplugins
    _cpp_class_name = "WallsPotentialLJ93"

    def __init__(self, walls):
        super().__init__(walls)
        params = TypeParameter(
            "params",
            "particle_types",
            TypeParameterDict(
                A=float, sigma=float, r_cut=float, r_extrap=0.0, len_keys=1
            ),
        )
        self._add_typeparam(params)
