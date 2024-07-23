# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2024, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""Wall potentials."""

from hoomd.azplugins import _azplugins
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.md.external import wall


class Colloid(wall.WallPotential):
    r"""Colloid (integrated Lennard-Jones) wall potential.

    Args:
        walls (`list` [`hoomd.wall.WallGeometry` ]): A list of wall definitions
            to use for the force.

    Wall force evaluated using the potential obtained by integrating a spherical
    particle of radius *a* with a half plane according to the standard
    Lennard-Jones potential:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V(r) = & \frac{\varepsilon \sigma^6}{7560} \left( \frac{7a-z}{(z-a)^7} +
                 \frac{7a+z}{(z+a)^7} \right)
             - \frac{\varepsilon}{6} \left( \frac{2 a z}{z^2-a^2} +
             \ln\left(\frac{z-a}{z+a}\right) \right) & r < r_{\rm cut} \\
              = & 0 & r \ge r_{\mathrm{cut}}
        \end{eqnarray*}

    Here, :math:`\varepsilon` is the Hamaker constant and :math:`\sigma` is the LJ
    diameter between the particle and the wall. The Hamaker constant for a colloid
    density :math:`\rho_{\rm c}`, a wall of density :math:`\rho_{\rm w}`, and
    Lennard-Jones interaction strength :math:`\varepsilon_{LJ}` is
    :math:`\varepsilon = 4 \pi^2 \varepsilon_{\rm LJ} \rho_{\rm c} \rho_{\rm w}
    \sigma^6`.
    See :py:class:`hoomd.md.external.wall.WallPotential` for generalized
    wall potential implementation, including cutoff and extrapolation schemes.

    .. py:attribute:: params

        The `Colloid` wall potential parameters. The dictionary has the following
        keys:

        * ``epsilon`` (`float`, **required**) -
        energy parameter :math:`\varepsilon` :math:`[\mathrm{energy}]`
        * ``sigma`` (`float`, **required**) -
          particle size :math:`\sigma` :math:`[\mathrm{length}]`
        * ``a`` (`float`, **required**) - Diameter of particle
          :math:`a` :math:`[\mathrm{length}]`
        * ``r_cut`` (`float`, **required**) -
          The cut off distance for the wall potential :math:`[\mathrm{length}]`
        * ``r_extrap`` (`float`, **optional**) -
          The distance to extrapolate the potential :math:`[\mathrm{length}]`,
          defaults to 0.

        Type: `TypeParameter` [``particle_types``, `dict`]

    Example::

        top = hoomd.wall.Plane(origin=(0, 7, 0), normal=(0, -1, 0))
        wall_potential = azplugins.wall.Colloid(walls=[top])
        wall_potential.params['A'] = {"epsilon": 100.0,
                                        "sigma": 1.0,
                                        "a": 1.0,
                                        "r_cut": 4.0}
        simulation.operations.integrator.forces.append(wall_potential)
    """

    _ext_module = _azplugins
    _cpp_class_name = 'WallPotentialColloid'

    def __init__(self, walls):
        super().__init__(walls)

        params = TypeParameter(
            'params',
            'particle_types',
            TypeParameterDict(
                epsilon=float,
                sigma=float,
                a=float,
                r_cut=float,
                r_extrap=0.0,
                len_keys=1,
            ),
        )
        self._add_typeparam(params)


class LJ93(wall.WallPotential):
    r"""Lennard-Jones 9-3 wall potential.

    Args:
        walls (`list` [`hoomd.wall.WallGeometry` ]): A list of wall definitions
            to use for the force.

    Wall force evaluated using the Lennard-Jones 9-3 potential, which is obtained
    by integrating the standard Lennard-Jones potential between a point and a half
    plane:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V(r)  = & \varepsilon \left( \frac{2}{15}\left(\frac{\sigma}{r}\right)^9
                                    - \left(\frac{\sigma}{r}\right)^3 \right)
                                    & r < r_{\rm cut} \\
              = & 0 & r \ge r_{\rm cut}
        \end{eqnarray*}

    Here, :math:`\varepsilon` is the Hamaker constant and :math:`\sigma` is the LJ
    diameter between the particle and the wall. The Hamaker constant for a wall of
    density :math:`\rho_{\rm w}` and with LJ interaction strength
    :math:`\varepsilon_{LJ}`
    is :math:`\varepsilon = (2/3) \pi \varepsilon_{\rm LJ} \rho_{\rm w} \sigma^3`.
    See :py:class:`hoomd.md.external.wall.WallPotential` for generalized wall
    potential implementation, including cutoff and extrapolation schemes.

    .. py:attribute:: params

        The `LJ93` wall potential parameters. The dictionary has the following
        keys:

        * ``epsilon`` (`float`, **required**) -
        energy parameter :math:`\varepsilon` :math:`[\mathrm{energy}]`
        * ``sigma`` (`float`, **required**) -
          particle size :math:`\sigma` :math:`[\mathrm{length}]`
        * ``r_cut`` (`float`, **required**) -
          The cut off distance for the wall potential :math:`[\mathrm{length}]`
        * ``r_extrap`` (`float`, **optional**) -
          The distance to extrapolate the potential :math:`[\mathrm{length}]`,
          defaults to 0.

        Type: `TypeParameter` [``particle_types``, `dict`]

    Example::

        top = hoomd.wall.Plane(origin=(0, 7, 0), normal=(0, -1, 0))
        wall_potential = azplugins.wall.LJ93(walls=[top])
        wall_potential.params['A'] = {"epsilon": 1.0, "sigma": 1.0, "r_cut": 2.0}
        simulation.operations.integrator.forces.append(wall_potential)
    """

    _ext_module = _azplugins
    _cpp_class_name = 'WallPotentialLJ93'

    def __init__(self, walls):
        super().__init__(walls)

        params = TypeParameter(
            'params',
            'particle_types',
            TypeParameterDict(
                epsilon=float, sigma=float, r_cut=float, r_extrap=0.0, len_keys=1
            ),
        )
        self._add_typeparam(params)
