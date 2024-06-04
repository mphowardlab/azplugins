# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2024, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""Pair potentials."""

from hoomd.azplugins import _azplugins
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.md import pair


class Hertz(pair.Pair):
    r"""Hertz potential.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list.
        r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    :py:class:`Hertz` is the Hertz potential:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V(r)  &= \varepsilon ( 1-\frac{ r }{ r_{\rm{cut}} } )^{5/2} ,
                & r < r_{\rm{cut}} \\
              &= 0,& r \ge r_{\rm{cut}}
        \end{eqnarray*}

    Example::

        nl = hoomd.md.nlist.cell()
        hertz = azplugins.pair.Hertz(r_cut=3.0, nlist=nl)
        hertz.params[('A', 'A')] = dict(epsilon=1.0)
        hertz.r_cut[('A', 'B')] = 3.0

    .. py:attribute:: params

        The Hertz potential parameters. The dictonary has the following key:

        * ``epsilon`` (`float`, **required**) - energy parameter
          :math:`\varepsilon` :math:`[\mathrm{energy}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    .. py:attribute:: mode

        Energy shifting/smoothing mode: ``"none"``, ``"shift"``, or ``"xplor"``.

        Type: `str`

    """

    _ext_module = _azplugins
    _cpp_class_name = 'PotentialPairHertz'
    _accepted_modes = ('none', 'shift', 'xplor')

    def __init__(self, nlist, default_r_cut=None, default_r_on=0, mode='none'):
        super().__init__(nlist, default_r_cut, default_r_on, mode)
        params = TypeParameter(
            'params',
            'particle_types',
            TypeParameterDict(epsilon=float, len_keys=2),
        )
        self._add_typeparam(params)


class PerturbedLennardJones(pair.Pair):
    r"""Perturbed Lennard-Jones potential.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    :py:class:`PerturbedLennardJones` is a Lennard-Jones perturbation potential.
    The potential has a purely repulsive (Weeks-Chandler-Andersen) core, with a
    parameter :math:`attraction_scale_factor` (\lambda) setting the strength of
    the attractive tail. When
    :math:`attraction_scale_factor` is 0, the potential is purely repulsive. When
    :math:`attraction_scale_factor` is 1, the potential is the standard
    Lennard-Jones potential (see :py:class:`hoomd.md.pair.LJ` for details).

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V(r)  = & V_{\mathrm{LJ}}(r, \varepsilon, \sigma) +
        (1-\lambda)\varepsilon & r < 2^{1/6}\sigma \\
              = & \lambda V_{\mathrm{LJ}}(
                r, \varepsilon, \sigma) & 2^{1/6}\sigma \ge
                  r < r_{\mathrm{cut}} \\
              = & 0 & r \ge r_{\mathrm{cut}}
        \end{eqnarray*}


    Example::

        nl = nlist.Cell()
        perturbed_lj = pair.PerturbedLennardJones(default_r_cut=3.0, nlist=nl)
        perturbed_lj.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0,
        attraction_scale_factor=0.5)
        perturbed_lj.r_cut[('A', 'B')] = 3.0

    .. py:attribute:: params

        The Perturbed LJ potential parameters. The dictionary has the following
        keys:

        * ``epsilon`` (`float`, **required**) - energy parameter
          :math:`\varepsilon` :math:`[\mathrm{energy}]`
        * ``sigma`` (`float`, **required**) - particle size :math:`\sigma`
          :math:`[\mathrm{length}]`
        * ``attraction_scale_factor`` (`float`, **required**) - scale factor
          for attraction, between 0 and 1 :math:`attraction_scale_factor`
          :math:`[\mathrm{dimensionless}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    .. py:attribute:: mode

        Energy shifting/smoothing mode: ``"none"``, ``"shift"``, or ``"xplor"``.

        Type: `str`
    """

    _ext_module = _azplugins
    _cpp_class_name = 'PotentialPairPerturbedLennardJones'
    _accepted_modes = ('none', 'shift', 'xplor')

    def __init__(self, nlist, default_r_cut=None, default_r_on=0, mode='none'):
        super().__init__(nlist, default_r_cut, default_r_on, mode)
        params = TypeParameter(
            'params',
            'particle_types',
            TypeParameterDict(
                epsilon=float, sigma=float, attraction_scale_factor=float, len_keys=2
            ),
        )
        self._add_typeparam(params)
