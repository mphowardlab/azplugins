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


class TwoPatchMorse(pair.aniso.AnisotropicPair):
    r"""Two-patch Morse potential.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting mode.

    :py:class:`TwoPatchMorse` is a Morse potential which is modulated by an
    orientation-dependent function.

    .. math::

        V_{\rm M2P}(\vec{r}_{ij}, \hat{n}_i, \hat{n}_j) = V_{\rm M}(|\vec{r}_{ij}|)
        \Omega(\hat{r}_{ij} \cdot \hat{n}_i) \Omega(\hat{r}_{ij} \cdot \hat{n}_j)

    where :math:`V_{\rm M}` is the potential that depends on distance

    .. math::

        V_{\rm M}(r) = M_d \left( \left[ 1 - \exp\left(
              -\frac{r-r_{\rm eq}}{M_r}\right) \right]^2 - 1 \right)

    and :math:`\Omega(\gamma)` depends on the orientations

    .. math::
        \Omega(\gamma) = \frac{1}{1+\exp[-\omega (\gamma^2 - \alpha)]}

    The potential can be smoothed to zero force (making it purely attractive)
    when :math:`r < r_{\rm eq}` by making :math:`V_{\rm M}(r < r_{\rm eq}) = -M_d`
    when the option  ``repulsion`` is ``False``.

    Here, :math:`vec{r}_{ij}` is the displacement vector between particles
    :math:`i` and :math:`j`, :math:`|\vec{r}_{ij}|` is the magnitude of
    that displacement, and :math:`\hat{n}` is the normalized
    orientation vector of the particle. The parameters :math:`M_d`,
    :math:`M_r`, and :math:`r_{\rm eq}` control the depth, width, and
    position of the potential well. The parameters :math:`\alpha` and
    :math:`\omega` control the width and steepness of the orientation dependence.

    Example::

        nl = hoomd.md.nlist.cell()
        m2p = azplugins.pair.TwoPatchMorse(r_cut=1.6, nlist=nl)
        m2p.pair_coeff.set('A', 'A', M_d=1.8347, M_r=0.0302, r_eq=1.0043,
        omega=20, alpha=0.50, repulsion=True)

    .. py:attribute:: params

        The Two Patch Morse potential parameters. The dictionary has the following
        keys:

        * ``M_d`` (`float`, **required**) - controls the depth of the
         potential well
          :math:`M_d` :math:`[\mathrm{energy}]`
        * ``M_r`` (`float`, **required**) - controls the width of the
         potential well :math:`M_r` :math:`[\mathrm{length}]`
        * ``r_eq`` (`float`, **required**) - controls the position of
        the potential well :math:`r_eq` :math:`[\mathrm{length}]`
        * ``omega`` (`float`, **required**) - controls the steepness
        of the orientation depndence :math:`\omega`
        :math:`[\mathrm{dimensionless}]`
        * ``alpha`` (`float`, **required**) - controls the width
        of the orientation depndence :math:`\omega`
        :math:`[\mathrm{dimensionless}]`
        * ``repulsion`` (`bool`, **required**) - repulsion

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    """

    _ext_module = _azplugins
    _cpp_class_name = 'AnisoPotentialPairTwoPatchMorse'

    def __init__(self, nlist, default_r_cut=None, mode='none'):
        super().__init__(nlist, default_r_cut, mode)
        params = TypeParameter(
            'params',
            'particle_types',
            TypeParameterDict(
                M_d=float,
                M_r=float,
                r_eq=float,
                omega=float,
                alpha=float,
                repulsion=bool,
                len_keys=2,
            ),
        )
        self._add_typeparam(params)
