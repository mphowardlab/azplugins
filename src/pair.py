# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2024, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""Pair potentials."""

from hoomd.azplugins import _azplugins
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.md import pair


class Colloid(pair.Pair):
    r"""Colloid pair potential.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list.
        r_cut (float):cDefault cutoff radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `Colloid` is the effective Lennard-Jones potential obtained by integrating
    the Lennard-Jones potential over zero, one, or two spheres. A discussion of
    the application of these potentials to colloidal suspensions can be found in
    `Grest et al.`_

    The pair potential has three different coupling styles between particles:

    - When the radii of both particles are zero, the potential is the usual
      Lennard-Jones coupling:

      .. math::

          U(r) = \frac{A}{36} \left[\left(\frac{\sigma}{r}\right)^{12}
              - \left(\frac{\sigma}{r}\right)^6 \right]

      The Hamaker constant is related to the Lennard-Jones parameters as
      :math:`A = 144 \varepsilon` (to give the usual prefactor
      :math:`4\varepsilon`).

    - When the radius of one particle is zero and the other radius :math:`a` is
      nonzero, the potential is the interaction between a pointlike particle and
      a sphere comprised of Lennard-Jones particles:

      .. math::

          U(r) = \frac{2 a^3 \sigma^3 A}{9(a^2-r^2)^3} \left[
              1 - \frac{(5 a^6 + 45 a^4 r^2 + 63 a^2 r^4 + 15 r^6) \sigma^6}
              {15 (a-r)^6 (a+r)^6}
              \right]

      The Hamaker constant is related to the Lennard-Jones parameters as
      :math:`A = 24 \pi \varepsilon \sigma^3 \rho`, where :math:`\rho` is the
      number density of particles in the sphere.

    - When the radii of both particles are non-zero, the potential is the
      interaction between two spheres comprised of Lennard-Jones particles,
      given by Eqs. (16) and (17) of `Everaers and Ejtehadi`_. The Hamaker
      constant is related to the Lennard-Jones parameters as
      :math:`A = 4 \pi^2 \varepsilon \sigma^6 \rho_1 \rho_2`, where
      :math:`\rho_1` and :math:`\rho_2` are the number densities of particles
      in each sphere.

    Example::

        # Explicit Solvent Model from https://doi.org/10.1063/1.5043401
        nl = nlist.Cell()
        colloid = pair.Colloid(default_r_cut=3.0, nlist=nl)
        # standard Lennard-Jones for solvent-solvent
        colloid.params[('S', 'S')] = dict(A=144.0, a_1=0, a_2=0 sigma=1.0)
        # solvent-colloid
        colloid.params[('S', 'C')] = dict(A=144.0, a_1=0, a_2=5.0 sigma=1.0)
        colloid.r_cut[('S', 'C')] = 9.0
        # colloid-colloid
        colloid.params[('C','C')] = dict(A=40.0, a_1=5.0, a_2=5.0 sigma=1.0)
        colloid.r_cut[('C', 'C')] = 10.581

    .. py:attribute:: params

        The `Colloid` potential parameters. The dictionary has the following
        keys:

        * ``A`` (`float`, **required**) - Hamaker constant :math:`A`
          :math:`[\mathrm{energy}]`
        * ``a_1`` (`float`, **required**) - Radius of first particle :math:`a_1`
          :math:`[\mathrm{length}]`
        * ``a_2`` (`float`, **required**) - Radius of second particle
          :math:`a_2` :math:`[\mathrm{length}]`
        * ``sigma`` (`float`, **required**) - Lennard-Jones particle size
          :math:`\sigma` :math:`[\mathrm{length}]`

        Type: :class:`~hoomd.data.typeparam.TypeParameter` [`tuple`
        [``particle_type``, ``particle_type``], `dict`]

    .. py:attribute:: mode

        Energy shifting/smoothing mode: ``"none"``, ``"shift"``, or ``"xplor"``.

        Type: `str`

    .. _Everaers and Ejtehadi: https://doi.org/10.1103/PhysRevE.67.041710
    .. _Grest et al.: https://doi.org/10.1063/1.3578181

    """

    _ext_module = _azplugins
    _cpp_class_name = 'PotentialPairColloid'
    _accepted_modes = ('none', 'shift', 'xplor')

    def __init__(self, nlist, default_r_cut=None, default_r_on=0, mode='none'):
        super().__init__(nlist, default_r_cut, default_r_on, mode)
        params = TypeParameter(
            'params',
            'particle_types',
            # TypeParameterDict needs updated still
            TypeParameterDict(A=float, a_1=float, a_2=float, sigma=float, len_keys=2),
        )
        self._add_typeparam(params)


class Hertz(pair.Pair):
    r"""Hertz potential.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list.
        r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    The `Hertz` potential is:

    .. math::

        U(r) = \varepsilon \left( 1-\frac{ r }{ r_{\rm{cut}} } \right)^{5/2} ,
                \quad r < r_{\rm{cut}}

    Example::

        nl = hoomd.md.nlist.cell()
        hertz = azplugins.pair.Hertz(r_cut=3.0, nlist=nl)
        hertz.params[('A', 'A')] = dict(epsilon=1.0)
        hertz.r_cut[('A', 'B')] = 3.0

    .. py:attribute:: params

        The `Hertz` potential parameters. The dictonary has the following key:

        * ``epsilon`` (`float`, **required**) - energy parameter
          :math:`\varepsilon` :math:`[\mathrm{energy}]`

        Type: :class:`~hoomd.data.typeparam.TypeParameter` [`tuple`
        [``particle_type``, ``particle_type``], `dict`]

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

    `PerturbedLennardJones` is a Lennard-Jones perturbation potential. The
    potential has a purely repulsive core, and the parameter
    ``attraction_scale_factor`` (\lambda) sets the strength of the attractive
    tail:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        U(r)  &= U_{\mathrm{LJ}}(r) +
                (1-\lambda)\varepsilon, & r \le 2^{1/6}\sigma \\
              &= \lambda U_{\mathrm{LJ}}(r), & r > 2^{1/6}\sigma
        \end{eqnarray*}

    where :math:`U_{\rm LJ}` is the standard Lennard-Jones potential (see
    `hoomd.md.pair.LJ`). When :math:`\lambda = 0`, :math:`U` is the standard
    Weeks-Chandler-Anderson repulsive potential, while when :math:`\lambda = 1`,
    :math:`U` is :math:`U_{\rm LJ}`.

    Example::

        nl = nlist.Cell()
        perturbed_lj = azplugins.pair.PerturbedLennardJones(
            default_r_cut=3.0, nlist=nl)
        perturbed_lj.params[('A', 'A')] = dict(
            epsilon=1.0, sigma=1.0, attraction_scale_factor=0.5)
        perturbed_lj.r_cut[('A', 'B')] = 3.0

    .. py:attribute:: params

        The Perturbed LJ potential parameters. The dictionary has the following
        keys:

        * ``epsilon`` (`float`, **required**) - energy parameter
          :math:`\varepsilon` :math:`[\mathrm{energy}]`
        * ``sigma`` (`float`, **required**) - particle size :math:`\sigma`
          :math:`[\mathrm{length}]`
        * ``attraction_scale_factor`` (`float`, **required**) - scale factor
          for attraction :math:`\lambda`, between 0 and 1

        Type: :class:`~hoomd.data.typeparam.TypeParameter` [`tuple`
        [``particle_type``, ``particle_type``], `dict`]

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

    :py:class:`TwoPatchMorse` is a Morse potential that is modulated by an
    orientation-dependent function.

    .. math::

        U(\vec{r}_{ij}, \hat{n}_i, \hat{n}_j) = U_{\rm M}(|\vec{r}_{ij}|)
        \Omega(\hat{r}_{ij} \cdot \hat{n}_i) \Omega(\hat{r}_{ij} \cdot \hat{n}_j)

    where :math:`U_{\rm M}` is the potential that depends on distance

    .. math::

        U_{\rm M}(r) = M_{\rm d} \left( \left[ 1 - \exp\left(
              -\frac{r-r_{\rm eq}}{M_r}\right) \right]^2 - 1 \right)

    and :math:`\Omega(\gamma)` depends on the orientations

    .. math::
        \Omega(\gamma) = \frac{1}{1+\exp[-\omega (\gamma^2 - \alpha)]}

    The potential can be smoothed to zero force (making it purely attractive)
    when :math:`r < r_{\rm eq}` by making :math:`U_{\rm M}(r < r_{\rm eq}) = -M_{\rm d}`
    with the option  ``repulsion = False``.

    Here, :math:`\vec{r}_{ij}` is the displacement vector between particles
    :math:`i` and :math:`j`, :math:`|\vec{r}_{ij}|` is the magnitude of
    that displacement, and :math:`\hat{n}_i` is the normalized
    orientation vector of particle :math:`i`. The parameters :math:`M_{\rm d}`,
    :math:`M_{\rm r}`, and :math:`r_{\rm eq}` control the depth, width, and
    position of the potential well. The parameters :math:`\alpha` and
    :math:`\omega` control the width and steepness of the orientation dependence.

    Example::

        nl = hoomd.md.nlist.Cell()
        m2p = azplugins.pair.TwoPatchMorse(nlist=nl)
        m2p.params[('A', 'A')] = dict(M_d=1.8347, M_r=0.0302, r_eq=1.0043,
            omega=20, alpha=0.50, repulsion=True)
        m2p.r_cut[('A', 'A')] = 3.0

    .. py:attribute:: params

        The two-patch Morse potential parameters. The dictionary has the following
        keys:

        * ``M_d`` (`float`, **required**) - :math:`M_{\rm d}`, controls the
          depth of the potential well :math:`[\mathrm{energy}]`
        * ``M_r`` (`float`, **required**) - :math:`M_r` controls the width of
          the potential well  :math:`[\mathrm{length}]`
        * ``r_eq`` (`float`, **required**) - :math:`r_{\rm eq}` controls the
          position of the potential well  :math:`[\mathrm{length}]`
        * ``omega`` (`float`, **required**) - :math:`\omega` controls the
          steepness of the orientation dependence
        * ``alpha`` (`float`, **required**) - :math:`\alpha` controls the width
          of the orientation depndence
        * ``repulsion`` (`bool`, **required**) - If ``True``, include the
          repulsive part of :math:`U_{\rm M}` for :math:`r < r_{\rm eq}`.
          Otherwise, set :math:`U_{\rm r} = -M_{\rm d}` for :math:`r < r_{\rm eq}`.

        Type: :class:`~hoomd.data.typeparam.TypeParameter` [`tuple`
        [``particle_type``, ``particle_type``], `dict`]

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
