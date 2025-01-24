# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2025, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""External potentials."""

import hoomd

from hoomd.azplugins import _azplugins
from hoomd.data.parameterdicts import ParameterDict, TypeParameterDict
from hoomd.data.typeconverter import OnlyTypes, variant_preprocessing
from hoomd.data.typeparam import TypeParameter
from hoomd.md import force


class HarmonicBarrier(force.Force):
    r"""Repulsive barrier implemented as a harmonic potential.

    This class should not be instantiated directly. Use a derived type.

    Args:
        location (`hoomd.variant.variant_like`): Location of the barrier.

    .. py:attribute:: location

        Location of the barrier. The meaning of this location is interpreted
        by derived types.

        Type: `hoomd.variant.variant_like`

    .. py:attribute:: params

        The parameters of the harmonic barrier for each particle type.
        The dictionary has the following keys:

        * ``k`` (`float`, **required**) - Spring constant
          :math:`[\mathrm{energy} \cdot \mathrm{length}^{-2}]`
        * ``offset`` (`float`, **required**) - Amount added to ``location``
          :math:`[\mathrm{length}]`


        Type: :class:`~hoomd.data.typeparam.TypeParameter` [`tuple`
        [``particle_type``], `dict`]

    .. warning::

        The contribution to the virial is not calculated!

    """

    def __init__(self, location):
        super().__init__()

        param_dict = ParameterDict(
            location=OnlyTypes(hoomd.variant.Variant, preprocess=variant_preprocessing),
        )
        param_dict["location"] = location
        self._param_dict.update(param_dict)

        params = TypeParameter(
            "params",
            "particle_types",
            TypeParameterDict(k=float, offset=float, len_keys=1),
        )
        self._add_typeparam(params)

    def _attach_hook(self):
        sim = self._simulation

        cpp_class = getattr(_azplugins, self._make_cpp_class_name())

        self._cpp_obj = cpp_class(sim.state._cpp_sys_def, self.location)

        super()._attach_hook()

    def _make_cpp_class_name(self):
        cpp_class_name = self.__class__.__name__
        if isinstance(self._simulation.device, hoomd.device.GPU):
            cpp_class_name += "GPU"
        return cpp_class_name


class PlanarHarmonicBarrier(HarmonicBarrier):
    r"""Planar harmonic barrier normal to *y*.

    Args:
        location (`hoomd.variant.variant_like`): *y* position of the
            barrier.

    `PlanarHarmonicBarrier` applies a purely repulsive harmonic potential
    in a planar geometry with a normal in the :math:`y` direction. Particles
    are pushed in the :math:`-y` direction when they are above the
    ``location``:

    .. math::

        U(y) = \begin{cases}
            0, & y \le H \\
            \dfrac{\kappa}{2} (y-H)^2, & y > H
        \end{cases}

    Here, the barrier is positioned at :math:`y=H`, specified by ``location``,
    which may then be modified per-particle-type by adding an ``offset``.
    :math:`\kappa` is a spring constant setting the strength of the barrier.

    Example::

        # moving barrier from H = 50. to H = 25.
        barrier = hoomd.variant.Ramp(A=50.0, B=25.0, t_start=100, t_ramp=1e6)
        evap = hoomd.azplugins.external.PlanarHarmonicBarrier(location=barrier)

        # small particle has diameter 1.0, offset by -0.5 to keep fully inside
        evap.params['S'] = dict(k=50.0, offset=-0.5)

        # big particle is twice as large (diameter 2.0)
        # spring constant scales with diameter squared, offset with diameter
        evap.params['B'] = dict(k=200.0, offset=-1.0)

    """


class SphericalHarmonicBarrier(HarmonicBarrier):
    r"""Spherical harmonic barrier.

    Args:
        location (`hoomd.variant.variant_like`): Radius of sphere.

    `SphericalHarmonicBarrier` applies a purely repulsive harmonic potential to
    particles outside the radius of a sphere, acting to push them inward:

    .. math::

        U(r) = \begin{cases}
            0, & r \le R \\
            \dfrac{\kappa}{2} (r-R)^2, & r > R
        \end{cases}

    Here, the barrier is positioned at radius *R*, specified by ``location``,
    which may then be modified per-particle-type by adding an ``offset``.
    :math:`\kappa` is a spring constant setting the strength of the barrier.

    Example::

        # moving barrier from R = 50 to R = 25
        barrier = hoomd.variant.Ramp(A=50.0, B=25.0, t_start=100, t_ramp=1e6)
        evap = hoomd.azplugins.external.SphericalHarmonicBarrier(location=barrier)

        # small particle has diameter 1.0, offset by -0.5 to keep fully inside
        evap.params['S'] = dict(k=50.0, offset=-0.5)

        # big particle is twice as large (diameter 2.0)
        # spring constant scales with diameter squared, offset with diameter
        evap.params['B'] = dict(k=200.0, offset=-1.0)

    """
