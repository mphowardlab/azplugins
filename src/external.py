# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2024, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""External harmonic potentials."""

import hoomd

from hoomd.azplugins import _azplugins
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.data.parameterdicts import ParameterDict
from hoomd.md import force


class HarmonicBarrier(force.Force):
    r"""Repulsive barrier implemented as a harmonic potential.

    This class should not be instantiated directly. Use a derived type.

    Args:
        interface (`hoomd.variant.variant_like`): Position of the interface.

    .. py:attribute:: interface

        Position of the interface. The meaning of this position is interpreted
        by derived types.

        Type: `hoomd.variant.variant_like`

    .. py:attribute:: params

        The parameters of the harmonic barrier for each particle type.
        The dictionary has the following keys:

        * ``k`` (`float`, **required**) - Spring constant :math:`k`
          :math:`[\mathrm{energy} \cdot \mathrm{length}^{-2}]`
        * ``offset`` (`float`, **required**) - Shift amount per-particle-type :math:`H`
          :math:`[\mathrm{length}]`


        Type: :class:`~hoomd.data.typeparam.TypeParameter` [`tuple`
        [``particle_type``], `dict`]

    .. warning::

        Virial calculation has not been implemented for this external field.

    """

    def __init__(self, interface):
        super().__init__()

        param_dict = ParameterDict(interface=hoomd.variant.Variant)
        param_dict["interface"] = interface
        param_dict.update(dict(interface=interface))
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

        self._cpp_obj = cpp_class(sim.state._cpp_sys_def, self.interface)

        super()._attach_hook()

    def _make_cpp_class_name(self):
        cpp_class_name = self.__class__.__name__
        if isinstance(self._simulation.device, hoomd.device.GPU):
            cpp_class_name += "GPU"
        return cpp_class_name


class PlanarHarmonicBarrier(HarmonicBarrier):
    r"""Planar harmonic barrier normal to *z*.

    Args:
        interface (`hoomd.variant.variant_like`): *z* position of the
            interface.

    `PlanarHarmonicBarrier` applies a purely harmonic potential in a planar
    geometry with a normal in the :math:`z` direction. Particles are pushed
    in the :math:`-z` direction when they are above the ``interface``:

    .. math::

        U(z) = \begin{cases}
            0, & z \le H \\
            \dfrac{\kappa}{2} (z-H)^2, & z > H
        \end{cases}

    Here, the interface is located at height *H*, which may change with time.
    ``interface`` specifies the nominal *H*, which may then be modified
    per-particle-type by adding an ``offset``. :math:`\kappa` is a spring
    constant setting the strength of the interface, similar to a surface tension.

    Example::

        # moving interface from H = 50. to H = 25.
        interf = hoomd.variant.Ramp(A=50.0, B=25.0, t_start=100, t_ramp=1e6)
        evap = hoomd.azplugins.external.PlanarHarmonicBarrier(interface=interf)

        # small particle has diameter 1.0
        evap.params['S'] = dict(k=50.0, offset=-0.5)

        # big particle is twice as large (diameter 2.0)
        # spring constant scales with diameter squared, offset with diameter
        evap.params['B'] = dict(k=200.0, offset=-1.0)

    """


class SphericalHarmonicBarrier(HarmonicBarrier):
    r"""Spherical harmonic barrier.

    Args:
        interface (`hoomd.variant.variant_like`): Radius of sphere.

    `SphericalHarmonicBarrier` applies a purely harmonic potential to particles
    outside the radius of a sphere, acting to push them inward:

    .. math::

        U(r) = \begin{cases}
            0, & r \le R \\
            \dfrac{\kappa}{2} (r-R)^2, & r > R
        \end{cases}

    Here, the interface is located at radius *R*, which may change with time.
    ``interface`` specifies the nominal *R*, which may then be modified
    per-particle-type by adding an ``offset``. :math:`\kappa` is a spring
    constant setting the strength of the interface, similar to a surface tension.

    Example::

        # moving interface from R = 50. to R = 25.
        interf = hoomd.variant.Ramp(A=50.0, B=25.0, t_start=100, t_ramp=1e6)
        evap = hoomd.azplugins.external.SphericalHarmonicBarrier(interface=interf)

        # small particle has diameter 1.0, offset by -0.5 to keep fully inside
        evap.params['S'] = dict(k=50.0, offset=-0.5)

        # big particle is twice as large (diameter 2.0)
        # spring constant scales with diameter squared, offset with diameter
        evap.params['B'] = dict(k=200.0, offset=-1.0)
    """
