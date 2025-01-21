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
    r"""Applies harmonic barrier.

    This class should not be instantiated directly. Use a derived type.

    Args:
        interface (`hoomd.variant.variant_like`): position of the interface

    .. py:attribute:: params

        The parameters of the harmonic barrier for each particle type.
        The dictionary has the following:
        keys:

        * ``k`` (`float`, **required**) - Spring constant :math:`k`
          :math:`[\mathrm{energy} \cdot \mathrm{length}^{-2}]`
        * ``offset`` (`float`, **required**) - Shift amount per-particle-type :math:`H`
          :math:`[\mathrm{length}]`


        Type: :class:`~hoomd.data.typeparam.TypeParameter` [`tuple`
        [``particle_type``], `dict`]

    .. warning::
        Virial calculation has not been implemented for this model because it is
        nonequilibrium. A warning will be raised if any calls to
        :py:class:`hoomd.logging.Logger` are made because the logger always requests
        the virial flags. However, this warning can be safely ignored if the
        pressure (tensor) is not being logged or the pressure is not of interest.

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
        interface (`hoomd.variant.variant_like`) : *z* position of the interface

    `PlanarHarmonicBarrier` apply a purely harmonic potential in planar geometry
    with a normal in the :math:`+z` direction:

    .. math::

        U(z) = \begin{cases}
         0 & z \le H \\
         \dfrac{\kappa}{2} (z-H)^2 & z > H
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
        evap.params['S'] = dict(k=50.0, offset=0.0)
        # big particle is twice as large (diameter 2.0), so coefficient is scaled
        evap.params['B'] = dict(k=50.0*2*2, offset=0.0)

    """

    pass


class SphericalHarmonicBarrier(HarmonicBarrier):
    r"""Spherical harmonic barrier.

    Args:
        interface (`hoomd.variant.variant_like`) : Radius of spherical interface.

    `SphericalHarmonicBarrier` applies a purely harmonic potential to particles
    near a spherical interface that pushes particles inward toward the center
    of the sphere. The potential is truncated at its minimum
    and has the following form:

    .. math::

        V(r) = \begin{cases}
            0, & \text{if } r < R \\
            \frac{\kappa}{2} (r-R)^2, & \text{if } r > R
        \end{cases}

    Here, the interface is located at radius *R*, and may change with time.
    The effective interface position *R* may be modified per-particle-type
    using an ``offset`` (added to *R* to determine the effective position).
    :math:`\kappa` is a spring constant that determines the strength of the interface
    (:math:`\kappa` is a proxy for surface tension).

    Example::

        # moving interface from R = 25. to R = 10.
        interf = hoomd.variant.Ramp(A=25.0, B=10.0, t_start=0, t_ramp=1e6)
        barrier = hoomd.azplugins.external.SphericalHarmonicBarrier(interface=interf)
        barrier.params['A', 'B'] = dict(k=50.0, offset=0.0)
    """

    pass
