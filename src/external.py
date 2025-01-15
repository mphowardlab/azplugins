# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2022, Auburn University
# This file is part of the azplugins project, released under the Modified BSD License.


import hoomd

from hoomd.azplugins import _azplugins
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import OnlyTypes, variant_preprocessing
from hoomd.md import force

class HarmonicBarrier(force.Force):

    r"""Applies Harmonic Potential.

    This class should not be instantiated directly. Use a derived type.

    Args:
        interface (hoomd.variant.variant): position of the interface
        geometry (str): Drying geometry ("film" or "droplet")

    Args:
        k (float): Harmonic coefficient

        offset (float): per-particle-type amount to shift in the potential

    """

    def __init__(self, interface=None, geometry=None):
        super().__init__()

        if geometry not in ('film', 'droplet'):
            raise ValueError(f"Unrecognized geometry: {geometry}")

        param_dict = ParameterDict(interface=hoomd.variant.Variant,
                                   geometry=str)
        param_dict["interface"] = interface
        param_dict["geometry"]= geometry
        param_dict.update(dict(interface=interface, geometry=geometry,))
        self._param_dict.update(param_dict)

        params = TypeParameter(
            "params",
            "particle_types",
            TypeParameterDict(
                k=float, offset=float, len_keys=1
            ),
        )
        self._add_typeparam(params)

    def _attach_hook(self):
        sim = self._simulation

        cpp_class = getattr(_azplugins, self._make_cpp_class_name())

        if self.geometry not in ('film', 'droplet'):
            raise ValueError(f"Unrecognized geometry: {self.geometry}")

        self._cpp_obj = cpp_class(
            sim.state._cpp_sys_def,
            self.interface
        )

        super()._attach_hook()

    def _make_cpp_class_name(self):
        cpp_class_name = self.__class__.__name__
        if isinstance(self._simulation.device, hoomd.device.GPU):
            cpp_class_name += "GPU"
        return cpp_class_name

class PlanarHarmonicBarrier(HarmonicBarrier):
    def __init__(self, interface=None):
        super().__init__(interface=interface, geometry='film')

    r"""Applies a purely repulsive harmonic potential to particles near an interface in planar geometry..

    Args:
        interface (`hoomd.variant`) : `z` position of the interface

    `PlanarHarmonicBarrier` apply a purely harmonic potential in planar geometry
    that pushes down on nonvolatile solutes. The potential is truncated at its minimum
    and have following form:

    .. math::

        \begin{eqnarray*}
        V(d) = & 0 & d < H \\
               & \frac{\kappa}{2} (d-H)^2 & d > H 
        \end{eqnarray*}

    Here, the interface is located at height *H*, and may change with time.
    The effective interface position *H* may be modified per-particle-type using a *offset*
    (*offset* is added to *H* to determine the effective *H*)
    :math:`\kappa` is a spring constant setting the strength of the interface
    (:math:`\kappa` is a proxy for the surface tension).
    
    Example::

        interf = hoomd.variant.Ramp(A=50.0, B=25.0, t_start=100, t_ramp=1e6)
        evap = hoomd.azplugins.external.PlanarHarmonicBarrier(interface = interf)

        evap.params['S'] = dict(k=50.0, offset=0.0)
        evap.params['B'] = dict(k=50.0*2*2, offset=0.0)

    .. py:attribute:: params

        The `PlanarHarmonicBarrier` parameters. The dictionary has the following
        keys:

        * ``k`` (`float`, **required**) - Spring constant :math:`k`
          :math:`[\mathrm{energy} \cdot \mathrm{length}^{-2}]`
        * ``offset`` (`float`, **required**) - per-particle-type amount to shift :math:`H`
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


class SphericalHarmonicBarrier(HarmonicBarrier):
    def __init__(self, interface=None):
        super().__init__(interface=interface, geometry='droplet')
    r"""Apply pure repulsive Harmonic Potential in a spherical geometry.

    SphericalHarmonicBarrier applies a purely repulsive harmonic potential to particles 
    near a spherical interface.

    Args:
        interface (`hoomd.variant`): The radial position of the spherical interface.

    The `SphericalHarmonicBarrier` applies a purely harmonic potential in spherical geometry 
    that pushes particles inward toward the center of the sphere. The potential is truncated 
    at its minimum and has the following form:

    .. math::

        V(r) =
        \begin{cases}
            0, & \text{if } r < R \\
            \frac{\kappa}{2} (r-R)^2, & \text{if } r > R
        \end{cases}

    Here, the interface is located at radius :math:`R`, and may change with time. The effective 
    interface position :math:`R` may be modified per-particle-type using an *offset* 
    (added to :math:`R` to determine the effective position). :math:`\kappa` is a spring constant 
    that determines the strength of the interface (:math:`\kappa` is a proxy for surface tension).

    Example::
        interf = hoomd.variant.Ramp(A=25.0, B=10.0, t_start=0, t_ramp=1e6)
        barrier = hoomd.azplugins.external.SphericalHarmonicBarrier(interface=interf)
        barrier.params['A'] = dict(k=100.0, offset=0.0)
        barrier.params['B'] = dict(k=150.0, offset=2.0)

    .. py:attribute:: params

        The `SphericalHarmonicBarrier` parameters. The dictionary has the following keys:

        * ``k`` (`float`, **required**) - Spring constant :math:`k`
          :math:`[\mathrm{energy} \cdot \mathrm{length}^{-2}]`
        * ``offset`` (`float`, **required**) - Per-particle-type amount to shift :math:`R`
          :math:`[\mathrm{length}]`

        Type: :class:`~hoomd.data.typeparam.TypeParameter` [`particle_type`, `dict`]

    .. warning::
        Virial calculation has not been implemented for this model because it is 
        nonequilibrium. A warning will be raised if any calls to 
        :py:class:`hoomd.logging.Logger` are made because the logger always requests 
        the virial flags. However, this warning can be safely ignored if the 
        pressure (tensor) is not being logged or the pressure is not of interest.
    """
