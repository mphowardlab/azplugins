# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2022, Auburn University
# This file is part of the azplugins project, released under the Modified BSD License.
"""
Variants
========

.. autosummary::
    :nosignatures:

    sphere_area

.. autoclass:: sphere_area

"""
import hoomd

from . import _azplugins

class sphere_area(hoomd.variant._variant):
    r"""Radius of sphere contracting with a constant rate of surface-area reduction.

    Args:
        R0 (float): Initial radius (units: length).
        alpha (float): Rate of surface-area reduction (units: area per timestep).

    The radius of the sphere :math:`R(t)` is reduced according to:

    .. math::

        R(t) = \sqrt{R(0)^2 - (\alpha/4\pi) t}

    where :math:`R(0)` is the initial radius and :math:`\alpha` is the rate of
    surface-area reduction per time. This physics corresponds to the evaporation
    of a droplet into stagnant air under the mapping:

    .. math::

        \alpha = \frac{8 \pi D_{\rm v} m \Delta p}{\rho_{\rm l} k_{\rm B} T}

    with :math:`D_{\rm v}` the diffusivity in the air, *m* the mass of the
    evaporating solvent molecule, :math:`\Delta p` the difference in vapor
    pressure between the droplet surface and far away, and :math:`\rho_{\rm l}`
    is the mass density of the liquid.

    To be physical, :math:`R(t)` is never negative. It is fixed to 0 if *t*
    would make the square-root argument negative. Setting :math:`\alpha < 0`
    will cause the sphere to instead expand.

    Example::

        variant.sphere_area(R0=50., alpha=1.)

    """
    def __init__(self, R0, alpha):
        hoomd.util.print_status_line()

        # initialize parent class
        hoomd.variant._variant.__init__(self)

        # set parameters of variant
        self.metadata_fields = ['R0','alpha']
        self.R0 = R0
        self.alpha = alpha

        # initialize c++ mirror class
        self.cpp_variant = _azplugins.VariantSphereArea(self.R0, self.alpha)
