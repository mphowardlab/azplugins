# Copyright (c) 2018-2019, Michael P. Howard
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: mphoward / Everyone is free to add additional variants

import hoomd

from . import _azplugins

class sphere(hoomd.variant._variant):
    r"""Radius of sphere contracting with a constant rate of surface reduction.

    The volume of the sphere is reduced according to:
    .. math::

        V(t)^{2/3} = V(0)^{2/3} - \alpha t

    where the sphere volume is :math:`V = 4 \pi R^3/3$ for radius *R* and
    :math:`\alpha` is the rate of surface reduction. This physics corresponds to
    the evaporation of a droplet into stagnant air.

    The droplet radius is then:
    .. math::

        R(t) = \sqrt{R(0)^2 - (1/4)(6/\pi)^{2/3} \alpha t}

    To be physical, :math:`R(t)`` is never negative. It is fixed to 0 if *t*
    would make the square-root argument negative.

    Setting \f$\alpha < 0\f$ will cause the sphere to expand.

    Example::

        variant.sphere(R0=50., alpha=1.)

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
        self.cpp_variant = _azplugins.VariantSphere(self.R0, self.alpha)
