# Copyright (c) 2016-2017, Panagiotopoulos Group, Princeton University
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: astatt / Everyone is free to add additional potentials

import hoomd
from hoomd import _hoomd
from hoomd.md import _md
import _azplugins
import math

class fene(hoomd.md.bond._bond):
    R""" FENE bond potential.

    Args:
        name (str): Name of the bond instance.

    :py:class:`fene` specifies a FENE potential energy between the two particles in each defined bond.

    .. math::

        V(r) = - \frac{1}{2} k r_0^2 \ln \left( 1 - \left( \frac{r}{r_0} \right)^2 \right) + V_{\mathrm{WCA}}(r)

    where :math:`\vec{r}` is the vector pointing from one particle to the other in the bond.
    The potential :math:`V_{\mathrm{WCA}}(r)` is given by:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{WCA}}(r)  = & 4 \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{12} - \left( \frac{\sigma}{r} \right)^{6} \right]  + \varepsilon & r < 2^{\frac{1}{6}}\sigma\\
                   = & 0          & r\ge 2^{\frac{1}{6}}\sigma
        \end{eqnarray*}

    Coefficients:

    - :math:`k` - attractive force strength ``k`` (in units of energy/distance^2)
    - :math:`r_0` - size parameter ``r0`` (in distance units)
    - :math:`\varepsilon` - repulsive force strength ``epsilon`` (in energy units)
    - :math:`\sigma` - repulsive force interaction distance ``sigma`` (in distance units)

    Examples::

        fene = azplugins.bond.fene()
        fene.bond_coeff.set('polymer', k=30.0, r0=1.5, sigma=1.0, epsilon=2.0)
        fene.bond_coeff.set('backbone', k=100.0, r0=1.0, sigma=1.0, epsilon= 2.0)

    """
    def __init__(self, name=None):
        hoomd.util.print_status_line()

        # check that some bonds are defined
        if hoomd.context.current.system_definition.getBondData().getNGlobal() == 0:
            hoomd.context.msg.error("azplugins.bond.fene(): No bonds are defined.\n")
            raise RuntimeError("Error creating bond forces")

        # initialize the base class
        hoomd.md.bond._bond.__init__(self, name)

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _azplugins.BondPotentialFENE(hoomd.context.current.system_definition,self.name)
        else:
            self.cpp_force =  _azplugins.BondPotentialFENE(hoomd.context.current.system_definition,self.name)

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

        # setup the coefficient options
        self.required_coeffs = ['k','r0','epsilon','sigma']

    def process_coeff(self, coeff):
        k = coeff['k']
        r0 = coeff['r0']
        epsilon = coeff['epsilon']
        sigma = coeff['sigma']
        lj1 = 4.0 * epsilon * math.pow(sigma, 12.0)
        lj2 = 4.0 * epsilon * math.pow(sigma, 6.0)

        if epsilon==0:
            hoomd.context.msg.error("azplugins.bond.fene(): epsilon must be non-zero.\n")
            raise ValueError('epsilon must be non-zero')
        if sigma==0:
            hoomd.context.msg.error("azplugins.bond.fene(): sigma must be non-zero.\n")
            raise ValueError('sigma must be non-zero')
        if k==0:
            hoomd.context.msg.error("azplugins.bond.fene(): k must be non-zero.\n")
            raise ValueError('k must be non-zero')
        if r0==0:
            hoomd.context.msg.error("azplugins.bond.fene(): r0 must be non-zero.\n")
            raise ValueError('r0 must be non-zero')

        return _hoomd.make_scalar4(k, r0, lj1, lj2)
