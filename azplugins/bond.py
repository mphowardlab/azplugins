# Copyright (c) 2018-2020, Michael P. Howard
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: astatt / Everyone is free to add additional potentials

import math

import hoomd
from hoomd import _hoomd

from . import _azplugins

class double_well(hoomd.md.bond._bond):
    R""" Double well bond potential.

    Args:
        name (str): Name of the bond instance.

    :py:class:`double_well` specifies a double well potential between the two particles in each defined bond.
    The potential is given by:

    .. math::

        V_{\rm{DW}}(r)  =  \frac{V_{max}}{b^4} \left[ \left( r - a/2 \right)^2 - b^2 \right]^2

    Coefficients:

    - :math:`V_max` - Potential maximum height between the two minima at ``a/2`` (in energy units)
    - :math:`a` - twice the location of the potential maximum, maximum is at ``a/2`` ( in distance units)
    - :math:`b` - tunes the disance between the potential minima at ``1/2(a +/- 2b)`` (in distance units)

    Examples::

        dw = azplugins.bond.double_well()
        dw.bond_coeff.set('polymer', V_max=2.0, a=2.5, b=0.5)

    """
    def __init__(self, name=None):
        hoomd.util.print_status_line()

        # check that some bonds are defined
        if hoomd.context.current.system_definition.getBondData().getNGlobal() == 0:
            hoomd.context.msg.error("azplugins.bond.double_well(): No bonds are defined.\n")
            raise RuntimeError("Error creating bond forces")

        # initialize the base class
        hoomd.md.bond._bond.__init__(self, name)

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _azplugins.BondPotentialDoubleWell(hoomd.context.current.system_definition,self.name)
        else:
            self.cpp_force =  _azplugins.BondPotentialDoubleWellGPU(hoomd.context.current.system_definition,self.name)

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

        # setup the coefficient options
        self.required_coeffs = ['V_max','a','b']

    def process_coeff(self, coeff):
        V_max = coeff['V_max']
        a = coeff['a']
        b = coeff['b']
        if b==0:
            hoomd.context.msg.error("azplugins.bond.double_well(): coefficient b must be non-zero.\n")
            raise ValueError('Coefficient b must be non-zero')

        return _hoomd.make_scalar3(V_max, a, b)


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
            self.cpp_force =  _azplugins.BondPotentialFENEGPU(hoomd.context.current.system_definition,self.name)

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


class fene24(hoomd.md.bond._bond):
    R""" modified FENE bond potential.

    Args:
        name (str): Name of the bond instance.

    :py:class:`fene24` specifies a modified FENE Ashbaugh-Hatch 48-24 potential between the two particles in each defined bond.

    .. math::
        V(r) = - \frac{1}{2} k r_0^2 \ln \left( 1 - \left( \frac{r}{r_0} \right)^2 \right) + V_{\mathrm{LJ,48-24}}(r)

    where :math:`\vec{r}` is the vector pointing from one particle to the other in the bond.
    The second part is the pair potential defined by  :py:class:`ashbaugh24`, which is a
    Lennard-Jones 48-24 potential, implemented as described by
    `L. Rovigatti, B. Capone, C. Likos, Nanoscale, 8 (2016) <http://dx.doi.org/10.1039/C5NR04661K>`_.
    The potential has a purely repulsive (Weeks-Chandler-Andersen like) core, with a
    parameter :math:`\lambda` setting the strength of the attractive tail.
    When :math:`\lambda` is 0, the potential is purely repulsive.
    When :math:`\lambda` is 1, the potential a generalized Lennard-Jones potential:
    .. math::
        :nowrap:

        \begin{eqnarray*}
        V(r)  = & V_{\mathrm{LJ,48-24}}(r, \varepsilon, \sigma) + (1-\lambda)\varepsilon & r < (2)^{1/24}\sigma \\
              = & \lambda V_{\mathrm{LJ,48-24}}(r, \varepsilon, \sigma) & (2)^{1/24}\sigma \ge r < r_{\mathrm{cut}} \\
              = & 0 & r \ge r_{\mathrm{cut}}
        \end{eqnarray*}

    Here, :math:`V_{\mathrm{LJ,48-24}}(r,\varepsilon,\sigma)` is a Lennard-Jones potential with
    parameters :math:`\varepsilon`, and :math:`\sigma`:
    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{LJ,48-24}}  = 4 \varepsilon \left(\left(\frac{\sigma}{r}\right)^{48} - \left(\frac{\sigma}{r}\right)^{24}\right)
        \end{eqnarray*}

     The following coefficients must be set per unqiue bond type:

    - :math:`k` - attractive force strength ``k`` (in units of energy/distance^2)
    - :math:`r_0` - maximal bond stretching ``r0`` (in distance units)
    - :math:`\varepsilon` -  force strength ``epsilon`` (in energy units)
    - :math:`\sigma` -  force interaction distance ``sigma`` (in distance units)
    - :math:`\lambda` -  force interaction attractive strength ``lambda`` (unitless)


    Examples::

        fene24 = azplugins.bond.fene24()
        fene24.bond_coeff.set('polymer', k=30.0, r0=1.5, sigma=1.0, lam=1.0, epsilon= 1.0)
        fene24.bond_coeff.set('backbone', k=100.0, r0=1.0, sigma=1.0,lam=0.0, epsilon= 1.0)
    """
    def __init__(self, name=None):
        hoomd.util.print_status_line()

        # check that some bonds are defined
        if hoomd.context.current.system_definition.getBondData().getNGlobal() == 0:
            hoomd.context.msg.error("No bonds are defined.\n")
            raise RuntimeError("Error creating bond forces")

        # initialize the base class
        hoomd.md.bond._bond.__init__(self,name)

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _azplugins.BondPotentialFENEAsh24(hoomd.context.current.system_definition,self.name)
        else:
            self.cpp_force = _azplugins.BondPotentialFENEAsh24GPU(hoomd.context.current.system_definition,self.name)

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

        # setup the coefficient options
        self.required_coeffs = ['k','r0','epsilon','sigma','lam']


    def process_coeff(self, coeff):
        k = coeff['k']
        r0 = coeff['r0']
        epsilon = coeff['epsilon']
        sigma = coeff['sigma']
        lam = coeff['lam']

        if epsilon==0:
            hoomd.context.msg.error("azplugins.bond.fene24(): epsilon must be non-zero.\n")
            raise ValueError('epsilon must be non-zero')
        if sigma==0:
            hoomd.context.msg.error("azplugins.bond.fene24(): sigma must be non-zero.\n")
            raise ValueError('sigma must be non-zero')
        if k==0:
            hoomd.context.msg.error("azplugins.bond.fene24(): k must be non-zero.\n")
            raise ValueError('k must be non-zero')
        if r0==0:
            hoomd.context.msg.error("azplugins.bond.fene24(): r0 must be non-zero.\n")
            raise ValueError('r0 must be non-zero')

        lj1 = 4.0 * epsilon * math.pow(sigma, 48.0)
        lj2 = 4.0 * epsilon * math.pow(sigma, 24.0)
        rwcasq = math.pow(2.0, 1.0/12.0) * sigma**2
        wca_shift = epsilon * (1. - lam)

        return _azplugins.make_ashbaugh_bond_params(lj1, lj2, lam, rwcasq, wca_shift,k,r0)
