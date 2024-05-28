# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2022, Auburn University
# This file is part of the azplugins project, released under the Modified BSD License.
"""
Bond potentials
===============

.. autosummary::
    :nosignatures:

    double_well

.. autoclass:: double_well

"""

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

        V_{\rm{DW}}(r)  =  \frac{V_{max}-c/2}{b^4} \left[ \left( r - a/2 \right)^2 - b^2 \right]^2 +\frac{c}{2b}(r - a/2) + c/2

    Coefficients:

    - :math:`V_max` - Potential maximum energy barrier between the two minima at ``a/2`` for c=0 (in energy units)
    - :math:`a` - twice the location of the potential maximum, maximum is at ``a/2`` for c=0 ( in distance units)
    - :math:`b` - tunes the distance between the potential minima at ``(a/2 +/- b)`` for c=0 (in distance units)
    - :math:`c` - tunes the energy offset between the two potential minima values, i.e. it tilts the
                  potential (in energy units). The default value of c is zero.

    Examples::

        dw = azplugins.bond.double_well()
        dw.bond_coeff.set('polymer', V_max=2.0, a=2.5, b=0.5)
        dw.bond_coeff.set('polymer', V_max=2.0, a=2.5, b=0.2, c=1.0)
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

        self.required_coeffs = ['V_max','a','b','c']
        self.bond_coeff.set_default_coeff('c', 0.0)

    def process_coeff(self, coeff):
        V_max = coeff['V_max']
        a = coeff['a']
        b = coeff['b']
        c = coeff['c']

        if b==0:
            hoomd.context.msg.error("azplugins.bond.double_well(): coefficient b must be non-zero.\n")
            raise ValueError('Coefficient b must be non-zero')

        return _hoomd.make_scalar4(V_max, a, b, c)
