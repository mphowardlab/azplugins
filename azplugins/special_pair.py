# Copyright (c) 2016-2018, Panagiotopoulos Group, Princeton University
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: mphoward / Everyone is free to add additional potentials

import hoomd
from hoomd import _hoomd
from hoomd.md import _md
import _azplugins
import math

class lj96(hoomd.md.special_pair._special_pair):
    R""" LJ special pair potential.

    Args:
        name (str): Name of the force instance.

    :py:class:`lj96` is a Lennard-Jones potential which is useful for implementing special 1-3 or 1-4 interactions

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V(r)  = & \frac{27}{4} \varepsilon \left(\left(\frac{\sigma}{r}\right)^9 - \alpha \left(\frac{\sigma}{r}\right)^6\right) \\
                & -\frac{27}{4} \varepsilon \left(\left(\frac{\sigma}{r_{\mathrm{cut}}}\right)^9 \\
                & -\alpha \left(\frac{\sigma}{r_{\mathrm{cut}}}\right)^6\right) & r < r_{\mathrm{cut}} \\
              = & 0 & r \ge r_{\mathrm{cut}}
        \end{eqnarray*}

    parameters :math:`\varepsilon`, :math:`\sigma`, and :math:`\alpha`. See :py:class:`hoomd.md.special_pair._special_pair`
    for details on how forces are calculated.
    Use :py:meth:`pair_coeff.set <coeff.set>` to set potential coefficients.

    The following coefficients must be set per unique pair of particle types:

    - :math:`\varepsilon` - *epsilon* (in energy units)
    - :math:`\sigma` - *sigma* (in distance units)
    - :math:`\alpha` - *alpha* (unitless) - *optional*: defaults to 1.0
    - :math:`r_{\mathrm{cut}}` - *r_cut* (in distance units)
    - mode - energy shift mode ("shift" or "no_shift") - optional: defaults to "no_shift"

    Example::

        lj96_intra = azplugins.special_pair.lj96()
        lj96_intra.pair_coeff.set('A-A', epsilon=2.0, sigma=1.0, r_cut=3.0)
        lj96_intra.pair_coeff.set('A-B', epsilon=1.5, sigma=0.8, alpha=0.5, r_cut=2.5)

    """
    def __init__(self,name=None):
        hoomd.util.print_status_line()

        # initiailize the base class
        hoomd.md.special_pair._special_pair.__init__(self)

        # check that some bonds are defined
        if hoomd.context.current.system_definition.getPairData().getNGlobal() == 0:
            hoomd.context.msg.error("No pairs are defined.\n")
            raise RuntimeError("Error creating special pair forces")

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _azplugins.SpecialPairPotentialLJ96(hoomd.context.current.system_definition,self.name)
        else:
            self.cpp_force = _azplugins.SpecialPairPotentialLJ96GPU(hoomd.context.current.system_definition,self.name)

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

        # setup the coefficient options
        self.required_coeffs = ['epsilon','sigma','alpha','r_cut', 'mode']
        self.pair_coeff.set_default_coeff('alpha', 1.0)
        self.pair_coeff.set_default_coeff('mode', "no_shift")

    def process_coeff(self, coeff):
        r_cut = coeff['r_cut']
        epsilon = coeff['epsilon']
        sigma = coeff['sigma']
        alpha = coeff['alpha']

        lj1 = 27.0 / 4.0 * epsilon * math.pow(sigma, 9.0)
        lj2 = alpha * 27.0 / 4.0 * epsilon * math.pow(sigma, 6.0)
        r_cut_squared = r_cut * r_cut

        if coeff['mode'] == "shift":
            energy_shift = True
        elif coeff['mode'] == "no_shift":
            energy_shift = False
        else:
            hoomd.context.msg.error("Energy shift mode not set to 'shift' or 'no_shift'")
            raise RuntimeError("Error creating special pair forces")

        return _azplugins.make_special_pair_params_lj96(_hoomd.make_scalar2(lj1, lj2), r_cut_squared, energy_shift)
