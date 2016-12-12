# Maintainer: mphoward / Everyone is free to add additional potentials
import hoomd
import _azplugins
import math

class ashbaugh(hoomd.md.pair.pair):
    R""" Ashbaugh-Hatch potential

    Args:
        r_cut (float): Default cutoff radius (in distance units).
        nlist (:py:mod:`hoomd.md.nlist`): Neighbor list
        name (str): Name of the force instance.

    :py:class:`ashbaugh` is a Lennard-Jones perturbation potential, implemented as described by
    `Ashbaugh and Hatch <http://dx.doi.org/10.1021/ja802124e>`_. The potential has a purely
    repulsive (Weeks-Chandler-Andersen) core, with a parameter :math:`\lambda` setting the
    strength of the attractive tail. When :math:`\lambda` is 0, the potential is purely repulsive.
    When :math:`\lambda` is 1, the potential is the standard Lennard-Jones potential
    (see :py:class:`hoomd.md.pair.lj` for details).

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V(r)  = & V_{\mathrm{LJ}}(r, \varepsilon, \sigma, \alpha) + (1-\lambda)\varepsilon & r < (2/\alpha)^{1/6}\sigma \\
              = & \lambda V_{\mathrm{LJ}}(r, \varepsilon, \sigma, \alpha) & (2/\alpha)^{1/6}\sigma \ge r < r_{\mathrm{cut}} \\
              = & 0 & r \ge r_{\mathrm{cut}}
        \end{eqnarray*}

    Here, :math:`V_{\mathrm{LJ}}(r,\varepsilon,\sigma,\alpha)` is the stanard Lennard-Jones potential with
    parameters :math:`\varepsilon`, :math:`\sigma`, and :math:`\alpha`. See :py:class:`hoomd.md.pair.pair`
    for details on how forces are calculated and the available energy shifting and smoothing modes.
    Use :py:meth:`pair_coeff.set <coeff.set>` to set potential coefficients.

    The following coefficients must be set per unique pair of particle types:

    - :math:`\varepsilon` - *epsilon* (in energy units)
    - :math:`\sigma` - *sigma* (in distance units)
    - :math:`\alpha` - *alpha* (unitless) - *optional*: defaults to 1.0
    - :math:`\lambda` - *lam* (unitless) - scale factor for attraction
    - :math:`r_{\mathrm{cut}}` - *r_cut* (in distance units)
      - *optional*: defaults to the global r_cut specified in the pair command
    - :math:`r_{\mathrm{on}}`- *r_on* (in distance units)
      - *optional*: defaults to the global r_cut specified in the pair command

    Example::

        nl = hoomd.md.nlist.cell()
        ash = azplugins.pair.ashbaugh(r_cut=3.0, nlist=nl)
        ash.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, lam=0.0)
        ash.pair_coeff.set(['A','B'], 'B', epsilon=2.0, sigma=1.0, lam=0.5, r_cut=3.0, r_on=2.0);

    """
    def __init__(self, r_cut, nlist, name=None):
        hoomd.util.print_status_line()

        # initialize the base class
        hoomd.md.pair.pair.__init__(self, r_cut, nlist, name)

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_class = _azplugins.PairPotentialAshbaugh
        else:
            self.cpp_class = _azplugins.PairPotentialAshbaughGPU
            self.nlist.cpp_nlist.setStorageMode(hoomd.md._md.NeighborList.storageMode.full)
        self.cpp_force = self.cpp_class(hoomd.context.current.system_definition, self.nlist.cpp_nlist, self.name)

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

        # setup the coefficent options
        self.required_coeffs = ['epsilon', 'sigma', 'alpha', 'lam']
        self.pair_coeff.set_default_coeff('alpha', 1.0)

    def process_coeff(self, coeff):
        epsilon = coeff['epsilon']
        sigma = coeff['sigma']
        lam = coeff['lam']

        lj1 = 4.0 * epsilon * math.pow(sigma, 12.0)
        lj2 = 4.0 * epsilon * math.pow(sigma, 6.0)
        rwcasq = math.pow(2.0/alpha, 1.0/3.0) * sigma**2
        wca_shift = epsilon * alpha**2 * (1. - lam)
        return _azplugins.make_ashbaugh_params(lj1, lj2, lam, rwcasq, wca_shift)

class colloid(hoomd.md.pair.pair):
    def __init__(self, r_cut, nlist=None, name=None):
        hoomd.util.print_status_line();

        # initialize the base class
        hoomd.md.pair.pair.__init__(self, r_cut, nlist, name);

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_class = _azplugins.PairPotentialColloid
        else:
            self.cpp_class = _azplugins.PairPotentialColloidGPU
            self.nlist.cpp_nlist.setStorageMode(hoomd.md._md.NeighborList.storageMode.full)
        self.cpp_force = self.cpp_class(hoomd.context.current.system_definition, self.nlist.cpp_nlist, self.name)

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficent options
        self.required_coeffs = ['epsilon', 'sigma', 'style'];
        self.pair_coeff.set_default_coeff('epsilon', 144.0);
        self.pair_coeff.set_default_coeff('sigma', 1.0);

    ## Process the coefficients
    def process_coeff(self, coeff):
        epsilon = coeff['epsilon'];
        sigma = coeff['sigma'];
        style = coeff['style']

        if style == 'slv-slv':
            style = 0
        elif style == 'coll-slv':
            style = 1
        elif style == 'coll-coll':
            style = 2
        else:
            hoomd.context.msg.error('Unknown interaction style\n')
            raise RuntimeError('Unknown interaction style')

        return hoomd._hoomd.make_scalar4(epsilon, sigma**3, sigma**6, hoomd._hoomd.int_as_scalar(style));
