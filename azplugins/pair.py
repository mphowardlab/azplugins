import hoomd
import _azplugins
import math

class ashbaugh(hoomd.md.pair.pair):
    R""" Ashbaugh-Hatch potential

    Args:
        r_cut (float): Default cutoff radius (in distance units).
        nlist (:py:mod:`hoomd.md.nlist`): Neighbor list
        name (str): Name of the force instance.

    :py:class:`ashbaugh` specifies that a Lennard-Jones-style pair potential should be applied between every
    non-excluded particle pair in the simulation. The potential has a purely repulsive (Weeks-Chandler-Andersen)
    core, with a parameter :math:`\lambda` setting the strength of the attractive tail. When :math:`\lambda`
    is 0, the potential is purely repulsive. When :math:`\lambda` is 1, the potential is the standard
    Lennard-Jones potential (see :py:class:`lj` for details).

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V(r)  = & V_{\mathrm{LJ}}(r, \varepsilon, \sigma) + (1-\lambda)\varepsilon & r < 2^{1/6}\sigma \\
              = & \lambda V_{\mathrm{LJ}}(r, \varepsilon, \sigma) & 2^{1/6}\sigma \ge r < r_{\mathrm{cut}} \\
              = & 0 & r \ge r_{\mathrm{cut}}
        \end{eqnarray*}

    Here, :math:`V_{\mathrm{LJ}}(r,\varepsilon,\sigma)` is the stanard Lennard-Jones potential with
    parameters :math:`\varepsilon`, :math:`\sigma`, and :math:`\alpha=1`. (See :py:class:`lj`
    for details.) See :py:class:`pair` for details on how forces are calculated and the available energy shifting
    and smoothing modes. Use :py:meth:`pair_coeff.set <coeff.set>` to set potential coefficients.

    The following coefficients must be set per unique pair of particle types:

    - :math:`\varepsilon` - *epsilon* (in energy units)
    - :math:`\sigma` - *sigma* (in distance units)
    - :math:`\lambda` - *lam* (unitless) - scale factor for attraction (between 0 and 1)
    - :math:`r_{\mathrm{cut}}` - *r_cut* (in distance units)
      - *optional*: defaults to the global r_cut specified in the pair command
    - :math:`r_{\mathrm{on}}`- *r_on* (in distance units)
      - *optional*: defaults to the global r_cut specified in the pair command

    Example::

        nl = nlist.cell()
        lj = pair.ashbaugh(r_cut=3.0, nlist=nl)
        lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, lam=0.0)
        lj.pair_coeff.set(['A','B'], 'B', epsilon=2.0, sigma=1.0, lam=0.5, r_cut=3.0, r_on=2.0);

    """
    def __init__(self, r_cut, nlist, name=None):
        hoomd.util.print_status_line();

        # initialize the base class
        hoomd.md.pair.pair.__init__(self, r_cut, nlist, name);

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            #self.cpp_force = _md.PotentialPairLJLambda(hoomd.context.current.system_definition, self.nlist.cpp_nlist, self.name);
            self.cpp_class = _azplugins.PairPotentialAshbaugh
        else:
            self.nlist.cpp_nlist.setStorageMode(hoomd.md._md.NeighborList.storageMode.full)
            #self.cpp_force = _azplugins.PairPotentialAshbaughGPU(hoomd.context.current.system_definition, self.nlist.cpp_nlist, self.name);
            self.cpp_class = _azplugins.PairPotentialAshbaughGPU
        self.cpp_force = self.cpp_class(hoomd.context.current.system_definition, self.nlist.cpp_nlist, self.name)

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficent options
        self.required_coeffs = ['epsilon', 'sigma', 'lam'];

    def process_coeff(self, coeff):
        epsilon = coeff['epsilon'];
        sigma = coeff['sigma'];
        lam = coeff['lam'];

        if lam < 0.0 or lam > 1.0:
            hoomd.context.msg.error("pair.ashbaugh: lambda parameter must be between 0 and 1\n")
            raise RuntimeError("lambda parameter must be between 0 and 1")

        lj1 = 4.0 * epsilon * math.pow(sigma, 12.0)
        lj2 = 4.0 * epsilon * math.pow(sigma, 6.0)
        rwcasq = math.pow(2.0, 1.0/3.0) * sigma**2
        wca_shift = epsilon * (1. - lam)
        return _azplugins.make_ashbaugh_params(lj1, lj2, lam, rwcasq, wca_shift)
