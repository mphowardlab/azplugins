# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2022, Auburn University
# This file is part of the azplugins project, released under the Modified BSD License.
"""
DPD potentials
==============

.. autosummary::
    :nosignatures:

    general

.. autoclass:: general

"""

import hoomd
from hoomd import _hoomd
from hoomd.md import _md

from . import _azplugins

class general(hoomd.md.pair.pair):
    R""" Dissipative Particle Dynamics with generalized weight function

    Args:
        r_cut (float): Default cutoff radius (in distance units).
        nlist (:py:mod:`hoomd.md.nlist`): Neighbor list
        kT (:py:mod:`hoomd.variant` or :py:obj:`float`): Temperature of thermostat (in energy units).
        seed (int): seed for the PRNG in the DPD thermostat.
        name (str): Name of the force instance.

    :py:class:`general` specifies that a DPD pair force should be applied between every
    non-excluded particle pair in the simulation, including an interaction potential,
    pairwise drag force, and pairwise random force. The form of the forces between
    pairs of particles is:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \mathbf{F} = \mathbf{F}_{\rm C} + \mathbf{F}_{\rm D} +  \mathbf{F}_{\rm R} \\
        \end{eqnarray*}

    The conservative force :math:`\mathbf{F}_{\rm C}` is the standard form:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \mathbf{F}_{\rm C} =& A (1- r_{ij}/r_{\rm cut}) & r \le r_{\rm cut} \\
                           =& 0 & r > r_{\rm cut}
        \end{eqnarray*}

    where *A* is the interaction parameter and :math:`r_{\rm cut}` is the cutoff radius.
    Here, :math:`r_{ij} = r_i - r_j`. See `Groot and Warren 1997 <http://dx.doi.org/10.1063/1.474784>`_
    for more details.

    The dissipative and random forces, respectively, are:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \mathbf{F}_{\rm D} =& -\gamma \omega_{\rm D}(r_{ij}) (\mathbf{v}_{ij} \cdot \mathbf{\hat r}_{ij}) \mathbf{\hat r}_{ij} \\
        \mathbf{F}_{\rm R} =& \sigma \omega_{\rm R}(r_{ij}) \xi_{ij} \mathbf{\hat r}_{ij}
        \end{eqnarray*}

    where :math:`\sigma = 2\gamma k_{\rm B}T` and :math:`\omega_{\rm D} = \left[\omega_{\rm R} \right]^2`
    to satisfy the fluctuation dissipation relation. The genealized weight function is given by the
    form proposed by `Fan et al. <https://doi.org/10.1063/1.2206595>`_:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        w_{\rm D}(r) = &\left( 1 - r/r_{\mathrm{cut}} \right)^s  & r \le r_{\mathrm{cut}} \\
                     = & 0 & r > r_{\mathrm{cut}} \\
        \end{eqnarray*}

    :py:class:`general` generates random numbers by hashing together the particle tags in the pair, the user seed,
    and the current time step index.

    .. attention::

        Change the seed if you reset the simulation time step to 0. If you keep the same seed, the simulation
        will continue with the same sequence of random numbers used previously and may cause unphysical correlations.

        For MPI runs: all ranks other than 0 ignore the seed input and use the value of rank 0.

    `C. L. Phillips et. al. 2011 <http://dx.doi.org/10.1016/j.jcp.2011.05.021>`_ describes the DPD implementation
    details in HOOMD-blue. Cite it if you utilize the DPD functionality in your work.

    The following coefficients must be set per unique pair of particle types:

    - :math:`A` - *A* (in force units)
    - :math:`\gamma` - *gamma* (in units of force/velocity)
    - :math:`s` - *s* (*optional*: defaults to 2 for standard DPD)
    - :math:`r_{\mathrm{cut}}` - *r_cut* (in distance units)
      - *optional*: defaults to the global r_cut specified in the pair command

    To use the DPD thermostat, an :py:class:`hoomd.md.integrate.nve` integrator must be applied to the system and
    the user must specify a temperature.  Use of the dpd thermostat pair force with other integrators will result
    in unphysical behavior. To use this DPD potential with a different conservative potential than :math:`F_C`,
    set A to zero and define the conservative pair potential separately.

    Example::

        nl = hoomd.md.nlist.cell()
        dpd = azplugins.dpd.general(r_cut=1.0, nlist=nl, kT=1.0, seed=42)
        dpd.pair_coeff.set('A', 'A', A=25.0, gamma=4.5, s=1.)
        hoomd.md.integrate.mode_standard(dt=0.02)
        hoomd.md.integrate.nve(group=group.all())

    """
    def __init__(self, r_cut, nlist, kT, seed, name=None):
        hoomd.util.print_status_line()

        # register the citation
        c = hoomd.cite.article(cite_key='phillips2011',
                         author=['C L Phillips', 'J A Anderson', 'S C Glotzer'],
                         title='Pseudo-random number generation for Brownian Dynamics and Dissipative Particle Dynamics simulations on GPU devices',
                         journal='Journal of Computational Physics',
                         volume=230,
                         number=19,
                         pages='7191--7201',
                         month='Aug',
                         year='2011',
                         doi='10.1016/j.jcp.2011.05.021',
                         feature='DPD')
        hoomd.cite._ensure_global_bib().add(c)

        # initialize the base class
        hoomd.md.pair.pair.__init__(self, r_cut, nlist, name)

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            cpp_class = _azplugins.DPDPotentialGeneralWeight
        else:
            cpp_class = _azplugins.DPDPotentialGeneralWeightGPU
            self.nlist.cpp_nlist.setStorageMode(_md.NeighborList.storageMode.full)
        self.cpp_force = cpp_class(hoomd.context.current.system_definition, self.nlist.cpp_nlist, self.name)

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

        # setup the coefficent options
        self.required_coeffs = ['A', 'gamma', 's']
        self.pair_coeff.set_default_coeff('s', 2)

        # set the seed for dpd thermostat
        self.cpp_force.setSeed(seed)

        hoomd.util.quiet_status()
        self.set_params(kT)
        hoomd.util.unquiet_status()

    def set_params(self, kT=None):
        R""" Changes parameters.

        Args:
            kT (:py:mod:`hoomd.variant` or :py:obj:`float`): Temperature of thermostat (in energy units).

        Example::

            dpd.set_params(kT=2.0)
        """
        hoomd.util.print_status_line()
        self.check_initialization()

        if kT is not None:
            kT = hoomd.variant._setup_variant_input(kT)
            self.cpp_force.setT(kT.cpp_variant)

    def process_coeff(self, coeff):
        a = coeff['A']
        gamma = coeff['gamma']
        s = coeff['s']
        return _hoomd.make_scalar3(a, gamma, s)
