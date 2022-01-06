# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021, Auburn University
# This file is part of the azplugins project, released under the Modified BSD License.
"""
Wall potentials
===============

.. autosummary::
    :nosignatures:

    colloid
    lj93

.. autoclass:: colloid
.. autoclass:: lj93

"""
import hoomd
from hoomd import _hoomd

from . import _azplugins

class colloid(hoomd.md.wall.wallpotential):
    R""" Colloid (integrated Lennard-Jones) wall potential

    Args:
        walls (:py:class:`group`): Wall group containing half-space geometries for the force to act in.
        r_cut (float): The global r_cut value for the force. Defaults to False or 0 if not specified.
        name (str): The force name which will be used in the metadata and log files.

    Wall force evaluated using the potential obtained by integrating a spherical particle of radius
    *a* with a half plane according to the standard Lennard-Jones potential:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V(r) = & \frac{\varepsilon \sigma^6}{7560} \left( \frac{7a-z}{(z-a)^7} + \frac{7a+z}{(z+a)^7} \right)
             - \frac{\varepsilon}{6} \left( \frac{2 a z}{z^2-a^2} + \ln\left(\frac{z-a}{z+a}\right) \right) & r < r_{\rm cut} \\
              = & 0 & r \ge r_{\mathrm{cut}}
        \end{eqnarray*}

    Here, :math:`\varepsilon` is the Hamaker constant and :math:`\sigma` is the Lennard-Jones diameter
    between the particle and the wall. The Hamaker constant for a colloid density :math:`\rho_{\rm c}`,
    a wall of density :math:`\rho_{\rm w}`, and Lennard-Jones interaction strength :math:`\varepsilon_{LJ}` is
    :math:`\varepsilon = 4 \pi^2 \varepsilon_{\rm LJ} \rho_{\rm c} \rho_{\rm w} \sigma^6`.
    See :py:class:`hoomd.md.wall.wallpotential` for generalized wall potential implementation,
    including cutoff and extrapolation schemes.

    The following coefficients must be set per particle type:

    - :math:`\varepsilon` - *epsilon* (in energy units)
    - :math:`\sigma` - *sigma* (in distance units)
    - :math:`r_{\rm cut}` - `r_cut` (in distance units)
        - *optional*: defaults to the global `r_cut` specified in the pair command
    - :math:`r_{\rm extrap}` - `r_extrap` (in distance units)
        - *optional*: defaults to 0.0

    .. warning::
        This potential makes use of the particle diameters. You must make
        sure you appropriately set the particle diameters in the particle data.

    Example::

        walls=hoomd.md.wall.group()
        walls.add_plane((0,0,0), (0,0,1))

        colloid = azplugins.wall.colloid(walls, r_cut=3.0)
        colloid.force_coeff.set('A', epsilon=30.0, sigma=1.0)
        colloid.force_coeff.set('B', epsilon=100.0, sigma=1.0, r_cut=4.0)
        colloid.force_coeff.set(['C','D'], epsilon=0.0, sigma=1.0, r_cut=False)
    """
    def __init__(self, walls, r_cut=False, name=""):
        hoomd.util.print_status_line()

        # initialize the base class
        hoomd.md.wall.wallpotential.__init__(self, walls, r_cut, name)

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_class = _azplugins.WallPotentialColloid
        else:
            self.cpp_class = _azplugins.WallPotentialColloidGPU
        self.cpp_force = self.cpp_class(hoomd.context.current.system_definition, self.name)

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

        # setup the coefficent options
        self.required_coeffs += ['epsilon', 'sigma']

    def process_coeff(self, coeff):
        epsilon = coeff['epsilon']
        sigma = coeff['sigma']

        lj1 = epsilon * sigma**6 / 7560.0
        lj2 = epsilon / 6.0

        return _azplugins.make_wall_colloid_params(_hoomd.make_scalar2(lj1, lj2), coeff['r_cut']**2, coeff['r_extrap'])

class lj93(hoomd.md.wall.wallpotential):
    R""" Lennard-Jones 9-3 wall potential

    Args:
        walls (:py:class:`group`): Wall group containing half-space geometries for the force to act in.
        r_cut (float): The global r_cut value for the force. Defaults to False or 0 if not specified.
        name (str): The force name which will be used in the metadata and log files.

    Wall force evaluated using the Lennard-Jones 9-3 potential, which is obtained by integrating the
    standard Lennard-Jones potential between a point and a half plane:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V(r)  = & \varepsilon \left( \frac{2}{15}\left(\frac{\sigma}{r}\right)^9
                                    - \left(\frac{\sigma}{r}\right)^3 \right) & r < r_{\rm cut} \\
              = & 0 & r \ge r_{\rm cut}
        \end{eqnarray*}

    Here, :math:`\varepsilon` is the Hamaker constant and :math:`\sigma` is the Lennard-Jones diameter
    between the particle and the wall. The Hamaker constant for a wall of density :math:`\rho_{\rm w}`
    and with Lennard-Jones interaction strength :math:`\varepsilon_{LJ}` is
    :math:`\varepsilon = (2/3) \pi \varepsilon_{\rm LJ} \rho_{\rm w} \sigma^3`.
    See :py:class:`hoomd.md.wall.wallpotential` for generalized wall potential implementation,
    including cutoff and extrapolation schemes.

    The following coefficients must be set per particle type:

    - :math:`\varepsilon` - *epsilon* (in energy units)
    - :math:`\sigma` - *sigma* (in distance units)
    - :math:`r_{\rm cut}` - *r_cut* (in distance units)
        - *optional*: defaults to the global *r_cut* specified in the pair command
    - :math:`r_{\rm extrap}` - *r_extrap* (in distance units)
        - *optional*: defaults to 0.0

    Example::

        import numpy as np
        walls=hoomd.md.wall.group()
        walls.add_plane((0,0,0), (0,0,1))

        lj93 = azplugins.wall.lj93(walls, r_cut=3.0)
        lj93.force_coeff.set('A', epsilon=1.0, sigma=1.0)
        lj93.force_coeff.set('B', epsilon=1.0, sigma=1.0, r_cut=np.power(2.0/5.0, 1.0/6.0))
        lj93.force_coeff.set(['C','D'], epsilon=2.0, sigma=1.0, r_cut=2.0)
    """
    def __init__(self, walls, r_cut=False, name=""):
        hoomd.util.print_status_line()

        # initialize the base class
        hoomd.md.wall.wallpotential.__init__(self, walls, r_cut, name)

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_class = _azplugins.WallPotentialLJ93
        else:
            self.cpp_class = _azplugins.WallPotentialLJ93GPU
        self.cpp_force = self.cpp_class(hoomd.context.current.system_definition, self.name)

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

        # setup the coefficent options
        self.required_coeffs += ['epsilon', 'sigma']

    def process_coeff(self, coeff):
        epsilon = coeff['epsilon']
        sigma = coeff['sigma']

        # lj coefficients
        lj1 = (2.0 / 15.0) * epsilon * sigma**9
        lj2 = epsilon * sigma**3
        return _azplugins.make_wall_lj93_params(_hoomd.make_scalar2(lj1, lj2), coeff['r_cut']**2, coeff['r_extrap'])
