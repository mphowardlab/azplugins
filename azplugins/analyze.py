# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2022, Auburn University
# This file is part of the azplugins project, released under the Modified BSD License.
"""
Analyzers
=========

.. autosummary::
    :nosignatures:

    group_velocity
    rdf

.. autoclass:: group_velocity

.. autoclass:: rdf

"""
import numpy
import hoomd

from . import _azplugins

class group_velocity(hoomd.compute._compute):
    r"""Group center-of-mass velocity compute

    Args:
        group (:py:mod:`hoomd.group`): Group to compute velocity of.
        suffix (str): Suffix to attach to logged quantities.

    This computes the center-of-mass velocity of a group:

    .. math::

        \mathbf{v}_{\rm cm} = \dfrac{\sum_i m_i \mathbf{v}_i}{\sum_i m_i}

    where :math:`\mathbf{v}_i` is the velocity and and :math:`m_i` is the mass
    of particle *i* in the group.

    The components of the result are exposed as loggable quantities ``vx``,
    ``vy``, and ``vz`` with ``suffix`` appended. By default, ``suffix`` is
    ``_name`` where ``name`` is the name of the ``group``, like ``_all`` for
    :py:class:`hoomd.group.all`. However, a custom suffix may be specified; the
    only requirement is that the same suffix cannot be used more than once. You
    can save these results using :py:class:`hoomd.analyze.log`.

    Example::

        # all particles
        azplugins.analyze.group_velocity(hoomd.group.all())
        hoomd.analyze.log(filename='velocity.dat', quantities=['vx_all'], period=10)

        # suffix comes from group name
        azplugins.analyze.group_velocity(hoomd.group.type('A',name='A'))
        hoomd.analyze.log(filename='velocity_A.dat', quantities=['vx_A'], period=50)

        # suffix is manually set
        azplugins.analyze.group_velocity(hoomd.group.type('B'), suffix='-big')
        hoomd.analyze.log(filename='velocity_big.dat', quantities=['vx-big'], period=100)

    """
    def __init__(self, group, suffix=None):
        hoomd.util.print_status_line()
        super().__init__()

        # group object
        self.group = group

        # create suffix for logged quantity
        if suffix is None:
            suffix = '_' + group.name

        if suffix in self._used_suffixes:
            hoomd.context.msg.error('azplugins.analyze.group_velocity: Suffix {} already used\n'.format(suffix))
            raise ValueError('Suffix {} already used for group velocity'.format(suffix))
        else:
            self._used_suffixes.append(suffix)

        # create the c++ mirror class
        self.cpp_compute = _azplugins.GroupVelocityCompute(hoomd.context.current.system_definition, group.cpp_group, suffix)
        hoomd.context.current.system.addCompute(self.cpp_compute, self.compute_name);

    _used_suffixes = []

class rdf(hoomd.analyze._analyzer):
    R""" Radial distribution function calculator

    Args:
        groups (list): Two groups (:py:mod:`hoomd.group.group`) to calculate RDF between
        rcut (float): Cutoff radius
        bin_width (float): Bin width of distribution function
        period (int): Distribution function will be computed every *period* time steps
        phase (int): When -1, start on the current time step. Otherwise, execute
                     on steps where *(step + phase) % period* is 0.

    The radial distribution function, :math:`g(r)`, is computed between the
    *groups*. (The *groups* can be the same.)

    Example::

        rdf_AA = azplugins.analyze.rdf(groups=(group_A, group_A), rcut=3.0, bin_width=0.1, period=100)
        rdf_AB = azplugins.analyze.rdf(groups=[group_A, group_B], rcut=6.0, bin_width=0.5, period=1000)
        run(10000)

        # read the current values of the RDF
        bins, gr = rdf_AA()

    .. warning::
        The calculation is currently implemented as an all-pairs check between
        the two *groups*. This calculation can proceed extremely slowly when the
        group sizes are large.

    .. warning::
        MPI is not currently supported by this analyzer. An error will be raised
        if one is created when multiple ranks are detected.

    """
    def __init__(self, groups, rcut, bin_width, period, phase=0):
        hoomd.util.print_status_line()

        if hoomd.comm.get_num_ranks() > 1:
            hoomd.context.msg.error('azplugins.analyze.rdf: MPI is not currently supported\n')
            raise RuntimeError('RDF in MPI is not currently supported')

        # validate the groups being passed in
        if (len(groups) != 2 or
            type(groups[0]) is not hoomd.group.group or
            type(groups[1]) is not hoomd.group.group):
            hoomd.context.msg.error('azplugins.analyze.rdf: Two groups must be passed as a list\n')
            raise TypeError('Two groups must be passed as a list')

        # validate cutoff
        if rcut <= 0.0:
            hoomd.context.msg.error('azplugins.analyze.rdf: Cutoff radius must be positive\n')
            raise ValueError('Cutoff radius must be positive')

        # validate bin width
        if bin_width <= 0.0:
            hoomd.context.msg.error('azplugins.analyze.rdf: Bin width must be positive\n')
            raise ValueError('Bin width must be positive')

        # call constructor now that the rest has been sanitized
        super(rdf, self).__init__()
        if not hoomd.context.exec_conf.isCUDAEnabled():
            cpp_class = _azplugins.RDFAnalyzer
        else:
            cpp_class = _azplugins.RDFAnalyzerGPU
        self.cpp_analyzer = cpp_class(hoomd.context.current.system_definition,
                                      groups[0].cpp_group,
                                      groups[1].cpp_group,
                                      rcut,
                                      bin_width)
        self.setupAnalyzer(period, phase)

        # log metadata fields
        self.metadata_fields = ['groups','rcut','bin_width','period','phase']
        self.groups = groups
        self.rcut = rcut
        self.bin_width = bin_width
        self.period = period
        self.phase = phase

    def __call__(self):
        R""" Get the current value of the distribution function

        Returns:
            (tuple): tuple containing

                bins (ndarray): Center of bins where distribution function was evaluated
                gr (ndarray): Distribution function

        """
        return numpy.array(self.cpp_analyzer.getBins()), numpy.array(self.cpp_analyzer.get())

    def reset(self):
        R""" Reset the accumulated distribution function

        The accumulated distribution function is zeroed so that analysis can begin freshly.
        """
        self.cpp_analyzer.reset()
