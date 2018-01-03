# Copyright (c) 2016-2018, Panagiotopoulos Group, Princeton University
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: mphoward / Everyone is free to add additional analyzers

import hoomd
from hoomd import _hoomd
from hoomd.md import _md
import _azplugins
import numpy

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
