# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2022, Auburn University
# This file is part of the azplugins project, released under the Modified BSD License.
"""
Analyzers
=========

.. autosummary::
    :nosignatures:

    group_velocity

.. autoclass:: group_velocity

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
