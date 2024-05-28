# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2022, Auburn University
# This file is part of the azplugins project, released under the Modified BSD License.
"""
MPCD methods
============

.. autosummary::
    :nosignatures:

    compute_velocity

.. autoclass:: compute_velocity

"""


import hoomd

from . import _azplugins

class compute_velocity(hoomd.compute._compute):
    r"""Compute center-of-mass velocity of MPCD particles

    Args:
        suffix (str): Suffix to attach to logged quantities.

    This computes the center-of-mass velocity of all MPCD particles:

    .. math::

        \mathbf{v}_{\rm cm} = \dfrac{\sum_i m_i \mathbf{v}_i}{\sum_i m_i}

    where :math:`\mathbf{v}_i` is the velocity and and :math:`m_i` is the mass
    of particle *i* in the group. Note that because all MPCD particles currently
    have the same mass, this is equivalent to the number-average velocity.

    The components of the result are exposed as loggable quantities ``mpcd_vx``,
    ``mpcd_vy``, and ``mpcd_vz`` with ``suffix`` appended. By default,
    ``suffix`` is an empty string, but a custom suffix may be specified if needed
    to distinguish the logged quantity from something else. You can save these
    results using :py:class:`hoomd.analyze.log`.

    Example::

        # without suffix
        azplugins.mpcd.compute_velocity()
        hoomd.analyze.log(filename='velocity.dat', quantities=['mpcd_vx'], period=10)

        # with suffix
        azplugins.mpcd.compute_velocity(suffix='-srd')
        hoomd.analyze.log(filename='velocity_srd.dat', quantities=['mpcd_vx-srd'], period=100)

    """
    def __init__(self, suffix=None):
        hoomd.util.print_status_line()
        super().__init__()

        # create suffix for logged quantity
        if suffix is None:
            suffix = ''

        if suffix in self._used_suffixes:
            hoomd.context.msg.error('azplugins.mpcd.velocity: Suffix {} already used\n'.format(suffix))
            raise ValueError('Suffix {} already used for MPCD velocity'.format(suffix))
        else:
            self._used_suffixes.append(suffix)

        # create the c++ mirror class
        self.cpp_compute = _azplugins.MPCDVelocityCompute(hoomd.context.current.mpcd.data, suffix)
        hoomd.context.current.system.addCompute(self.cpp_compute, self.compute_name);

    _used_suffixes = []
