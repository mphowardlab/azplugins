# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2025, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""Variant."""

import numpy
import hoomd
from hoomd.azplugins import _azplugins


class VariantInterpolated(_azplugins.VariantInterpolated, hoomd.variant.Variant):
    """Piecewise-linear variant on a uniform grid of time."""

    def __init__(self, data, t_lo, t_hi):
        hoomd.variant.Variant.__init__(self)

        data = numpy.asarray(data, dtype=float)
        if data.size < 2:
            raise ValueError("data must contain at least 2 values.")
        if t_hi <= t_lo:
            raise ValueError("t_hi must be greater than t_lo.")

        self._data = data
        _azplugins.VariantInterpolated.__init__(
            self, data, data.size, float(t_lo), float(t_hi)
        )
