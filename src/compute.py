# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2024, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""Computes."""

import itertools

import numpy

from hoomd import _hoomd
from hoomd.azplugins import _azplugins
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import OnlyTypes
from hoomd.filter import ParticleFilter
from hoomd.logging import log
from hoomd.operation import Compute


class CylindricalVelocityField(Compute):
    """Compute velocity field in cylindrical coordinates."""

    def __init__(
        self,
        num_bins,
        lower_bounds,
        upper_bounds,
        filter=None,
        include_mpcd_particles=False,
    ):
        super().__init__()

        param_dict = ParameterDict(
            num_bins=(int, int, int),
            lower_bounds=(float, float, float),
            upper_bounds=(float, float, float),
            filter=OnlyTypes(ParticleFilter, allow_none=True),
            include_mpcd_particles=bool(include_mpcd_particles),
        )
        param_dict.update(
            dict(
                num_bins=num_bins,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                filter=filter,
            )
        )
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        sim = self._simulation

        cpp_class = _azplugins.CylindricalFlowProfileCompute

        num_bins = _hoomd.uint3()
        num_bins.x = self.num_bins[0]
        num_bins.y = self.num_bins[1]
        num_bins.z = self.num_bins[2]

        lower_bounds = _hoomd.make_scalar3(
            self.lower_bounds[0], self.lower_bounds[1], self.lower_bounds[2]
        )
        upper_bounds = _hoomd.make_scalar3(
            self.upper_bounds[0], self.upper_bounds[1], self.upper_bounds[2]
        )

        if self.filter is not None:
            group = sim.state._get_group(self.filter)
        else:
            group = None

        self._cpp_obj = cpp_class(
            sim.state._cpp_sys_def,
            num_bins,
            lower_bounds,
            upper_bounds,
            group,
            self.include_mpcd_particles,
        )

    @property
    def coordinates(self):
        """numpy.ndarray: Coordinates of bin centers."""
        coords = []
        shape = []
        for num, lo, hi in zip(self.num_bins, self.lower_bounds, self.upper_bounds):
            if num > 0:
                dx = (hi - lo) / num
                x = lo + dx * (numpy.arange(num) + 0.5)
                coords.append(x)
                shape.append(num)

        if len(shape) == 0:
            return None

        if len(shape) > 1:
            shape.append(len(shape))

        return numpy.reshape(list(itertools.product(*coords)), shape)

    @log(category="sequence", requires_run=True)
    def velocities(self):
        """numpy.ndarray: Mass-averaged velocity of bin."""
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.velocities
