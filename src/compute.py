# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2024, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""Computes."""

import itertools

import numpy

import hoomd
from hoomd.azplugins import _azplugins
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import OnlyTypes
from hoomd.error import DataAccessError


class VelocityCompute(hoomd.operation.Compute):
    r"""Center-of-mass velocity compute.

    Args:
        filter (hoomd.filter.ParticleFilter): HOOMD particles to include.
            The default value of `None` means no HOOMD particles are included.
        include_mpcd_particles (bool): If `True`, include MPCD particles in
            the calculation. This argument only takes effect if HOOMD was
            compiled with the MPCD component.

    `VelocityCompute` calculates the center-of-mass velocity
    :math:`\mathbf{v}_{\rm cm}` of a group of particles:

    .. math::

        \mathbf{v}_{\rm cm} = \dfrac{\sum_i m_i \mathbf{v}_i}{\sum_i m_i}

    where :math:`\mathbf{v}_i` is the velocity and and :math:`m_i` is the mass
    of particle *i*. These particles include regular HOOMD particles (specified
    with `filter`) and MPCD particles (specified with `include_mpcd_particles`).

    Example::

        v_com = hoomd.azplugins.compute.VelocityCompute(
            filter=hoomd.filter.All()
        )
        v_com.velocity

    Attributes:
        filter (hoomd.filter.filter_like): HOOMD particles to include
            (*read only*).

        include_mpcd_particles(bool): If `True`, include MPCD particles in the
            calculation (*read only*).

    """

    def __init__(self, filter=None, include_mpcd_particles=False):
        super().__init__()

        param_dict = ParameterDict(
            filter=OnlyTypes(hoomd.filter.ParticleFilter, allow_none=True),
            include_mpcd_particles=bool(include_mpcd_particles),
        )
        param_dict["filter"] = filter

        self._param_dict.update(param_dict)

    def _attach_hook(self):
        sim = self._simulation

        if isinstance(sim.device, hoomd.device.GPU):
            cpp_class = _azplugins.VelocityComputeGPU
        else:
            cpp_class = _azplugins.VelocityCompute

        if self.filter is not None:
            group = sim.state._get_group(self.filter)
        else:
            group = None

        self._cpp_obj = cpp_class(
            sim.state._cpp_sys_def,
            group,
            self.include_mpcd_particles,
        )

        super()._attach_hook()

    @hoomd.logging.log(category="sequence", requires_run=True)
    def velocity(self):
        """tuple[float]: Center-of-mass velocity of group."""
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.velocity


class VelocityFieldCompute(hoomd.operation.Compute):
    r"""Compute velocity field.

    This class should not be instantiated directly. Use a derived type.

    Args:
        num_bins (tuple[int]): Number of bins along each of the 3 cylindrical
            coordinates. A value of zero indicates the coordinate is ignored.
        lower_bounds (tuple[float]): Lower bounds for each coordinate. The
            value of this bound is ignored if the number of bins is zero.
        upper_bounds (tuple[float]): Upper bounds for each coordinate. The
            value of this bound is ignored if the number of bins is zero.
        filter (hoomd.filter.ParticleFilter): HOOMD particles to include in calculation.
            The default value of `None` means no HOOMD particles are included.
        include_mpcd_particles (bool): If `True`, include MPCD particles in
            the calculation. This argument only takes effect if HOOMD was
            compiled with the MPCD component.

    Args:
        num_bins (tuple[int]): Number of bins along each of the 3 cylindrical
            coordinates.

            A value of zero indicates the coordinate is ignored.

        lower_bounds (tuple[float]): Lower bounds for each coordinate.

            The value of this bound is ignored if the number of bins is zero.

        upper_bounds (tuple[float]): Upper bounds for each coordinate.

            The value of this bound is ignored if the number of bins is zero.

        filter (hoomd.filter.ParticleFilter): HOOMD particles to include in
            calculation (*read only*).

            `None` means no HOOMD particles are included.

        include_mpcd_particles (bool): If `True`, include MPCD particles in
            the calculation (*read only*).

            This property only has meaning effect if HOOMD was compiled with
            the MPCD component.

    """

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
            filter=OnlyTypes(hoomd.filter.ParticleFilter, allow_none=True),
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

        cpp_class = getattr(_azplugins, self._make_cpp_class_name())

        num_bins = hoomd._hoomd.uint3()
        num_bins.x = self.num_bins[0]
        num_bins.y = self.num_bins[1]
        num_bins.z = self.num_bins[2]

        lower_bounds = hoomd._hoomd.make_scalar3(
            self.lower_bounds[0], self.lower_bounds[1], self.lower_bounds[2]
        )
        upper_bounds = hoomd._hoomd.make_scalar3(
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

        super()._attach_hook()

    def _make_cpp_class_name(self):
        cpp_class_name = self.__class__.__name__
        if isinstance(self._simulation.device, hoomd.device.GPU):
            cpp_class_name += "GPU"
        return cpp_class_name

    @property
    def coordinates(self):
        """numpy.ndarray: Coordinates of bin centers.

        If binning is performed in more than 1 dimension, a multidimensional
        array is returned.

        If binning is performed in 1 dimension, a 1 dimensional array is
        returned.

        """
        coords = []
        shape = []
        for num, lo, hi in zip(self.num_bins, self.lower_bounds, self.upper_bounds):
            if num > 0:
                x, dx = numpy.linspace(lo, hi, num, endpoint=False, retstep=True)
                x += 0.5 * dx
                coords.append(x)
                shape.append(num)

        if len(shape) == 0:
            return None

        if len(shape) > 1:
            shape.append(len(shape))

        return numpy.reshape(list(itertools.product(*coords)), shape)

    @property
    def velocities(self):
        """numpy.ndarray: Mass-averaged velocity vector of bin.

        The velocities are returned as an array of 3-dimensional vectors
        matching the binning shape. This quantity is only available after the
        simulation has been run.

        """
        if not self._attached:
            raise DataAccessError("velocities")
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.velocities


class CartesianVelocityFieldCompute(VelocityFieldCompute):
    r"""Compute velocity field in Cartesian coordinates.

    Args:
        num_bins (tuple[int]): Number of bins along each of the 3 cylindrical
            coordinates. A value of zero indicates the coordinate is ignored.
        lower_bounds (tuple[float]): Lower bounds for each coordinate. The
            value of this bound is ignored if the number of bins is zero.
        upper_bounds (tuple[float]): Upper bounds for each coordinate. The
            value of this bound is ignored if the number of bins is zero.
        filter (hoomd.filter.ParticleFilter): HOOMD particles to include in calculation.
            The default value of `None` means no HOOMD particles are included.
        include_mpcd_particles (bool): If `True`, include MPCD particles in
            the calculation. This argument only takes effect if HOOMD was
            compiled with the MPCD component.

    `CartesianVelocityFieldCompute` calculates the mass-averaged velocity in a
    bin using the Cartesian coordinate system :math:`(x, y, z)`. Particles
    that lie outside the lower and upper bounds are ignored.

    Example::

        velocity_field = hoomd.azplugins.compute.CartesianVelocityFieldCompute(
            num_bins=(10, 8, 0),
            lower_bounds=(-5, 0, 0),
            upper_bounds=(5, 5, 0),
            filter=hoomd.filter.All(),
        )

    """


class CylindricalVelocityFieldCompute(VelocityFieldCompute):
    r"""Compute velocity field in cylindrical coordinates.

    Args:
        num_bins (tuple[int]): Number of bins along each of the 3 cylindrical
            coordinates. A value of zero indicates the coordinate is ignored.
        lower_bounds (tuple[float]): Lower bounds for each coordinate. The
            value of this bound is ignored if the number of bins is zero.
        upper_bounds (tuple[float]): Upper bounds for each coordinate. The
            value of this bound is ignored if the number of bins is zero.
        filter (hoomd.filter.ParticleFilter): HOOMD particles to include in calculation.
            The default value of `None` means no HOOMD particles are included.
        include_mpcd_particles (bool): If `True`, include MPCD particles in
            the calculation. This argument only takes effect if HOOMD was
            compiled with the MPCD component.

    `CylindricalVelocityFieldCompute` calculates the mass-averaged velocity in a
    bin using the cylindrical coordinate system :math:`(r, \theta, z)`, where
    :math:`0 \le \theta < 2\pi`. The cylindrical position coordinates are
    related to the Cartesian coordinates :math:`(x, y, z)` by

    .. math::

        r = \sqrt{x^2 + y^2} \\
        \theta = \arctan\left(\frac{y}{x}\right) \\
        z = z

    Before averaging, Cartesian velocity vectors are converted to the
    cylindrical coordinate system using the change-of-basis matrix:

    .. math::

        \left(\begin{array}{ccc}
        \cos \theta & \sin \theta & 0 \\
        -\sin \theta & \cos \theta & 0 \\
        0 & 0 & 1
        \end{array}\right)

    Particles that lie outside the lower and upper bounds are ignored.

    Example::

        velocity_field = hoomd.azplugins.compute.CylindricalVelocityFieldCompute(
            num_bins=(10, 8, 0),
            lower_bounds=(0, 0, 0),
            upper_bounds=(10, 2*numpy.pi, 0),
            filter=hoomd.filter.All(),
            )

    """
