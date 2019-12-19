# Copyright (c) 2018-2019, Michael P. Howard
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: wes_reinhart

import numpy
import hoomd
from hoomd import _hoomd
from hoomd.md import force

from . import _azplugins

class cylinder(force._force):
    r"""Apply a harmonic potential to restrain particles to a cylinder.

    Args:
        group (:py:mod:`hoomd.group`): Group of particles to apply potential to.
        radius (float): Radius of the cylinder.
        origin (tuple): Center of the cylinder.
        axis (tuple): Axis of the cylinder.
        k (float): Harmonic spring constant.

    The harmonic potential is:

    .. math::

        U(d) = \frac{k}{2} d^2

    where *k* is the spring constant and *d* is the distance of the point from the cylinder
    using the **unwrapped** particle position.

    The reason that the position is unwrapped is to ensure that the harmonic potential is always
    increasing. Wrapping **r** would introduce force discontinuities and set a maximum value
    for *U*. However, the virial contribution is still computed by applying the force at the
    wrapped position, as the same force is applied to all images. Note that there is still a
    maximum set on *U* by the *radius*.

    .. note::
        The cylinder must be transformed to point along *axis*. This achieved by a
        quaternion rotation relative to the *z* axis. Choosing an axis other than `(0,0,1)`
        may incur small numerical errors if HOOMD is compiled in single-precision.

    Examples::

        hp = azplugins.restrain.cylinder(group=hoomd.group.all(), radius=10, origin=(0,0,0), axis=(0,0,1), k=10.0)

    """
    def __init__(self, group, radius, origin, axis, k):
        hoomd.util.print_status_line()

        # initialize the base class
        force._force.__init__(self)

        # create the c++ mirror class
        if hoomd.context.exec_conf.isCUDAEnabled():
            _cpp = _azplugins.CylinderRestraintComputeGPU
        else:
            _cpp = _azplugins.CylinderRestraintCompute

        # process the parameters
        self._radius = radius
        self._origin = _hoomd.make_scalar3(origin[0],origin[1],origin[2])
        self._axis = _hoomd.make_scalar3(axis[0],axis[1],axis[2])

        self.cpp_force = _cpp(hoomd.context.current.system_definition,
                              group.cpp_group,
                              _azplugins._CylinderWall(self._radius, self._origin, self._axis, True),
                              k)

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

    def set_params(self, radius=None, origin=None, axis=None, k=None):
        R""" Update the cylinder geometry or spring constant.

        Args:
            radius (float): Radius of the cylinder.
            origin (tuple): Center of the cylinder.
            axis (tuple): Axis of the cylinder.
            k (float): Harmonic spring constant.

        Parameters are only updated if they are specified.

        Examples::

            hp.set_params(radius=5.)
            hp.set_params(radius=8, origin=(0,1,0), axis=(0,1,0), k=5.0)

        """
        hoomd.util.print_status_line()
        self.check_initialization()

        if radius is not None:
            self._radius = radius
            self.cpp_force.getWall().radius = self._radius

        if origin is not None:
            self._origin = _hoomd.make_scalar3(origin[0],origin[1],origin[2])

        if axis is not None:
            self._axis = _hoomd.make_scalar3(axis[0],axis[1],axis[2])

        # replace the wall object if something has changed
        if origin is not None or axis is not None:
            self.cpp_force.setWall(_azplugins._CylinderWall(self._radius, self._origin, self._axis, True))

        if k is not None:
            self.cpp_force.setForceConstant(k)

    def update_coeffs(self):
        pass

class plane(force._force):
    r"""Apply a harmonic potential to restrain particles to a plane.

    Args:
        group (:py:mod:`hoomd.group`): Group of particles to apply potential to.
        point (tuple): Point in the plane.
        normal (tuple): Normal of the plane.
        k (float): Harmonic spring constant.

    The harmonic potential is:

    .. math::

        U(d) = \frac{k}{2} d^2

    where *k* is the spring constant and *d* is the distance of the point from the plane:

    .. math::

        d = (\mathbf{r)-\mathbf{p}) \cdot \mathbf{n}

    where **r** is the **unwrapped** particle position, **p** is a point in the plane,
    and **n** is the unit normal of the plane.

    The reason that **r** is unwrapped is to ensure that the harmonic potential is always
    increasing. Wrapping **r** would introduce force discontinuities and set a maximum value
    for *U*. However, the virial contribution is still computed by applying the force at the
    wrapped position, as the same force is applied to all images.

    This potential is especially useful if a group of particles needs to be restrained to a
    region of space.

    Examples::

        hp = azplugins.restrain.plane(group=hoomd.group.all(), point=(0,0,0), normal=(0,0,1), k=10.0)

    """
    def __init__(self, group, point, normal, k):
        hoomd.util.print_status_line()

        # initialize the base class
        force._force.__init__(self)

        # create the c++ mirror class
        if hoomd.context.exec_conf.isCUDAEnabled():
            _cpp = _azplugins.PlaneRestraintComputeGPU
        else:
            _cpp = _azplugins.PlaneRestraintCompute

        # process the parameters
        self._p = _hoomd.make_scalar3(point[0],point[1],point[2])
        self._n = _hoomd.make_scalar3(normal[0],normal[1],normal[2])

        self.cpp_force = _cpp(hoomd.context.current.system_definition,
                              group.cpp_group,
                              _azplugins._PlaneWall(self._p, self._n, True),
                              k)

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

    def set_params(self,point=None, normal=None, k=None):
        R""" Update the plane geometry or spring constant.

        Args:
            point (tuple): Point in the plane.
            normal (tuple): Normal of the plane.
            k (float): Harmonic spring constant.

        Parameters are only updated if they are specified.

        Examples::

            hp.set_params(point=(1,0,0))
            hp.set_params(point=(0,1,0), normal=(0,1,0), k=5.0)

        """
        hoomd.util.print_status_line()
        self.check_initialization()

        if point is not None:
            self._p = _hoomd.make_scalar3(point[0],point[1],point[2])

        if normal is not None:
            self._n = _hoomd.make_scalar3(normal[0],normal[1],normal[2])

        # replace the wall object if something has changed
        if point is not None or normal is not None:
            self.cpp_force.setWall(_azplugins._PlaneWall(self._p, self._n, True))

        if k is not None:
            self.cpp_force.setForceConstant(k)

    def update_coeffs(self):
        pass

class sphere(force._force):
    r"""Apply a harmonic potential to restrain particles to a sphere.

    Args:
        group (:py:mod:`hoomd.group`): Group of particles to apply potential to.
        radius (float): Radius of the sphere.
        origin (tuple): Center of the cylinder.
        k (float): Harmonic spring constant.

    The harmonic potential is:

    .. math::

        U(d) = \frac{k}{2} d^2

    where *k* is the spring constant and *d* is the distance of the point from the sphere
    using the **unwrapped** particle position.

    The reason that the position is unwrapped is to ensure that the harmonic potential is always
    increasing. Wrapping **r** would introduce force discontinuities and set a maximum value
    for *U*. However, the virial contribution is still computed by applying the force at the
    wrapped position, as the same force is applied to all images. Note that there is still a
    maximum set on *U* by the *radius*.

    Examples::

        hp = azplugins.restrain.sphere(group=hoomd.group.all(), radius=10, origin=(0,0,0), k=10.0)

    """
    def __init__(self, group, radius, origin, k):
        hoomd.util.print_status_line()

        # initialize the base class
        force._force.__init__(self)

        # create the c++ mirror class
        if hoomd.context.exec_conf.isCUDAEnabled():
            _cpp = _azplugins.SphereRestraintComputeGPU
        else:
            _cpp = _azplugins.SphereRestraintCompute

        # process the parameters
        self._radius = radius
        self._origin = _hoomd.make_scalar3(origin[0],origin[1],origin[2])

        self.cpp_force = _cpp(hoomd.context.current.system_definition,
                              group.cpp_group,
                              _azplugins._SphereWall(self._radius, self._origin, True),
                              k)

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

    def set_params(self, radius=None, origin=None, k=None):
        R""" Update the sphere geometry or spring constant.

        Args:
            radius (float): Radius of the sphere.
            origin (tuple): Center of the sphere.
            k (float): Harmonic spring constant.

        Parameters are only updated if they are specified.

        Examples::

            hp.set_params(radius=5.)
            hp.set_params(radius=8, origin=(0,1,0), k=5.0)

        """
        hoomd.util.print_status_line()
        self.check_initialization()

        if radius is not None:
            self._radius = radius
            self.cpp_force.getWall().radius = self._radius

        if origin is not None:
            self._origin = _hoomd.make_scalar3(origin[0],origin[1],origin[2])
            self.cpp_force.setWall(_azplugins._SphereWall(self._radius, self._origin, True))

        if k is not None:
            self.cpp_force.setForceConstant(k)

    def update_coeffs(self):
        pass

class position(force._force):
    R""" Add a harmonic restraining potential based on initial (or specified) position

    Args:
        group Group of atoms to apply restraint to
        k (float or array_like) Force constant, isotropic or in each of *x*-, *y*-, and *z*-directions
        name (str) (optional) name for potential

    The Hamiltonian is augmented with a harmonic potential calculated based on the distance between the current and
    initial positions of particles. This effectively allows particles to have nearly fixed position while still retaining
    their degrees of freedom. The potential has the following form:

    .. math::
        :nowrap:
        \begin{equation*}
        V(\mathbf{r}) = \frac{1}{2} \mathbf{k} \mathbf{\Delta r} \mathbf{\Delta r}^T
        \end{equation*}

    The strength of the potential depends on a spring constant :math:`\mathbf{k}` so that the particle position can be
    restrained in any of the three coordinate directions. If :math:`\mathbf{k}` is very large, the particle position is
    essentially constrained. However, because the particles retain their degrees of freedom, shorter integration timesteps
    must be taken for large :math:`\mathbf{k}` to maintain stability.

    .. note::
        The displacement is calculated using the minimum image convention.

    Examples::

        restrain.position(group=group.all(), k=1.0)
        springs = azplugins.restrain.position(group=wall, k=(100.0, 200.0, 300.0))

    .. warning::
        Virial calculation is not implemented because the particles are tethered to fixed positions. A warning will be raised
        if any calls to :py:class:`hoomd.analyze.log` are made because the logger always requests the virial flags. However,
        this warning can be safely ignored if the pressure (tensor) is not being logged or the pressure is not of interest.

    """
    def __init__(self, group, k, name=""):
        hoomd.util.print_status_line()

        # initialize the base class
        force._force.__init__(self,name)

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            cpp_class = _azplugins.PositionRestraintCompute
        else:
            cpp_class = _azplugins.PositionRestraintComputeGPU
        self.cpp_force = cpp_class(hoomd.context.current.system_definition,group.cpp_group)

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

        hoomd.util.quiet_status()
        self.set_params(k=k)
        hoomd.util.unquiet_status()

    def set_params(self, k):
        R""" Set the position restraint parameters.

        Args:
            k (float or array_like): Force constant, isotropic or in each of *x*-, *y*-, and *z*-directions

        Examples::

            springs = azplugins.restrain.position(group=hoomd.group.all(), k=0.0)
            springs.set_params(k=1.)
            springs.set_params(k=(1.,2.,3.))
            springs.set_params(k=[1.,2.,3.]))

        """
        hoomd.util.print_status_line()

        try:
            kx, ky, kz = k
        except:
            try:
                kx = ky = kz = k
            except:
                hoomd.context.msg.error('restrain.position.set_params: k must be a scalar or 3-item iterable\n')
                raise ValueError('k must be a scalar or 3-item iterable')
        for val in kx, ky, kz:
            try:
                val = float(val)
            except:
                hoomd.context.msg.error('restrain.position.set_params: k must be composed of scalars\n')
                raise ValueError('k must be composed of scalars')

        self.cpp_force.setForceConstant(float(kx), float(ky), float(kz))

    def set_reference_positions(self, ref_pos):
        R""" Set the reference positions for all particles.

        Args:
            ref_pos (ndarray): Reference positions for each particle. Array should be of size :math:`N \times 3`

        Examples::

            springs = azplugins.restrain.position(group=hoomd.group.all(), k=1.0)
            lattice = numpy.random.rand(N,3)
            springs.set_reference_positions(lattice)

        """
        hoomd.util.print_status_line()
        hoomd.util.quiet_status()
        # try to cast as ndarray
        try:
            ref_pos_ndarray = numpy.asarray(ref_pos)
        except:
            hoomd.context.msg.error('restrain.position.set_reference_positions: ref_pos cannot be cast to ndarray\n')
            raise ValueError('ref_pos cannot be cast to ndarray')
        # append each particle position
        for i in range(0, len(ref_pos_ndarray.shape)):
            self.set_position(i,ref_pos_ndarray[i,:])
        hoomd.util.unquiet_status()

    def set_position(self, i, ref_pos):
        R""" Set the reference position for a particular particle.

        Args:
            i (int): Index of the particle
            ref_pos (array_like): Reference position for the particle

        Examples::

            springs = azplugins.restrain.position(group=hoomd.group.all(), k=1.0)
            lattice = [1., 2., 3.]
            springs.set_reference_position(0,lattice)

        """
        hoomd.util.print_status_line()

        try:
            xyz = _hoomd.make_scalar4(ref_pos[0],ref_pos[1],ref_pos[2],0.0)
        except:
            hoomd.context.msg.error('restrain.orientation.set_positions: ref_pos must be a 3-item iterable composed of scalars\n')
            raise ValueError('ref_pos must be a 3-item iterable composed of scalars')
        self.cpp_force.setPosition(i,xyz)

    def update_coeffs(self):
        pass

class orientation(force._force):
    R""" Add a :math:`D_{\infty h}`-symmetric restraining potential based on initial (or specified) orientation

    Args:
        group Group of atoms to apply restraint to
        k (float) Force constant, isotropic
        name (str) (optional) name for potential

    The Hamiltonian is augmented with a potential field calculated based on the angle between the current and
    initial orientations of particles. This effectively allows particles to have nearly fixed orientation while
    still retaining their degrees of freedom. The potential has the following form:

    .. math::
        :nowrap:
        \begin{equation*}
        V(\mathbf{r}) = V(\theta) = k \sin^2(\theta)
        \end{equation*}

    The strength of the potential depends on a force constant :math:`k` so that the particle orientation can be
    restrained in any of the three coordinate directions. If :math:`\mathbf{k}` is very large, the particle orientation is
    essentially constrained. However, because the particles retain their degrees of freedom, shorter integration timesteps
    must be taken for large :math:`k` to maintain stability.

    Examples::

        azplugins.restrain.orientation(group=group.all(), k=1.0)
        field = azplugins.restrain.orientation(group=rotators, k=500.0)

    """
    def __init__(self, group, k, name=""):
        hoomd.util.print_status_line()

        # initialize the base class
        force._force.__init__(self,name)

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            cpp_class = _azplugins.OrientationRestraintCompute
        else:
            cpp_class = _azplugins.OrientationRestraintComputeGPU
        self.cpp_force = cpp_class(hoomd.context.current.system_definition,group.cpp_group)

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

        hoomd.util.quiet_status()
        self.set_params(k=k)
        hoomd.util.unquiet_status()

    def set_params(self, k):
        R""" Set the orientation restraint parameters.

        Args:
            k (float): Force constant, isotropic

        Examples::

            field = azplugins.restrain.orientation(group=hoomd.group.all(), k=0.0)
            field.set_params(k=1.)

        """
        hoomd.util.print_status_line()

        try:
            k = float(k)
        except:
            hoomd.context.msg.error('restrain.orientation.set_params: k must be a scalar\n')
            raise ValueError('k must be a scalar')

        self.cpp_force.setForceConstant(k)

    def set_reference_orientations(self, ref_orient):
        R""" Set the reference orientations for all particles.

        Args:
            ref_pos (ndarray): Reference orientations for each particle (in quaternion form). Array should be of size :math:`N \times 4`

        Examples::

            field = azplugins.restrain.orientation(group=hoomd.group.all(), k=1.0)
            directors = numpy.random.rand(N,4)
            field.set_reference_orientations(directors)

        """
        hoomd.util.print_status_line()
        hoomd.util.quiet_status()
        # try to cast as ndarray
        try:
            ref_orient_ndarray = numpy.asarray(ref_orient)
        except:
            hoomd.context.msg.error('restrain.orientation.set_reference_orientations: ref_orient cannot be cast to ndarray\n')
            raise ValueError('ref_orient cannot be cast to ndarray')
        # append each particle orientation
        for i in range(0, len(ref_orient_ndarray)):
            self.set_orientation(i,ref_orient_ndarray[i,:])
        hoomd.util.unquiet_status()

    def set_orientation(self, i, ref_orient):
        R""" Set the reference orientation for a particular particle.

        Args:
            i (int): Index of the particle
            ref_orient (array_like): Reference orientation for the particle

        Examples::

            field = azplugins.restrain.position(group=hoomd.group.all(), k=1.0)
            director = [1., 0., 0., 0.]
            field.set_reference_orientation(0,director)

        """
        hoomd.util.print_status_line()
        try:
            xyzw = _hoomd.make_scalar4(ref_orient[0],ref_orient[1],ref_orient[2],ref_orient[3])
        except:
            hoomd.context.msg.error('restrain.orientation.set_orientation: ref_orient must be a 4-item iterable composed of scalars\n')
            raise ValueError('ref_orient must be a 4-item iterable composed of scalars')
        self.cpp_force.setOrientation(i,xyzw)

    def update_coeffs(self):
        pass
