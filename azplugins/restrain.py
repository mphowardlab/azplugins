# Copyright (c) 2016-2017, Panagiotopoulos Group, Princeton University
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: wes_reinhart

import hoomd
from hoomd import _hoomd
from hoomd.md import _md
from hoomd.md import force
import _azplugins
import numpy

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
        hoomd.util.print_status_line();

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
        hoomd.util.print_status_line();

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
        hoomd.util.print_status_line();
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
        hoomd.util.print_status_line();

        try:
            xyz = _hoomd.make_scalar4(ref_pos[0],ref_pos[1],ref_pos[2],0.0);
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
        hoomd.util.print_status_line();

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
        hoomd.util.print_status_line();

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
        hoomd.util.print_status_line();
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
        hoomd.util.print_status_line();
        try:
            xyzw = _hoomd.make_scalar4(ref_orient[0],ref_orient[1],ref_orient[2],ref_orient[3]);
        except:
            hoomd.context.msg.error('restrain.orientation.set_orientation: ref_orient must be a 4-item iterable composed of scalars\n')
            raise ValueError('ref_orient must be a 4-item iterable composed of scalars')
        self.cpp_force.setOrientation(i,xyzw)

    def update_coeffs(self):
        pass
