# Copyright (c) 2018-2020, Michael P. Howard
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: mphoward / Everyone is free to add additional potentials

import hoomd
from hoomd.mpcd import _mpcd
from . import _azplugins

class _bounce_back(hoomd.md.integrate._integration_method):
    """ NVE integration with bounce-back rules.

    Args:
        group (:py:mod:`hoomd.group`): Group of particles on which to apply this method.

    :py:class:`_bounce_back` is a base class integration method. It must be used with
    :py:class:`hoomd.md.mode_standard`. Deriving classes implement the specific geometry and
    valid parameters for those geometries. Currently, there is no mechanism to share geometries
    between multiple instances of the same integration method.

    A :py:class:`hoomd.compute.thermo` is automatically specified and associated with *group*.

    """
    def __init__(self, group):
        # initialize base class
        hoomd.md.integrate._integration_method.__init__(self)

        # create the compute thermo
        hoomd.compute._get_unique_thermo(group=group)

        # store metadata
        self.group = group
        self.boundary = None
        self.metadata_fields = ['group','boundary']

    def _process_boundary(self, bc):
        """ Process boundary condition string into enum

        Args:
            bc (str): Boundary condition, either "no_slip" or "slip"

        Returns:
            A valid boundary condition enum.

        The enum interface is still fairly clunky for the user since the boundary
        condition is buried too deep in the package structure. This is a convenience
        method for interpreting.

        """
        if bc == "no_slip":
            return _mpcd.boundary.no_slip
        elif bc == "slip":
            return _mpcd.boundary.slip
        else:
            hoomd.context.msg.error("azplugins.integrate: boundary condition " + bc + " not recognized.\n")
            raise ValueError("Unrecognized streaming boundary condition")
            return None

class slit(_bounce_back):
    """ NVE integration with bounce-back rules in a slit channel.

    Args:
        group (:py:mod:`hoomd.group`): Group of particles on which to apply this method.
        H (float): channel half-width
        V (float): wall speed (default: 0)
        boundary : 'slip' or 'no_slip' boundary condition at wall (default: 'no_slip')

    This integration method applies to particles in *group* in the parallel-plate channel geometry. This geometry is
    characterized by a half-width *H*, so that the distance between the plates is :math:`2H`. Each plate can optionally
    be assigned a velocity *V*. The upper plate moves forward with velocity :math:`+V` while the bottom plate moves in
    the opposite direction at :math:`-V`. Hence, the shear rate for a linear velocity field between these plates would
    be :math:`\dot\gamma = V/H`.

    The boundary condition at the wall can be specified with *boundary* to either `'slip'` or `'no_slip'`. A no-slip
    condition is more common for solid boundaries. If the no-slip condition is combined with a body-force on the
    particles, a parabolic flow field with be generated. A shear flow profile is generated with a wall velocity and
    no-slip boundary conditions.

    Note:
        It may be necessary to add additional 'ghost' particles near the boundaries in order to correctly enforce the
        boundary conditions and to reduce density fluctuations near the wall.

    HOOMD uses a periodic simulation box, but the geometry imposes inherent non-periodic boundary conditions. You
    **must** ensure that the box is sufficiently large to enclose the geometry (i.e., :math:`L_z > 2H`). An error will
    be raised if the simulation box is not large enough to contain the geometry. Additionally, all particles must lie
    initially **inside** the geometry (i.e., all particles are between the plates). The particle configuration will be
    validated on the first call to :py:meth:`hoomd.run()`, and an error will be raised if this condition is not met.

    Warning:
        You must also ensure that particles do not self-interact through the periodic boundaries. This is usually
        achieved for simple pair potentials by padding the box size by the largest cutoff radius. Failure to do so
        may result in unphysical interactions.

    :py:class:`slit` is an integration method. It must be used with :py:class:`hoomd.md.mode_standard`.

    Warning:
        Bounce-back methods do not support anisotropic integration. If an anisotropic pair potential is specified,
        the torques will be ignored during the integration, and the particle orientations will not be updated. Do
        **not** use a bounce-back integrator with anisotropic particles or rigid bodies.

    A :py:class:`hoomd.compute.thermo` is automatically specified and associated with *group*.

    Examples::

        all = group.all()
        integrate.slit(group=all, H=5.0)
        integrate.slit(group=all, H=10.0, V=1.0)

    """
    def __init__(self, group, H, V=0.0, boundary="no_slip"):
        hoomd.util.print_status_line()

        # initialize base class
        _bounce_back.__init__(self,group)
        self.metadata_fields += ['H','V']

        # initialize the c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            cpp_class = _azplugins.BounceBackNVESlit
        else:
            cpp_class = _azplugins.BounceBackNVESlitGPU

        self.H = H
        self.V = V
        self.boundary = boundary

        bc = self._process_boundary(boundary)
        geom = _mpcd.SlitGeometry(H, V, bc)

        self.cpp_method = cpp_class(hoomd.context.current.system_definition, group.cpp_group, geom)
        self.cpp_method.validateGroup()

    def set_params(self, H=None, V=None, boundary=None):
        """ Set parameters for the slit geometry.

        Args:
            H (float): channel half-width
            V (float): wall speed (default: 0)
            boundary : 'slip' or 'no_slip' boundary condition at wall (default: 'no_slip')

        Examples::

            slit.set_params(H=8.)
            slit.set_params(V=2.0)
            slit.set_params(boundary='slip')
            slit.set_params(H=5, V=0., boundary='no_slip')

        """
        hoomd.util.print_status_line()

        if H is not None:
            self.H = H

        if V is not None:
            self.V = V

        if boundary is not None:
            self.boundary = boundary

        bc = self._process_boundary(self.boundary)
        self.cpp_method.geometry = _mpcd.SlitGeometry(self.H,self.V,bc)
