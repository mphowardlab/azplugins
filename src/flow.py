# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2025, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""Flow fields."""

import hoomd
from hoomd.azplugins import _azplugins
from hoomd.data.parameterdicts import ParameterDict
from hoomd.operation import _HOOMDBaseObject


class FlowField(_HOOMDBaseObject):
    """Base flow field."""

    pass


# hoomd.azplugins.flow.constant
class ConstantFlow(FlowField):
    r"""Constant flow profile.

    Args:
        velocity (tuple): Flow field.

    This flow corresponds to a constant vector field, e.g., a constant
    backflow in bulk or a plug flow in a channel. The flow field is
    independent of the position it is evaluated at.

    Example::

        u = hoomd.azplugins.flow.ConstantFlow(velocity=(1, 0, 0))

    """

    def __init__(self, velocity):
        super().__init__()
        param_dict = ParameterDict(velocity=(float, float, float))
        param_dict["velocity"] = velocity
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        self._cpp_obj = _azplugins.ConstantFlow(
            hoomd._hoomd.make_scalar3(
                self.velocity[0], self.velocity[1], self.velocity[2]
            )
        )
        super()._attach_hook()


class ParabolicFlow(FlowField):
    r"""Parabolic flow profile between parallel plates.

    Args:
         mean_velocity (float): Mean velocity.
         separation (float): Separation between parallel plates defining the flow field.

    This flow field generates the parabolic flow profile in a slit geomtry:

    .. math::

        u_x(y) = \frac{3}{2}U \left[1 - \left(\frac{y}{H}\right)^2 \right]

    The flow is along *x* with the gradient in *y*. The ``separation`` between the
    two plates is :math:`2H`, and the channel is centered around :math:`y=0`.
    The ``mean_velocity`` is :math:`U`.

    Example::

        u = hoomd.azplugins.flow.ParabolicFlow(
            mean_velocity=2.0, separation=0.5
        )

    Note:
        Creating a flow profile does **not** imply anything about the simulation
        box boundaries. It is the responsibility of the user to construct
        appropriate bounding walls commensurate with the flow profile geometry.

    """

    def __init__(self, mean_velocity, separation):
        super().__init__()

        param_dict = ParameterDict(
            mean_velocity=float(mean_velocity), separation=float(separation)
        )

        self._param_dict.update(param_dict)

    def _attach_hook(self):
        self._cpp_obj = _azplugins.ParabolicFlow(self.mean_velocity, self.separation)
        super()._attach_hook()
