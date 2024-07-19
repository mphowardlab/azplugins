# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2024, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

import hoomd
from hoomd.azplugins import _azplugins
from hoomd.data.parameterdicts import ParameterDict
from hoomd.custom import Action
import pickle

class FlowField(Action):
    """Base class for flow fields."""
    def __init__(self):
        super().__init__()
        self._param_dict = ParameterDict()
        self._cpp_obj = None
        self._simulation = None

    def attach(self, simulation):
        """Attach the flow field to the simulation."""
        self._simulation = simulation
        self._attach_hook()

    def _attach(self, simulation):
        """Attach the flow field to the simulation."""
        self.attach(simulation)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_cpp_obj']
        del state['_simulation']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cpp_obj = None
        self._simulation = None

class ConstantFlow(FlowField):
    """Constant flow profile.

    Args:
        velocity (tuple): Flow field.

    Example:
        u = hoomd.azplugins.flow.ConstantFlow(velocity=(1,0,0))
    """
    def __init__(self, velocity):
        super().__init__()
        self.velocity = velocity
        param_dict = ParameterDict(velocity=(float, float, float))
        param_dict['velocity'] = velocity
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        """Initialize C++ object for constant flow."""
        self._cpp_obj = _azplugins.ConstantFlow(
            hoomd._hoomd.make_scalar3(
                self.velocity[0], self.velocity[1], self.velocity[2]
            )
        )

    def act(self, timestep):
        pass

    @property
    def velocity(self):
        return self._param_dict['velocity']

    @velocity.setter
    def velocity(self, value):
        self._param_dict['velocity'] = value
        if self._cpp_obj is not None:
            self._cpp_obj.velocity = value

class ParabolicFlow(FlowField):
    """Parabolic flow profile between parallel plates.

    Args:
        mean_velocity (float): Mean velocity.
        separation (float): Separation between parallel plates.

    Example:
        u = hoomd.azplugins.flow.ParabolicFlow(mean_velocity=2.0, separation=0.5)
    """
    def __init__(self, mean_velocity, separation):
        super().__init__()
        self.mean_velocity = mean_velocity
        self.separation = separation
        param_dict = ParameterDict(mean_velocity=float, separation=float)
        param_dict['mean_velocity'] = mean_velocity
        param_dict['separation'] = separation
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        """Initialize C++ object for parabolic flow."""
        self._cpp_obj = _azplugins.ParabolicFlow(
            self.mean_velocity, self.separation
        )

    def act(self, timestep):
        pass

    @property
    def mean_velocity(self):
        return self._param_dict['mean_velocity']

    @mean_velocity.setter
    def mean_velocity(self, value):
        self._param_dict['mean_velocity'] = value
        if self._cpp_obj is not None:
            self._cpp_obj.mean_velocity = value

    @property
    def separation(self):
        return self._param_dict['separation']

    @separation.setter
    def separation(self, value):
        self._param_dict['separation'] = value
        if self._cpp_obj is not None:
            self._cpp_obj.separation = value

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_cpp_obj']
        del state['_simulation']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cpp_obj = None
        self._simulation = None
