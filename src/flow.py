import hoomd
from hoomd import _hoomd
from hoomd.data.parameterdicts import ParameterDict
from hoomd.operation import _HOOMDBaseObject
from . import _azplugins



class FlowField(_HOOMDBaseObject):
    pass
    
class ConstantFlow(FlowField):
    def __init__(self, mean_velocity):
        super().__init__()
        param_dict = ParameterDict(mean_velocity=(float, float, float))
        param_dict["mean_velocity"] = mean_velocity
        self._param_dict.update(param_dict)
        
    def _attach_hook(self):
        self._cpp_obj = _azplugins.ConstantFlow(_hoomd.make_scalar3(self.mean_velocity[0], self.mean_velocity[1], self.mean_velocity[2]))
        super()._attach_hook()

class ParabolicFlow(FlowField):
    def __init__(self, mean_velocity, separation):
        super().__init__()

        param_dict = ParameterDict(
            mean_velocity=float(mean_velocity),separation=float(separation))
        
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        self._cpp_obj = _azplugins.ParabolicFlow(self.mean_velocity, self.separation)
        super()._attach_hook()
