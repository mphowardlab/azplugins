# Import the C++ module.
from hoomd.azplugins import _azplugins

from hoomd.md.external import wall
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.data.typeparam import TypeParameter

class LJ93(wall.WallPotential):
    _ext_module = _azplugins
    _cpp_class_name = "WallPotentialLJ93"
    def __init__(self, walls):
        super().__init__(walls)

        params = TypeParameter(
            "params", "particle_types",
            TypeParameterDict(epsilon=float,
                                                        sigma=float,
                                                        r_cut=float,
                                                        r_extrap=0.0,
                                                        len_keys=1))
        self._add_typeparam(params)
