# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2022, Auburn University
# This file is part of the azplugins project, released under the Modified BSD License.


import hoomd

from hoomd.azplugins import _azplugins
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import OnlyTypes, variant_preprocessing
from hoomd.md import force

class HarmonicBarrier(force.Force):

    _ext_module = _azplugins
#    _cpp_class_name = "PlanarHarmonicBarrier"

    def __init__(self, interface=None, geometry='film'):
        super().__init__()

        if geometry not in ('film', 'droplet'):
            raise ValueError(f"Unrecognized geometry: {geometry}")

        param_dict = ParameterDict(interface=hoomd.variant.Variant)
        param_dict["interface"] = interface
        self._param_dict.update(param_dict)
        self.geometry = geometry

        params = TypeParameter(
            "params",
            "particle_types",
            TypeParameterDict(
                k=float, offset=float, g=float, cutoff=float, len_keys=1
            ),
        )
        self._add_typeparam(params)

    def _attach_hook(self):
        sim = self._simulation

        if self.geometry not in ('film', 'droplet'):
            raise ValueError(f"Unrecognized geometry: {self.geometry}")

        if self.geometry == 'film':
            base_class_name = "PlanarHarmonicBarrier"
        elif self.geometry == 'droplet':
            base_class_name = "SphericalHarmonicBarrier"

        if isinstance(sim.device, hoomd.device.GPU):
            cpp_class_name = f"{base_class_name}GPU"
        else:
            cpp_class_name = base_class_name

        self._cpp_obj = getattr(_azplugins, cpp_class_name)(
            sim.state._cpp_sys_def,
            self.interface
        )

        super()._attach_hook()