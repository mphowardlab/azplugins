# Copyright (c) 2016, Panagiotopoulos Group, Princeton University
# This file is part of the azplugins project, released under the Modified BSD License.

# Maintainer: mphoward / Everyone is free to add additional potentials

import hoomd
from hoomd import _hoomd
import _azplugins

class types(hoomd.update._updater):
    def __init__(self, inside, outside, lo, hi, period=1, phase=0):
        hoomd.update._updater.__init__(self)

        if not hoomd.context.exec_conf.isCUDAEnabled():
            cpp_class = _azplugins.TypeUpdater
        else:
            cpp_class = _azplugins.TypeUpdaterGPU
        self.cpp_updater = cpp_class(hoomd.context.current.system_definition)
        self.setupUpdater(period, phase)

        self.metadata_fields = ['inside','outside','lo','hi']

        hoomd.util.quiet_status()
        self.set_params(inside, outside, lo, hi)
        hoomd.util.unquiet_status()

    def set_params(self, inside=None, outside=None, lo=None, hi=None):
        if inside is not None:
            self.inside = inside
            try:
                type_id = hoomd.context.current.system_definition.getParticleData().getTypeByName(inside)
            except RuntimeError:
                hoomd.context.msg.error('update.type: inside type ' + self.inside + ' not recognized\n')
                raise ValueError('update.type: inside type ' + self.inside + ' not recognized')
            self.cpp_updater.inside = type_id

        if outside is not None:
            self.outside = outside
            try:
                type_id = hoomd.context.current.system_definition.getParticleData().getTypeByName(outside)
            except RuntimeError:
                hoomd.context.msg.error('update.type: outside type ' + self.outside + ' not recognized\n')
                raise ValueError('update.type: outside type ' + self.outside + ' not recognized')
            self.cpp_updater.outside = type_id

        if self.inside == self.outside:
            hoomd.context.msg.error('update.type: inside type (' + self.inside + ') cannot be the same as outside type\n')
            raise ValueError('update.type: inside type (' + self.inside + ') cannot be the same as outside type')

        if lo is not None:
            self.lo = lo
            self.cpp_updater.lo = lo

        if hi is not None:
            self.hi = hi
            self.cpp_updater.hi = hi

        if self.lo > self.hi:
            hoomd.context.msg.error('update.type: lower z bound ' + str(self.lo) + ' > upper z bound ' + str(self.hi) + '.\n')
            raise ValueError('update.type: upper and lower bounds are inverted')
