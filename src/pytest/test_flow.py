import collections

import hoomd
import numpy
import pytest
from hoomd.conftest import pickling_check
import hoomd.azplugins

def test_constant_flow_field(simulation_factory):
    # make the force
    U = hoomd.azplugins.flow.ConstantFlow(mean_velocity= (1,0,0))
    numpy.testing.assert_array_almost_equal(U.mean_velocity, (1, 0, 0))
    pickling_check(U)
    
    U.mean_velocity= (1,1,1)
    numpy.testing.assert_array_almost_equal(U.mean_velocity, (1, 1, 1))
    
    sim = simulation_factory()
    U._attach(sim)
    numpy.testing.assert_array_almost_equal(U.mean_velocity, (1, 1, 1))
    
    
def test_parabolic_flow_field(simulation_factory):

    U = hoomd.azplugins.flow.ParabolicFlow(mean_velocity= 4, separation= 10)
    assert U.mean_velocity == 4
    assert U.separation == 10
    pickling_check(U)
    
    U.mean_velocity = 10
    U.separation = 20
    numpy.testing.assert_array_almost_equal((U.mean_velocity, U.separation), (10, 20))
    
    sim = simulation_factory()
    U._attach(sim)
    numpy.testing.assert_array_almost_equal((U.mean_velocity, U.separation), (10, 20))
