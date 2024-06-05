# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2024, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""Test flow fields."""

import hoomd
import numpy
from hoomd.conftest import pickling_check


def test_constant_flow_field(simulation_factory):
    """Test constant flow field."""
    # make the flow field
    U = hoomd.azplugins.flow.ConstantFlow(velocity=(1, 0, 0))
    numpy.testing.assert_array_almost_equal(U.velocity, (1, 0, 0))
    pickling_check(U)

    # set velocity
    U.velocity = (1, 2, 3)
    numpy.testing.assert_array_almost_equal(U.velocity, (1, 2, 3))
    pickling_check(U)

    # get and set velocity
    sim = simulation_factory()
    U._attach(sim)
    numpy.testing.assert_array_almost_equal(U.velocity, (1, 2, 3))
    pickling_check(U)


def test_parabolic_flow_field(simulation_factory):
    """Test parabolic flow field."""
    # make the flow field
    U = hoomd.azplugins.flow.ParabolicFlow(mean_velocity=4, separation=10)
    assert U.mean_velocity == 4
    assert U.separation == 10
    pickling_check(U)

    # set velocity and separation
    U.mean_velocity = 10
    U.separation = 20
    numpy.testing.assert_array_almost_equal((U.mean_velocity, U.separation), (10, 20))
    pickling_check(U)

    # get and set velocity and separation
    sim = simulation_factory()
    U._attach(sim)
    numpy.testing.assert_array_almost_equal((U.mean_velocity, U.separation), (10, 20))
    pickling_check(U)
