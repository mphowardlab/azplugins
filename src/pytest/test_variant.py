# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2025, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

import hoomd
import hoomd.azplugins
import numpy

import pytest


def interpolated_eval(data, t_lo, t_hi):
    data = numpy.asarray(data, dtype=float)
    return hoomd.azplugins.variant.VariantInterpolated(data=data, t_lo=t_lo, t_hi=t_hi)


variant_cases = [
    ([2.0, 4.0, 6.0, 8.0], 0, 300),
    ([1.0, -2.0, 5.0, 0.5, 3.0], 10, 90),
    ([-1.0, 1.0], 0, 100),
]


@pytest.mark.parametrize("variant_case", variant_cases)
def test_construction(variant_case):
    data, t_lo, t_hi = variant_case

    variant = interpolated_eval(data, t_lo, t_hi)
    assert variant.t_lo == pytest.approx(t_lo)
    assert variant.t_hi == pytest.approx(t_hi)

    variant.t_lo = 5
    variant.t_hi = 5

    assert variant.t_lo == pytest.approx(5)
    assert variant.t_hi == pytest.approx(5)


@pytest.mark.parametrize("variant_case", variant_cases)
def test_interpolated_values(variant_case):
    data, t_lo, t_hi = variant_case
    time = numpy.linspace(t_lo, t_hi, len(data))
    for timestep in range(int(t_lo), int(t_hi) + 1):
        expected = numpy.interp(timestep, time, data)
        assert interpolated_eval(data, t_lo, t_hi)(timestep) == pytest.approx(expected)


@pytest.mark.parametrize("variant_case", variant_cases)
def test_boundary_values(variant_case):
    data, t_lo, t_hi = variant_case
    variant = interpolated_eval(data, t_lo, t_hi)
    assert variant(t_lo) == pytest.approx(data[0])
    assert variant(t_hi) == pytest.approx(data[-1])


@pytest.mark.parametrize("variant_case", variant_cases)
def test_min_max(variant_case):
    data, t_lo, t_hi = variant_case
    variant = interpolated_eval(data, t_lo, t_hi)
    assert variant.min == pytest.approx(numpy.min(data))
    assert variant.max == pytest.approx(numpy.max(data))
