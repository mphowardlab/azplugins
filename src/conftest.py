# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2024, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""pytest configuration for azplugins."""

import pytest


@pytest.fixture(scope='session')
def bonded_two_particle_snapshot_factory(two_particle_snapshot_factory):
    """Fixture for a single bond."""

    def make_snapshot(bond_types=None, **kwargs):
        if bond_types is None:
            bond_types = ['A-A']
        snap = two_particle_snapshot_factory(**kwargs)
        if snap.communicator.rank == 0:
            snap.bonds.types = bond_types
            snap.bonds.N = 1
            snap.bonds.group[0] = [0, 1]
        return snap

    return make_snapshot
