"""Graph invariants required by exact IDs and safe native integer propagation."""

import numpy as np
import pytest

from mlx32.model import Drive, make_graph


def test_large_ids_parallel_edge_order_and_owned_arrays():
    ids = np.array([2**59 + 1, 2**59 + 2, 2**59 + 3], dtype=np.int64)
    contacts = np.array([7, -3, 11, 5], dtype=np.int64)
    graph = make_graph(ids, [2, 0, 2, 0], [1, 2, 1, 2], contacts)
    np.testing.assert_array_equal(graph.neuron_ids, ids)
    np.testing.assert_array_equal(graph.row_ptr, [0, 2, 2, 4])
    np.testing.assert_array_equal(graph.destinations, [2, 2, 1, 1])
    np.testing.assert_array_equal(graph.contacts, [-3, 5, 7, 11])
    assert graph.max_incoming_abs == 18
    ids[:] = 0
    contacts[:] = 0
    assert graph.neuron_ids[0] == 2**59 + 1
    assert graph.contacts[0] == -3
    for array in (graph.neuron_ids, graph.row_ptr, graph.destinations, graph.contacts):
        assert not array.flags.writeable


def test_contact_bound_rejects_cancellation_overflow_and_fractional_contacts():
    limit = 2**31 - 1
    assert make_graph([1], [0], [0], [limit]).max_incoming_abs == limit
    for contacts in ([limit, 1], [limit, -1]):
        with pytest.raises(ValueError, match="safe int32"):
            make_graph([1], [0, 0], [0, 0], contacts)
    with pytest.raises(ValueError, match="without rounding"):
        make_graph([1], [0], [0], [0.5])
    with pytest.raises(ValueError, match="without float conversion"):
        make_graph([float(2**59)], [], [], [])


def test_drive_keeps_zero_rate_duplicate_channel_occurrences():
    targets = np.array([1, 1, 0, 1], dtype=np.int64)
    rates = np.array([100, 0, 200, 300], dtype=np.float64)
    drive = Drive(targets, rates, seed=19)
    targets[:] = 0
    rates[:] = 0
    np.testing.assert_array_equal(drive.targets, [1, 1, 0, 1])
    np.testing.assert_array_equal(drive.rates_hz, [100, 0, 200, 300])
    assert drive.seed == 19
    assert not drive.targets.flags.writeable
    assert not drive.rates_hz.flags.writeable
