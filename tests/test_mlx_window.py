"""Retained recording helpers and supported runner checks on actual Metal."""

import hashlib

import numpy as np
import pytest

from native_execution import require_metal
mx = require_metal('Metal is required', allow_module_level=True)

from mlx_backend import window
from mlx_backend.models import Connectome, Stimulus

pytestmark = pytest.mark.metal


def toy(counts=(400, -100, 200)):
    values = np.asarray(counts, dtype=np.int32)
    return Connectome(
        np.arange(3, dtype=np.int64),
        np.array([0, 0, 1], np.int32),
        np.array([1, 2, 2], np.int32),
        values,
        values.astype(np.float64) * 0.275,
    )


def stimulus(events, targets):
    return Stimulus(
        events,
        targets,
        (200.0,) * len(targets),
        tuple(range(events.shape[0])),
        0,
        0,
        hashlib.sha256(events.tobytes()).hexdigest(),
    )


def test_recording_compacts_active_words_and_detects_overflow():
    words = [
        mx.array([0, 3, 0, 8], dtype=mx.uint32),
        mx.array([4, 0, 0, 0], dtype=mx.uint32),
    ]
    windows, indices, masks = window.collect_words(words)
    actual = sorted(zip(windows.tolist(), indices.tolist(), masks.tolist()))
    assert actual == [(0, 1, 3), (0, 3, 8), (1, 0, 4)]
    with pytest.raises(OverflowError, match="capacity"):
        window.collect_words(words, capacity=2)


@pytest.mark.parametrize("capacity", [1, 5, 7])
def test_recording_splits_dense_words_across_rows_and_windows(monkeypatch, capacity):
    full_word = (1 << 18) - 1
    host = np.full((3, 7), full_word, dtype=np.uint32)
    host[0, 6] = (1 << 17) | 1
    host[1, 0] = 1 << 4
    host[1, 4] = 0
    host[2, 6] = 1 << 17
    calls = []
    kernel = window.recording_kernel()

    def recording(**kwargs):
        calls.append((kwargs["inputs"][0].size, kwargs["output_shapes"][1][0]))
        return kernel(**kwargs)

    monkeypatch.setattr(window, "recording_kernel", lambda: recording)
    windows, indices, masks = window.collect_words(
        [mx.array(row) for row in host], capacity=capacity, allow_split=True
    )
    actual = sorted(zip(windows.tolist(), indices.tolist(), masks.tolist()))
    expected = [
        (w, i, int(host[w, i]))
        for w in range(host.shape[0])
        for i in range(host.shape[1])
        if host[w, i]
    ]
    assert actual == expected
    assert calls[0] == (host.size, capacity)
    assert len(calls) == 1 + (host.size + capacity - 1) // capacity
    assert all(size <= capacity and slots <= capacity for size, slots in calls[1:])


@pytest.mark.parametrize("active", [False, True])
def test_recording_split_mode_keeps_sparse_and_empty_blocks_on_one_launch(
    monkeypatch, active
):
    host = np.zeros((3, 7), dtype=np.uint32)
    if active:
        host[0, 6], host[2, 0] = 1 << 17, 5
    calls = []
    kernel = window.recording_kernel()

    def recording(**kwargs):
        calls.append(kwargs["inputs"][0].size)
        return kernel(**kwargs)

    monkeypatch.setattr(window, "recording_kernel", lambda: recording)
    windows, indices, masks = window.collect_words(
        [mx.array(row) for row in host], capacity=2, allow_split=True
    )
    actual = sorted(zip(windows.tolist(), indices.tolist(), masks.tolist()))
    assert actual == ([(0, 6, 1 << 17), (2, 0, 5)] if active else [])
    assert calls == [host.size]
    assert windows.dtype == indices.dtype == np.int64
    assert masks.dtype == np.uint32


def test_production_default_window_matches_independent_cpu():
    from configured_reference import reference
    from mlx_backend.model import resolve_model
    from mlx_backend.runner import run

    events = np.zeros((2, 77, 2), np.uint8)
    events[:, [0, 3, 16, 17, 18, 34, 35], :] = 1
    events[1, 3] = 0
    s, c = stimulus(events, (0, 0)), toy()
    expected = reference(c, s, resolve_model(), ())
    actual = run(c, s, (), precision="0")
    for name in ("trials", "neurons", "steps"):
        np.testing.assert_array_equal(getattr(actual.spikes, name), getattr(expected.spikes, name))


def test_production_runner_rejects_retired_reference_selector():
    from mlx_backend.runner import run

    with pytest.raises(ValueError, match="Unknown MLX engine: bucketed"):
        run(toy(), stimulus(np.zeros((1, 0, 0), np.uint8), ()), (),
            precision="0", engine="bucketed")
