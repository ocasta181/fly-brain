"""Streamed events must be byte-identical, bounded and independent of reads."""

import hashlib
import json
from pathlib import Path

import mlx_engine
import numpy as np
import pytest
from native_execution import require_metal
from mlx_backend.models import Connectome
from mlx_backend.stimuli import CombinedEvents, SeededEvents, trial_events, uniforms


@pytest.mark.parametrize("channels", [0, 1, 2, 21])
def test_random_access_chunks_match_existing_canonical_generator(channels):
    rates = tuple(np.linspace(0, 10000, channels))
    source = SeededEvents(197, rates, (0, 7, 2), 20261004, 4)
    expected = np.stack([
        trial_events(rates, uniforms((20261004, 4, trial), (197, channels)))
        for trial in (0, 7, 2)
    ])
    for begin, end in [(180, 197), (0, 1), (17, 39), (0, 197), (17, 39), (197, 197)]:
        actual = source.chunk(begin, end)
        np.testing.assert_array_equal(actual, expected[:, begin:end])
        assert actual.flags.c_contiguous
    joined = np.concatenate([source.chunk(a, b) for a, b in [(0, 17), (17, 96), (96, 197)]], axis=1)
    np.testing.assert_array_equal(joined, expected)


@pytest.mark.parametrize("case", ["sugar", "p9"])
def test_seeded_source_reproduces_independent_saved_stimulus(case):
    directory = Path(__file__).parent / "data/mlx-replay" / case
    metadata = json.loads((directory / "stimulus.json").read_text())
    with np.load(directory / "stimulus.npz", allow_pickle=False) as saved:
        expected = saved["events"]
    source = SeededEvents(expected.shape[1], tuple(metadata["rates_hz"]),
                          tuple(metadata["trial_indices"]), metadata["seed"], metadata["generator_code"])
    pieces = [source.chunk(a, min(a + 113, source.shape[1])) for a in range(0, source.shape[1], 113)]
    actual = np.concatenate(pieces, axis=1)
    np.testing.assert_array_equal(actual, expected)
    assert hashlib.sha256(actual.tobytes()).hexdigest() == metadata["canonical_event_sha256"]


def test_largest_standard_event_shape_needs_only_requested_chunk():
    source = SeededEvents(10_000_000, (200.0,) * 21, tuple(range(32)), 17, 0)
    assert source.size == 6_720_000_000
    assert source.size > np.iinfo(np.int32).max
    tail = source.chunk(9_999_927, 10_000_000)
    assert tail.shape == (32, 73, 21)
    assert tail.nbytes == 49_056
    np.testing.assert_array_equal(tail, source.chunk(9_999_927, 10_000_000))


def test_combining_lazy_and_array_trials_preserves_order_and_rates(monkeypatch):
    ids = np.arange(3, dtype=np.int64)
    connectome = Connectome(ids, np.array([0], np.int32), np.array([1], np.int32),
                           np.array([1], np.int32), np.array([0.275]))
    spec = mlx_engine.experiment_spec({"key": "custom", "neu_exc": [2, 0, 2], "stim_rate": 200.0})
    eager, _ = mlx_engine.stimuli(connectome, spec, 137, 3, 33, stream=False)
    monkeypatch.setattr(mlx_engine, "MAX_EAGER_EVENT_BYTES", 0)
    streamed, _ = mlx_engine.stimuli(connectome, spec, 137, 3, 33)
    assert all(isinstance(item.events, SeededEvents) and item.sha256 is None for item in streamed)
    combined = mlx_engine.combine_trials(streamed)
    assert isinstance(combined.events, CombinedEvents)
    expected = np.concatenate([item.events for item in eager], axis=0)
    np.testing.assert_array_equal(combined.events.chunk(0, 137), expected)
    mixed = CombinedEvents((eager[0].events, streamed[1].events, eager[2].events))
    np.testing.assert_array_equal(mixed.chunk(7, 103), expected[:, 7:103])
    assert combined.trial_indices == (0, 1, 2)


def test_channels_and_trials_are_independent_with_expected_bernoulli_rates():
    # Fixed seed, declared 7-sigma marginal bounds, and a broad independence
    # bound make this a reproducible generator regression, not a cohort study.
    rates = (0.0, 100.0, 100.0, 200.0, 10000.0)
    source = SeededEvents(100_000, rates, (0, 1), 721, 4)
    events = source.chunk(0, 100_000)
    assert not events[:, :, 0].any()
    assert events[:, :, 4].all()
    for trial in range(2):
        for channel in (1, 2, 3):
            probability = rates[channel] * 0.0001
            expected = 100_000 * probability
            sigma = np.sqrt(expected * (1 - probability))
            assert abs(int(events[trial, :, channel].sum()) - expected) < 7 * sigma
    assert abs(np.corrcoef(events[0, :, 1], events[0, :, 2])[0, 1]) < 0.03
    assert abs(np.corrcoef(events[0, :, 1], events[1, :, 1])[0, 1]) < 0.03


@pytest.mark.parametrize("begin,end", [(-1, 2), (1, 11), (3, 2), (0.0, 2)])
def test_event_source_rejects_invalid_chunk_bounds(begin, end):
    with pytest.raises(ValueError, match="chunk"):
        SeededEvents(10, (200.0,), (0,), 1, 1).chunk(begin, end)


@pytest.mark.parametrize("duration,expected", [(0.0, 0), (0.00015, 2), (0.0002, 2)])
def test_adapter_uses_reference_partial_horizon_policy(duration, expected):
    assert mlx_engine.dimensions(duration, np.int64(2)) == expected


@pytest.mark.metal
@pytest.mark.parametrize("engine,record", [
    ("window", True), ("window", False), ("bucketed", True), ("bucketed", False),
])
def test_zero_horizon_preserves_empty_events_and_skips_execution(monkeypatch, engine, record):
    mx = require_metal('Metal is required')
    from mlx_backend import runner, configured
    if engine == "bucketed":
        from mlx_reference import runner
    from mlx_backend.models import Stimulus

    connectome = Connectome(np.arange(3), np.array([0], np.int32), np.array([1], np.int32),
                           np.array([1], np.int32), np.array([0.275]))

    class EmptyEvents(SeededEvents):
        def chunk(self, begin, end):
            pytest.fail("A zero-horizon run must not read events")

    source = EmptyEvents(0, (200.0,), (0, 1), 0, 0)
    stimulus = Stimulus(source, (0,), (200.0,), (0, 1), 0, 0, None)

    def unexpected_advance(*args, **kwargs):
        pytest.fail("A zero-horizon run must not execute a timestep")

    monkeypatch.setattr(configured, "advance", unexpected_advance)
    if engine == "bucketed":
        from dataclasses import replace
        prepare = runner.prepare
        monkeypatch.setattr(runner, "prepare", lambda *a, **kw: replace(
            prepare(*a, **kw), advance=unexpected_advance
        ))
    result = runner.run(connectome, stimulus, (), precision="0", engine=engine, record=record)
    assert result.spikes.steps.size == result.spikes.neurons.size == 0
    assert result.input_sha256 == (hashlib.sha256(b"").hexdigest(),) * 2
    assert result.timings["simulation_wall_s"] >= 0
