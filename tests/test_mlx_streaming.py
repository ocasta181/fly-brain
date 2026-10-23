"""Bounded staging and continuation on the supported configured engine."""
from state_assertions import assert_state_bytes_equal as assert_state

import hashlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx_backend import configured, window
from mlx_backend.models import Connectome, Stimulus


class ArraySource:
    """Random-access source that records every requested bounded chunk."""

    dtype = np.dtype(np.uint8)
    ndim = 3

    def __init__(self, events):
        self.events = events
        self.shape = events.shape
        self.calls = []

    def chunk(self, begin, end):
        self.calls.append((begin, end))
        return np.ascontiguousarray(self.events[:, begin:end, :])


def stimulus(events, targets=(0,), trial_ids=(9, 4)):
    return Stimulus(events, targets, (200.0,) * len(targets), trial_ids, 0, 0, None)


def metadata(monkeypatch, events, targets=(0,), trial_ids=(9, 4), *, neurons=3, **kwargs):
    """Reach the real runner's first event allocation without allocating events."""
    class Program:
        collective_reset = False
        random_program = None

    model = SimpleNamespace(dt_ms=.1, poisson_weight_v=.06875)
    prepared = SimpleNamespace(neurons=neurons, targets=targets, model=model,
        amplitude_bits=np.full(len(targets), model.poisson_weight_v, np.float64).tobytes(),
        independent_neurons=False, compact_idle=False, program=Program(),
        window=18, scan_arrivals=False)
    options = dict(block_windows=2, chunk_steps=19, record=True)
    options.update(kwargs)
    allocations = []

    class ReachedAllocation(Exception):
        pass

    def allocate(shape, *args, **kw):
        allocations.append(shape)
        raise ReachedAllocation

    with monkeypatch.context() as patch:
        patch.setattr(configured, "initial_state", lambda *args, **kw: None)
        patch.setattr(configured, "_state_layout", lambda program: SimpleNamespace(groups=()))
        patch.setattr(mx, "reset_peak_memory", lambda: None)
        patch.setattr(mx, "zeros", allocate)
        with pytest.raises(ReachedAllocation):
            configured.run_prepared(prepared, stimulus(events, targets, trial_ids), **options)
    return allocations[0]


@pytest.mark.parametrize("lazy", [False, True])
def test_logical_inputs_larger_than_int32_need_only_bounded_indexing(monkeypatch, lazy):
    # The broadcast view has a >4 GiB logical size and one backing byte.
    events = np.broadcast_to(np.zeros((1, 1, 1), np.uint8), (2, 1_000_000, 2048))
    source = ArraySource(events) if lazy else events
    allocation = metadata(monkeypatch, source, targets=(0,) * 2048, chunk_steps=31)
    assert 0 < allocation[0] <= 2 * 31 * 2048
    assert events.size > np.iinfo(np.int32).max
    if lazy:
        assert source.calls == []
    chunk = window._event_chunk(source, 21, 25, 2, 2048)
    assert chunk.shape == (2, 4, 2048)
    assert chunk.flags.c_contiguous
    assert not chunk.any()


def test_staged_length_is_reduced_to_preserve_index_safety(monkeypatch):
    events = np.broadcast_to(np.zeros((1, 1, 1), np.uint8), (2, 1_000_000, 2048))
    source = ArraySource(events)
    allocation = metadata(monkeypatch, source, targets=(0,) * 2048, chunk_steps=1_000_000)
    staged_steps = allocation[0] // (2 * 2048)
    maximum = np.iinfo(np.int32).max // (2 * 2048)
    assert 0 <= maximum - staged_steps < 18
    assert staged_steps % 18 == 0
    assert allocation[0] <= np.iinfo(np.int32).max
    assert source.calls == []


@pytest.mark.parametrize("trial_ids,error", [
    ((4, 4), ValueError), ((-1, 4), ValueError), ((1.5, 4), TypeError),
    ((2**63, 4), ValueError), ((4,), ValueError),
])
def test_trial_identity_is_checked_before_reading_source(monkeypatch, trial_ids, error):
    source = ArraySource(np.zeros((2, 7, 1), np.uint8))
    with pytest.raises(error):
        metadata(monkeypatch, source, trial_ids=trial_ids)
    assert source.calls == []


@pytest.mark.parametrize("chunk_steps,error", [(0, ValueError), (-1, ValueError),
    (2.5, TypeError), (True, ValueError)])
def test_invalid_chunk_sizes_are_rejected(monkeypatch, chunk_steps, error):
    with pytest.raises(error):
        metadata(monkeypatch, np.zeros((2, 7, 1), np.uint8), chunk_steps=chunk_steps)


def test_source_trial_identities_must_match_the_stimulus(monkeypatch):
    source = ArraySource(np.zeros((2, 7, 1), np.uint8))
    source.trial_indices = (4, 9)
    with pytest.raises(ValueError, match="canonical stimulus"):
        metadata(monkeypatch, source)
    assert source.calls == []


@pytest.mark.parametrize("changes", [dict(dt_ms=0.2), dict(amplitudes_mv=(1.0,)),
    dict(amplitudes_mv=()), dict(amplitudes_mv=(68.75, 68.75))])
def test_prepared_execution_rejects_changed_timestep_or_amplitudes(monkeypatch, changes):
    events = np.zeros((2, 7, 1), np.uint8)
    changed = replace(stimulus(events), **changes)
    prepared = SimpleNamespace(targets=(0,), model=SimpleNamespace(dt_ms=.1, poisson_weight_v=.06875),
        amplitude_bits=np.array([.06875], np.float64).tobytes())
    with pytest.raises(ValueError, match="canonical stimulus|amplitudes"):
        configured.run_prepared(prepared, changed)


def test_source_timestep_must_match_stimulus(monkeypatch):
    source = ArraySource(np.zeros((2, 7, 1), np.uint8))
    source.dt_ms = 0.2
    with pytest.raises(ValueError, match="canonical stimulus"):
        metadata(monkeypatch, source)
    assert source.calls == []


def test_recording_and_event_indices_fail_before_allocation(monkeypatch):
    with pytest.raises(OverflowError, match="recording index"):
        metadata(monkeypatch, np.zeros((2, 7, 1), np.uint8), neurons=2**31)
    events = np.broadcast_to(np.zeros((1, 1, 1), np.uint8), (65536, 1, 2048))
    source = ArraySource(events)
    with pytest.raises(OverflowError, match="event window"):
        metadata(monkeypatch, source, targets=(0,) * 2048, trial_ids=tuple(range(65536)))
    assert source.calls == []


@pytest.mark.parametrize("defect", ["dtype", "shape", "layout", "binary", "array"])
def test_lazy_chunks_are_validated_before_device_staging(defect):
    events = np.zeros((2, 7, 1), np.uint8)

    class MalformedSource(ArraySource):
        def chunk(self, begin, end):
            result = super().chunk(begin, end)
            if defect == "dtype":
                return result.astype(np.float32)
            if defect == "shape":
                return result[:, :-1]
            if defect == "layout":
                return result[:, ::-1]
            if defect == "binary":
                result[0, 0, 0] = 2
                return result
            return result.tolist()

    with pytest.raises(ValueError, match="stimulus chunk"):
        window._event_chunk(MalformedSource(events), 0, 5, 2, 1)


def metal():
    if not mx.metal.is_available():
        pytest.skip("Metal is required")


def toy():
    counts = np.array([400, -100, 200], np.int32)
    return Connectome(np.arange(3, dtype=np.int64), np.array([0, 0, 1], np.int32),
        np.array([1, 2, 2], np.int32), counts, counts.astype(np.float64) * .275)




def assert_spikes(actual, expected):
    # CPU trials run in input order; exports group integer trial IDs in order.
    # Stable grouping preserves every independent CPU event's within-trial rank.
    order = np.argsort(expected.trials, kind="stable")
    for name in ("trials", "neurons", "steps"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name)[order])


@pytest.mark.metal
@pytest.mark.parametrize("trial_ids,expected", [((True, 4), (1, 4)),
    ((np.bool_(False), 4), (0, 4))])
def test_boolean_trial_ids_keep_configured_integer_canonicalization(trial_ids, expected):
    metal()
    import brian2 as b
    from mlx_backend.model import resolve_model
    model = resolve_model(dict(eqs="v:volt\ng:volt\nrfc:second", eq_th="True",
                               eq_rst="v=v_rst;g=0*mV", t_rfc=0*b.ms))
    s = stimulus(np.zeros((2, 2, 0), np.uint8), (), trial_ids)
    prepared = configured.prepare(toy(), s, model)
    result = configured.run_prepared(prepared, s)
    assert result.spikes.trials.dtype == np.int64
    np.testing.assert_array_equal(result.spikes.trials, np.repeat(expected, 6))


@pytest.mark.metal
@pytest.mark.parametrize("chunk_steps,record,targets", [
    (1, True, (0, 0)), (7, True, (0, 0)), (19, True, (0, 0)),
    (36, False, (0, 0)), (19, False, ()), (18, True, (0,)),
])
def test_array_and_lazy_chunks_preserve_spikes_state_and_input_hashes(monkeypatch, chunk_steps, record, targets):
    metal()
    from configured_reference import reference
    from mlx_backend.model import resolve_model
    steps = 77
    events = np.zeros((2, steps, len(targets)), np.uint8)
    if targets:
        events[:, [0, 3, 6, 16, 17, 18, 20, 34, 35, 37, 54, 55, 75], :] = 1
        events[1, 3] = 0
    model = resolve_model()
    s = stimulus(events, targets)
    prepared = configured.prepare(toy(), s, model)
    expected = reference(toy(), s, model)
    if not record:
        def unexpected_collection(*args, **kwargs):
            pytest.fail("Recording-disabled execution collected spikes")
        monkeypatch.setattr(configured, "_collect_unit_windows", unexpected_collection)
    source = ArraySource(events)
    actual, state = configured.run_prepared(prepared, replace(s, events=source),
        chunk_steps=chunk_steps, block_windows=2, record=record, return_state=True)
    assert_state(configured.host_state(state, prepared), expected.state)
    if record:
        assert_spikes(actual.spikes, expected.spikes)
    else:
        for name in ("trials", "neurons", "steps"):
            assert not getattr(actual.spikes, name).size
    effective = max(prepared.window, chunk_steps // prepared.window * prepared.window)
    assert source.calls == [(begin, min(begin + effective, steps)) for begin in range(0, steps, effective)]
    assert actual.input_sha256 == tuple(hashlib.sha256(trial.tobytes()).hexdigest() for trial in events)
    assert actual.timings["input_staging_s"] >= 0


@pytest.mark.metal
def test_direct_chunk_advance_preserves_pending_arrivals():
    metal()
    from mlx_backend.model import resolve_model
    events = np.zeros((2, 77, 1), np.uint8)
    events[:, [0, 3, 6, 16, 17, 18, 25, 26, 44, 45, 75], :] = 1
    events[1, 3] = 0
    prepared = configured.prepare(toy(), stimulus(events), resolve_model())
    whole = configured.initial_state(prepared, 2)
    device_events = mx.array(events.reshape(-1))
    while whole.step < 77:
        whole, _ = configured.advance(prepared, whole, device_events, 77,
            min(prepared.window - whole.step % prepared.window, 77 - whole.step))
    split = configured.initial_state(prepared, 2)
    for begin, end in ((0, 7), (7, 26), (26, 45), (45, 77)):
        chunk = mx.array(np.ascontiguousarray(events[:, begin:end]).reshape(-1))
        while split.step < end:
            split, _ = configured.advance(prepared, split, chunk, 77,
                min(prepared.window - split.step % prepared.window, end - split.step),
                event_begin=begin, event_steps=end - begin)
    assert_state(configured.host_state(split, prepared), configured.host_state(whole, prepared))
    for name in ("history", "due"):
        assert_state({key: np.asarray(value) for key, value in getattr(split, name).items()},
                     {key: np.asarray(value) for key, value in getattr(whole, name).items()})
    with pytest.raises(ValueError, match="outside the staged events"):
        configured.advance(prepared, replace(split, step=25), chunk, 77, 1,
                           event_begin=26, event_steps=19)


@pytest.mark.metal
@pytest.mark.parametrize("combined,record", [(False, True), (True, True), (True, False)])
def test_seeded_sources_match_original_canonical_streams(combined, record):
    metal()
    from configured_reference import reference
    from mlx_backend.model import resolve_model
    from mlx_backend.stimuli import CombinedEvents, SeededEvents, trial_events, uniforms
    steps, seed, generator_code = 77, 20261017, 6
    trial_ids, rates = (9, 4), (1000.0, 3000.0)
    canonical = np.stack([trial_events(rates, uniforms((seed, generator_code, trial), (steps, 2)))
                          for trial in trial_ids])
    source = (CombinedEvents(tuple(SeededEvents(steps, rates, (trial,), seed, generator_code)
                                  for trial in trial_ids)) if combined else
              SeededEvents(steps, rates, trial_ids, seed, generator_code))
    model = resolve_model()
    s = stimulus(canonical, (0, 0))
    prepared = configured.prepare(toy(), s, model)
    expected = reference(toy(), s, model)
    actual, state = configured.run_prepared(prepared, replace(s, events=source),
        chunk_steps=19, record=record, return_state=True)
    assert_state(configured.host_state(state, prepared), expected.state)
    if record:
        assert_spikes(actual.spikes, expected.spikes)
    else:
        for name in ("trials", "neurons", "steps"):
            assert not getattr(actual.spikes, name).size
    assert actual.input_sha256 == tuple(hashlib.sha256(trial.tobytes()).hexdigest() for trial in canonical)
