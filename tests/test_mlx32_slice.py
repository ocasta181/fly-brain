"""Small, non-threshold-sensitive Brian controls and actual Metal witnesses."""

from dataclasses import asdict, dataclass
import json

import numpy as np
import pytest

from mlx32_reference import ReferenceModel, run_reference


TRIAL = 3
ROUND = 2


@dataclass(frozen=True)
class Case:
    name: str
    neurons: int
    steps: int
    edges: tuple[tuple[int, int, int], ...] = ()
    targets: tuple[int, ...] = ()
    rates: tuple[float, ...] = ()
    pulses: tuple[tuple[int, int], ...] = ()
    expected: tuple[tuple[int, int], ...] | None = ()  # (tick, neuron)
    silenced: tuple[int, ...] = ()
    generated: bool = False


CASES = (
    Case("zero_steps", 2, 0),
    Case("empty_input", 2, 5),
    Case("isolated", 1, 12, targets=(0,), rates=(100,),
         pulses=((0, 0), (4, 0), (8, 0)), expected=((1, 0), (5, 0), (9, 0))),
    Case("delayed_edge", 2, 24, edges=((0, 1, 10000),), targets=(0,),
         rates=(100,), pulses=((0, 0),), expected=((1, 0), (20, 1))),
    Case("mixed_sign_duplicate_edges", 4, 48,
         edges=((0, 2, 20000), (1, 2, -10000), (0, 2, 10000),
                (0, 3, -10000), (1, 3, 5000)),
         targets=(0, 1), rates=(100, 100), pulses=((0, 0), (0, 1)),
         expected=((1, 0), (1, 1), (20, 2))),
    Case("refractory_arrival_release", 3, 47,
         edges=((0, 2, 10000), (1, 2, 10000)), targets=(0, 1), rates=(100, 100),
         pulses=((0, 0), (4, 0), (22, 0), (23, 1)),
         expected=((1, 0), (5, 0), (20, 2), (23, 0), (24, 1), (43, 2))),
    Case("outgoing_only_silencing", 3, 55,
         edges=((0, 1, 10000), (1, 2, 10000)), targets=(0, 1), rates=(100, 100),
         pulses=((0, 0), (30, 1)), expected=((1, 0), (20, 1), (31, 1)),
         silenced=(1,)),
    Case("zero_rate_target_has_no_refractory", 2, 27,
         edges=((0, 1, 10000),), targets=(0, 1), rates=(100, 0),
         pulses=((0, 0), (3, 0)), expected=((1, 0), (4, 0), (20, 1), (23, 1))),
    Case("duplicate_overlapping_channels", 2, 5, targets=(0, 0, 1, 0),
         rates=(100, 100, 100, 100), pulses=((0, 2), (4, 0), (4, 1), (4, 3)),
         expected=((1, 1),)),
    Case("canonical_random_channel_positions", 2, 37, targets=(0, 0, 1),
         rates=(3000, 0, 1700), expected=None, generated=True),
)


def trial_events(events: np.ndarray) -> np.ndarray:
    events = np.asarray(events, dtype=np.int64).reshape(-1, 2)
    return np.column_stack(
        (np.full(len(events), TRIAL, dtype=np.int64), events[:, 1], events[:, 0])
    )


@pytest.mark.metal
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_curated_slice(case, native, record_property):
    api, run_native, _ = native
    specification = ReferenceModel()
    tape = np.zeros((case.steps, len(case.targets)), dtype=np.uint8)
    if case.generated:
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([17, ROUND, TRIAL])))
        probabilities = np.asarray(case.rates) * specification.dt_s
        tape[:] = rng.random(tape.shape) < probabilities
    else:
        for tick, channel in case.pulses:
            tape[tick, channel] = 1
    controls = [
        run_reference(
            case.neurons, case.edges, case.targets, tape, model=specification,
            dtype=dtype, silenced=case.silenced, trial=TRIAL,
        )
        for dtype in (np.float32, np.float64)
    ]
    reference = controls[0]
    record_property("brian_observations", json.dumps({
        "f32_bits": reference.raw_bits(),
        "f64_bits": controls[1].raw_bits(),
        "events_trial_neuron_tick": reference.events.tolist(),
        "monitor_phases": [reference.spike_phase, reference.state_phase],
    }))
    assert reference.spike_phase == ("thresholds", 1)
    assert reference.state_phase == ("end", 0)
    assert reference.v_trace.shape == (case.steps, case.neurons)
    np.testing.assert_array_equal(reference.events, controls[1].events)
    if case.expected is not None:
        np.testing.assert_array_equal(reference.events, trial_events(case.expected))
    if case.name == "delayed_edge":
        assert reference.g_trace[18, 1] == 0
        assert reference.g_trace[19, 1] > 2
    if case.name == "refractory_arrival_release":
        assert not reference.eligible_trace[41, 2]
        assert reference.eligible_trace[42, 2]
        assert reference.g_trace[41, 2] == 0
        assert reference.g_trace[42, 2] > 2
    if case.name == "duplicate_overlapping_channels":
        assert reference.v[0] > specification.v_rest_v + 2.5 * specification.input_v
    if case.name == "mixed_sign_duplicate_edges":
        assert reference.v[3] < specification.v_rest_v
        assert reference.g[3] < 0

    pre, post, contacts = (zip(*case.edges) if case.edges else ((), (), ()))
    graph = api.make_graph(np.arange(case.neurons, dtype=np.int64), pre, post, contacts)
    drive = api.Drive(case.targets, case.rates, seed=17)
    model = api.Model(**asdict(specification))
    options = dict(model=model, trial=TRIAL, round_idx=ROUND, silenced=case.silenced)
    if not case.generated:
        options["input_tape"] = tape
    baseline = run_native(graph, case.steps, drive, chunk_size=1, **options)
    record_property("native_observations", json.dumps({
        "f32_bits": {
            "v": baseline.v.view(np.uint32).tolist(),
            "g": baseline.g.view(np.uint32).tolist(),
        },
        "events_tick_neuron": baseline.spikes.tolist(),
    }))
    assert baseline.step == case.steps
    assert baseline.v.dtype == baseline.g.dtype == np.float32
    assert baseline.refractory.dtype == np.int32
    assert baseline.refractory.shape == (case.neurons,)
    assert baseline.delay_ring.dtype == bool
    assert np.issubdtype(baseline.counts.dtype, np.integer)
    assert baseline.spikes.dtype == np.int64
    assert baseline.spikes.ndim == 2 and baseline.spikes.shape[1] == 2
    assert isinstance(baseline.timings, dict)
    np.testing.assert_array_equal(trial_events(baseline.spikes), reference.events)
    np.testing.assert_array_equal(baseline.counts, reference.counts)
    # Brian's independent solver and per-edge writes need not share float32 bits.
    np.testing.assert_allclose(baseline.v, reference.v, rtol=2e-5, atol=2e-7)
    np.testing.assert_allclose(baseline.g, reference.g, rtol=2e-5, atol=2e-7)
    expected_ring = np.zeros((specification.delay_steps, case.neurons), dtype=bool)
    for _, neuron, tick in reference.events:
        if tick >= max(0, case.steps - specification.delay_steps):
            expected_ring[tick % specification.delay_steps, neuron] = True
    np.testing.assert_array_equal(baseline.delay_ring, expected_ring)

    batches = []
    variants = (
        run_native(graph, case.steps, drive, chunk_size=7, **options),
        run_native(graph, case.steps, drive, chunk_size=32, record=False, **options),
        run_native(graph, case.steps, drive, chunk_size=5,
                   spike_sink=lambda events: batches.append(events.copy()), **options),
    )
    for result in variants:
        assert result.step == case.steps
        for field in ("v", "g", "refractory", "delay_ring", "counts"):
            actual, expected = getattr(result, field), getattr(baseline, field)
            assert actual.dtype == expected.dtype
            assert actual.shape == expected.shape
            assert actual.tobytes() == expected.tobytes(), field
    np.testing.assert_array_equal(variants[0].spikes, baseline.spikes)
    assert variants[1].spikes.shape == variants[2].spikes.shape == (0, 2)
    streamed = np.concatenate(batches) if batches else np.empty((0, 2), dtype=np.int64)
    np.testing.assert_array_equal(streamed, baseline.spikes)


def test_empty_graph_snapshots(native):
    api, run_native, _ = native
    graph = api.make_graph([], [], [], [])
    result = run_native(graph, 5, api.Drive([], []))
    assert result.step == 5
    assert result.v.shape == result.g.shape == result.counts.shape == (0,)
    assert result.delay_ring.shape == (18, 0)
    assert result.spikes.shape == (0, 2)


@pytest.mark.parametrize("chunk_size", [1, 4])
def test_recording_overflow_is_explicit(native, chunk_size):
    from mlx32.engine import RecordOverflow

    api, run_native, _ = native
    tape = np.array([[1], [0], [1], [0]], dtype=np.uint8)
    graph = api.make_graph([1], [], [], [])
    with pytest.raises(RecordOverflow):
        run_native(graph, 4, api.Drive([0], [100]), chunk_size=chunk_size,
                   input_tape=tape, max_recorded_events=1)


def test_missing_metal_is_rejected(monkeypatch):
    import mlx.core as mx
    from mlx32.engine import run
    from mlx32.model import Drive, make_graph

    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="Metal GPU"):
        run(make_graph([1], [], [], []), 0, Drive([], []))
