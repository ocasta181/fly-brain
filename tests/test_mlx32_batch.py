"""Independent Brian controls for trial batches and canonical event drains."""

from dataclasses import asdict
import json

import numpy as np
import pytest

from mlx32_reference import ReferenceModel, run_reference


TRIALS = (2**53 + 7, 3, 29)
ROUND = 2
FIELDS = ("v", "g", "refractory", "delay_ring", "counts")


def assert_exact(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.metal
@pytest.mark.parametrize("case", ["supplied_recurrence", "generated_channels"])
def test_batch_matches_independent_brian_and_scalar(case, native, record_property):
    api, run_native, run_batch_native = native
    specification = ReferenceModel()
    if case == "supplied_recurrence":
        steps, edges = 28, ((0, 1, 10000),)
        targets, rates = (0, 0, 0), (100, 0, 100)
        tape = np.zeros((steps, len(TRIALS), len(targets)), dtype=np.uint8)
        tape[0, 1, 0] = 1
        tape[26, 0, 0] = 1
        tape[26, 2, 2] = 1
        tape[27, 1, [0, 2]] = 1
    else:
        steps, edges = 37, ()
        targets, rates = (0, 0, 1, 0), (3000, 0, 1700, 2400)
        probabilities = np.asarray(rates) * specification.dt_s
        tape = np.stack([
            np.random.Generator(np.random.PCG64(np.random.SeedSequence([17, ROUND, trial])))
            .random((steps, len(targets))) < probabilities
            for trial in TRIALS
        ], axis=1).astype(np.uint8)
    controls = [
        [
            run_reference(2, edges, targets, tape[:, position], model=specification,
                          dtype=dtype, trial=trial)
            for dtype in (np.float32, np.float64)
        ]
        for position, trial in enumerate(TRIALS)
    ]
    event_parts = []
    for position, (reference, wide) in enumerate(controls):
        assert reference.spike_phase == wide.spike_phase == ("thresholds", 1)
        assert reference.state_phase == wide.state_phase == ("end", 0)
        np.testing.assert_array_equal(reference.events, wide.events)
        np.testing.assert_array_equal(reference.counts, wide.counts)
        event_parts.append(np.column_stack((
            reference.events[:, 2],
            np.full(len(reference.events), position, dtype=np.int64),
            reference.events[:, 1],
        )))
    expected_events = np.concatenate(event_parts)
    order = np.lexsort((expected_events[:, 2], expected_events[:, 1], expected_events[:, 0]))
    expected_events = expected_events[order]
    if case == "supplied_recurrence":
        np.testing.assert_array_equal(expected_events, np.array([
            [1, 1, 0], [20, 1, 1], [27, 0, 0], [27, 2, 0],
        ], dtype=np.int64))
        for position, (reference, _) in enumerate(controls):
            if position == 1:
                assert reference.g_trace[18, 1] == 0
                assert reference.g_trace[19, 1] > 2
            else:
                assert not reference.g_trace[:, 1].any()
        assert controls[1][0].v[0] > specification.v_rest_v + 1.5 * specification.input_v

    pre, post, contacts = (zip(*edges) if edges else ((), (), ()))
    graph = api.make_graph(np.array([100, 200], dtype=np.int64), pre, post, contacts)
    drive = api.Drive(targets, rates, seed=17)
    model = api.Model(**asdict(specification))
    options = dict(model=model, trials=TRIALS, round_idx=ROUND)
    if case == "supplied_recurrence":
        options["input_tape"] = tape
    baseline = run_batch_native(graph, steps, drive, chunk_size=1, **options)
    assert baseline.step == steps
    assert baseline.v.dtype == baseline.g.dtype == np.float32
    assert baseline.v.shape == baseline.g.shape == (len(TRIALS), 2)
    assert baseline.refractory.dtype == np.int32
    assert baseline.refractory.shape == baseline.counts.shape == (len(TRIALS), 2)
    assert np.issubdtype(baseline.counts.dtype, np.integer)
    assert baseline.delay_ring.dtype == bool
    assert baseline.delay_ring.shape == (specification.delay_steps, len(TRIALS), 2)
    assert baseline.spikes.dtype == np.int64
    assert baseline.spikes.ndim == 2 and baseline.spikes.shape[1] == 3
    assert isinstance(baseline.timings, dict)
    np.testing.assert_array_equal(baseline.spikes, expected_events)
    expected_ring = np.zeros_like(baseline.delay_ring)
    for position, (reference, _) in enumerate(controls):
        np.testing.assert_array_equal(baseline.counts[position], reference.counts)
        np.testing.assert_allclose(baseline.v[position], reference.v, rtol=2e-5, atol=2e-7)
        np.testing.assert_allclose(baseline.g[position], reference.g, rtol=2e-5, atol=2e-7)
        for _, neuron, tick in reference.events:
            if tick >= max(0, steps - specification.delay_steps):
                expected_ring[tick % specification.delay_steps, position, neuron] = True
        scalar_options = dict(model=model, trial=TRIALS[position], round_idx=ROUND)
        if case == "supplied_recurrence":
            scalar_options["input_tape"] = tape[:, position]
        scalar = run_native(graph, steps, drive, chunk_size=7, **scalar_options)
        assert scalar.step == steps
        for field in FIELDS:
            batched = getattr(baseline, field)
            expected = batched[:, position] if field == "delay_ring" else batched[position]
            assert_exact(getattr(scalar, field), expected)
        expected_scalar = baseline.spikes[baseline.spikes[:, 1] == position][:, (0, 2)]
        assert_exact(scalar.spikes, expected_scalar)
    np.testing.assert_array_equal(baseline.delay_ring, expected_ring)
    record_property("batch_observations", json.dumps({
        "trial_identifiers": TRIALS,
        "brian_f32_bits": [pair[0].raw_bits() for pair in controls],
        "brian_f64_bits": [pair[1].raw_bits() for pair in controls],
        "native_f32_bits": {
            "v": baseline.v.view(np.uint32).tolist(),
            "g": baseline.g.view(np.uint32).tolist(),
        },
        "events_tick_batch_neuron": expected_events.tolist(),
    }))

    batches = []
    recorded = run_batch_native(graph, steps, drive, chunk_size=7, **options)
    unrecorded = run_batch_native(graph, steps, drive, chunk_size=32, record=False, **options)
    streamed = run_batch_native(graph, steps, drive, chunk_size=5,
                                spike_sink=lambda events: batches.append(events.copy()), **options)
    for result in (recorded, unrecorded, streamed):
        assert result.step == steps
        for field in FIELDS:
            assert_exact(getattr(result, field), getattr(baseline, field))
    assert_exact(recorded.spikes, baseline.spikes)
    for result in (unrecorded, streamed):
        assert_exact(result.spikes, np.empty((0, 3), dtype=np.int64))
    drained = np.concatenate(batches) if batches else np.empty((0, 3), dtype=np.int64)
    assert_exact(drained, baseline.spikes)
    if case == "supplied_recurrence":
        assert np.any(drained[:, 0] == steps - 1)
    else:
        supplied = run_batch_native(graph, steps, drive, chunk_size=7,
                                   input_tape=tape, **options)
        for field in FIELDS + ("spikes",):
            assert_exact(getattr(supplied, field), getattr(baseline, field))


@pytest.mark.metal
@pytest.mark.parametrize("chunk_size", [1, 4])
@pytest.mark.parametrize("n_neurons", [1, 31, 32, 33, 257])
def test_batch_recording_cap_counts_all_trials(native, chunk_size, n_neurons):
    from mlx32.engine import RecordOverflow

    api, _, run_batch_native = native
    neurons = np.arange(n_neurons)
    graph = api.make_graph(100 + neurons, [], [], [])
    tape = np.zeros((4, 2, n_neurons), dtype=np.uint8)
    tape[0] = 1
    options = dict(trials=(101, 7), chunk_size=chunk_size, input_tape=tape)
    drive = api.Drive(neurons, np.full(n_neurons, 100))
    result = run_batch_native(graph, 4, drive, max_recorded_events=2 * n_neurons, **options)
    expected_events = np.array([[1, position, neuron] for position in range(2)
                                for neuron in range(n_neurons)], dtype=np.int64)
    np.testing.assert_array_equal(result.spikes, expected_events)
    np.testing.assert_array_equal(result.counts, np.ones((2, n_neurons), dtype=np.int64))
    with pytest.raises(RecordOverflow):
        run_batch_native(graph, 4, drive, max_recorded_events=2 * n_neurons - 1, **options)
