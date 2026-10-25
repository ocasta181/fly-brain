"""Fixed-model contract tests and exact independent C++/Metal comparisons."""

from __future__ import annotations

import builtins
import json
import platform
import sys
from dataclasses import FrozenInstanceError, asdict

import numpy as np
import pytest

from brian_cpp_reference import ReferenceModel, run_reference
from mlx_backend.model import Drive, Model, make_graph


STATE_FIELDS = ("v", "g", "lastspike", "not_refractory", "counts", "delay_ring")
TRIALS = (2**53 + 7, 3, 29, 101)
ROUND = 7
PARAMETERS = (
    pytest.param(ReferenceModel(), id="original-eon"),
    pytest.param(ReferenceModel(
        dt_s=0.00023, v_rest_v=-0.061, v_reset_v=-0.057, threshold_v=-0.043,
        tau_m_s=0.017, tau_g_s=0.003, refractory_steps=3, delay_steps=7,
    ), id="nondefault"),
    pytest.param(ReferenceModel(v_rest_v=0.0, v_reset_v=0.0, threshold_v=0.01),
                 id="zero-rest"),
)


def assert_exact(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    if actual.dtype == np.float64:
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    else:
        np.testing.assert_array_equal(actual, expected)


@pytest.fixture(scope="module")
def metal():
    if sys.platform != "darwin" or platform.machine() != "arm64":
        pytest.skip("Native fixed-model tests require Apple Silicon Metal")
    import mlx.core as mx
    from mlx_backend.engine import run_batch

    assert mx.metal.is_available(), "This Apple Silicon qualification requires actual Metal"
    launches = []
    original_kernel = mx.fast.metal_kernel

    def kernel(*args, **kwargs):
        compiled = original_kernel(*args, **kwargs)

        def dispatch(*arguments, **options):
            launches.append(kwargs.get("name", ""))
            return compiled(*arguments, **options)

        return dispatch

    def reject_cpu(*args, **kwargs):
        raise AssertionError("the fixed backend attempted Brian CPU timestepping")

    def run_native(*args, **kwargs):
        before = len(launches)
        with pytest.MonkeyPatch.context() as isolated:
            brian = sys.modules.get("brian2")
            if brian is not None:
                isolated.setattr(brian.Network, "run", reject_cpu)
                isolated.setattr(brian, "run", reject_cpu)
            result = run_batch(*args, **kwargs)
        if args[0].n_neurons and args[1]:
            assert "eon_vf64_state" in launches[before:]
        return result

    with pytest.MonkeyPatch.context() as isolated:
        isolated.setattr(mx.fast, "metal_kernel", kernel)
        yield run_native


def test_default_preserves_original_eon_unit_conversion_bits():
    model = Model()
    assert np.float64(model.v_rest_v).view(np.uint64) == 0xBFAA9FBE76C8B43A
    assert np.float64(model.v_reset_v).view(np.uint64) == 0xBFAA9FBE76C8B43A
    assert np.float64(-0.052).view(np.uint64) != np.float64(model.v_rest_v).view(np.uint64)
    assert asdict(model) == asdict(ReferenceModel())
    words = model.parameters()
    assert words.dtype == np.uint64
    assert words.shape == (11,)
    for offset, value in ((3, model.v_rest_v), (4, model.v_reset_v),
                          (5, model.threshold_v), (6, model.input_v), (7, model.dt_s)):
        assert words[offset] == np.float64(value).view(np.uint64)
    with pytest.raises(FrozenInstanceError):
        model.dt_s = 1.0


@pytest.mark.parametrize("name,value", [
    ("dt_s", 0), ("dt_s", -1), ("dt_s", float("nan")), ("dt_s", True),
    ("tau_m_s", 0), ("tau_g_s", -1), ("tau_m_s", 0.005),
    ("v_rest_v", float("inf")), ("v_reset_v", float("nan")),
    ("threshold_v", -0.1), ("input_v", -1), ("weight_v", -1),
    ("delay_steps", 0), ("delay_steps", 1.5), ("delay_steps", True),
    ("refractory_steps", -1), ("refractory_steps", 2.5),
])
def test_model_rejects_unsupported_parameters(name, value):
    with pytest.raises((TypeError, ValueError, OverflowError)):
        Model(**{name: value})


def test_graph_keeps_exact_ids_and_each_source_original_edge_position():
    ids = np.array([2**59 + 1, 2**59 + 2, -7], dtype=np.int64)
    contacts = np.array([7, -3, 11, 5], dtype=np.int64)
    graph = make_graph(ids, [2, 0, 2, 0], [1, 2, 1, 2], contacts)
    assert graph.n_neurons == 3 and graph.n_edges == 4
    for name, expected, dtype in (
        ("neuron_ids", ids.copy(), np.int64),
        ("edge_ptr", [0, 0, 2, 4], np.uint32),
        ("edge_pre", [2, 2, 0, 0], np.uint32),
        ("contacts", [7, 11, -3, 5], np.int64),
        ("edge_post", [1, 1, 2, 2], np.uint32),
        ("out_ptr", [0, 2, 2, 4], np.uint32),
        ("out_edge", [2, 3, 0, 1], np.uint32),
    ):
        array = getattr(graph, name)
        assert_exact(array, np.asarray(expected, dtype=dtype))
        assert not array.flags.writeable
    ids[:] = contacts[:] = 0
    assert graph.neuron_ids[0] == 2**59 + 1 and graph.contacts[0] == 7
    big = make_graph([1], [0, 0, 0], [0, 0, 0], [2**55, 1, -(2**55)])
    assert big.contacts.tolist() == [2**55, 1, -(2**55)]


@pytest.mark.parametrize("arguments", [
    ([1, 1], [], [], []), ([float(2**59)], [], [], []),
    ([2**63], [], [], []), ([1], [0], [], [1]),
    ([1], [1], [0], [1]), ([1], [-1], [0], [1]),
    ([1], [0.5], [0], [1]), ([1], [True], [0], [1]),
    ([1], [0], [0], [0.5]), ([1], [0], [0], [2**63]),
])
def test_graph_rejects_lossy_or_invalid_inputs(arguments):
    with pytest.raises((TypeError, ValueError, OverflowError)):
        make_graph(*arguments)


def test_drive_owns_duplicate_and_zero_rate_occurrences():
    targets = np.array([1, 1, 0, 1], dtype=np.int64)
    rates = np.array([100, 0, 200, 300], dtype=np.float64)
    drive = Drive(targets, rates, seed=2**53 + 1)
    targets[:] = rates[:] = 0
    assert_exact(drive.targets, np.array([1, 1, 0, 1], dtype=np.int32))
    assert_exact(drive.rates_hz, np.array([100, 0, 200, 300], dtype=np.float64))
    assert drive.seed == 2**53 + 1
    assert not drive.targets.flags.writeable and not drive.rates_hz.flags.writeable


@pytest.mark.parametrize("targets,rates,seed", [
    ([0], [], 0), ([0.5], [1], 0), ([-1], [1], 0),
    ([0], [-1], 0), ([0], [float("nan")], 0), ([0], [float("inf")], 0),
    ([0], [1], -1), ([0], [1], True), ([0], [1], 1.5),
])
def test_drive_rejects_invalid_occurrences_rates_and_seed(targets, rates, seed):
    with pytest.raises((TypeError, ValueError, OverflowError)):
        Drive(targets, rates, seed=seed)


@pytest.fixture(scope="module", params=PARAMETERS)
def exact_case(request, tmp_path_factory, metal):
    specification = request.param
    n, steps = 70, 85
    edges = [
        (1, 2, 10000), (3, 5, 10000), (4, 5, 10000),
        (6, 7, 10000), (7, 8, 10000), (12, 16, 2000),
        (13, 14, 1), (12, 14, 2**55), (12, 14, 1), (12, 14, -(2**55)),
    ]
    edges += [(12, 15, 1)] * 33 + [(13, 15, -1)] * 33
    edges += [(12, neuron, contacts) for neuron, contacts in zip(range(60, 70), range(5100, 5200, 10))]
    targets = (0, 1, 3, 4, 6, 7, 9, 9, 10, 9, 12, 13)
    rates = (100, 100, 100, 100, 100, 100, 100, 0, 0, 100, 100, 100)
    tape = np.zeros((steps, 4, len(targets)), dtype=np.uint8)
    for position in range(4):
        offset = position * 2
        for tick, channel in ((0, 0), (4, 0), (8, 0), (0, 1),
                              (0, 2), (4, 2), (22, 2), (23, 3),
                              (0, 4), (30, 5), (4, 6), (4, 7), (4, 9),
                              (0, 10), (1, 10), (0, 11), (84 - offset, 0)):
            tape[tick + offset if tick < 40 else tick, position, channel] = 1
        for tick in range(50, 50 + 4 * position, 4):
            tape[tick, position, 0] = 1
    silenced = (7,)
    graph = make_graph(np.arange(n, dtype=np.int64) + 2**54, *zip(*edges))
    drive = Drive(targets, rates, seed=17)
    model = Model(**asdict(specification))
    root = tmp_path_factory.mktemp("fixed-cpp") / "oracle"
    reference = run_reference(n, edges, targets, tape, output_directory=root,
                              model=specification, silenced=silenced)
    baseline = metal(graph, steps, drive, trials=TRIALS, model=model,
                     round_idx=ROUND, silenced=silenced, input_tape=tape,
                     chunk_size=7, trace=True)
    return graph, drive, model, tape, silenced, reference, baseline


def test_all_physical_state_bits_and_end_phase_traces_match_cpp(exact_case, record_property):
    graph, _, model, tape, _, reference, actual = exact_case
    assert actual.step == len(tape)
    for name in STATE_FIELDS + ("spikes", "v_trace", "g_trace", "eligible_trace"):
        assert_exact(getattr(actual, name), getattr(reference, name))
    assert actual.v.shape == actual.g.shape == (4, graph.n_neurons)
    assert actual.lastspike.dtype == np.float64 and actual.counts.dtype == np.int64
    assert actual.not_refractory.dtype == bool and actual.delay_ring.dtype == np.uint8
    assert actual.delay_ring.shape == (model.delay_steps, 4, graph.n_neurons)
    receipt = json.loads(reference.receipt_path.read_text())
    assert receipt["backend"] == "brian2_cpp_standalone"
    assert receipt["brian2"] == "2.8.0" and receipt["status"] == "complete"
    assert receipt["spike_phase"] == ["thresholds", 1]
    assert receipt["state_phase"] == ["end", 0]
    assert receipt["compiler_version"] and receipt["compiler_flags"]
    assert receipt["compiler_path"] and receipt["build_commands"]
    assert receipt["executable_sha256"]
    sources = receipt["generated_sources"]
    assert any("stateupdater" in name and name.endswith(".cpp") for name in sources)
    assert any("synapses" in name and name.endswith(".cpp") for name in sources)
    assert any(name.endswith("makefile") for name in sources)
    record_property("independent_cpp_receipt", str(reference.receipt_path))
    record_property("observed_raw_bits", json.dumps({
        name: getattr(actual, name).view(np.uint64).tolist()
        for name in ("v", "g", "lastspike")
    }))


def test_oracle_scenarios_exercise_delay_refractory_reset_and_order(exact_case):
    _, _, model, _, _, reference, _ = exact_case
    events = reference.spikes
    assert len(events) and len({tuple(row) for row in reference.counts}) == 4
    assert not reference.counts[:, 8].any()
    assert reference.counts[:, 7].min() >= 2
    assert not reference.counts[:, 10].any()
    assert not reference.g_trace[:model.delay_steps + 1, :, 2].any()
    assert np.all(reference.g_trace[model.delay_steps + 1, 0, 2] > 0)
    assert np.all(reference.g_trace[model.delay_steps + 1, 0, 14] == model.weight_v)
    assert reference.v_trace[4, 0, 9] > model.v_rest_v + 2.5 * model.input_v
    for tick, position, neuron in events:
        assert reference.v_trace[tick, position, neuron] == model.v_reset_v
        assert reference.g_trace[tick, position, neuron] == 0
        assert not reference.eligible_trace[tick, position, neuron]
    if model == Model():
        assert not reference.eligible_trace[41, 0, 5]
        assert reference.eligible_trace[42, 0, 5]
        assert reference.g_trace[41, 0, 5] == 0
        assert reference.g_trace[42, 0, 5] > 0


def test_distinct_batch_trials_equal_fresh_single_trials(exact_case, metal):
    graph, drive, model, tape, silenced, _, baseline = exact_case
    for position, trial in enumerate(TRIALS):
        single = metal(graph, len(tape), drive, trials=(trial,), model=model,
                       input_tape=tape[:, position:position + 1].astype(bool), silenced=silenced,
                       chunk_size=11, round_idx=ROUND)
        for name in STATE_FIELDS:
            expected = getattr(baseline, name)
            expected = expected[:, position:position + 1] if name == "delay_ring" else expected[position:position + 1]
            assert_exact(getattr(single, name), expected)
        expected = baseline.spikes[baseline.spikes[:, 1] == position].copy()
        expected[:, 1] = 0
        assert_exact(single.spikes, expected)


def test_threshold_equality_is_not_a_spike_using_independent_cpp_state(exact_case, metal):
    _, _, model, _, _, reference, _ = exact_case
    first_integration = model.delay_steps + 2
    threshold = float(reference.v_trace[first_integration, 0, 16])
    assert max(model.v_rest_v, model.v_reset_v) < threshold < model.threshold_v
    specification = asdict(model)
    specification["threshold_v"] = threshold
    threshold_model = Model(**specification)
    steps = first_integration + 2
    tape = np.zeros((steps, 1, 1), dtype=np.uint8)
    tape[:2] = 1
    result = metal(make_graph([1, 2], [0], [1], [2000]), steps, Drive([0], [100]),
                   model=threshold_model, input_tape=tape, chunk_size=7, trace=True)
    assert np.float64(result.v_trace[first_integration, 0, 1]).view(np.uint64) == np.float64(threshold).view(np.uint64)
    assert_exact(result.spikes, np.array([
        [1, 0, 0], [first_integration + 1, 0, 1],
    ], dtype=np.int64))


def test_chunks_recording_off_and_bounded_sink_preserve_state(exact_case, metal):
    graph, drive, model, tape, silenced, _, baseline = exact_case
    batches = []
    options = dict(trials=TRIALS, model=model, input_tape=tape,
                   silenced=silenced, round_idx=ROUND)
    recorded = metal(graph, len(tape), drive, chunk_size=32, **options)
    unrecorded = metal(graph, len(tape), drive, chunk_size=5, record=False, **options)
    streamed = metal(graph, len(tape), drive, chunk_size=4,
                     spike_sink=lambda batch: batches.append(batch.copy()),
                     max_recorded_events=1, **options)
    for result in (recorded, unrecorded, streamed):
        for name in STATE_FIELDS:
            assert_exact(getattr(result, name), getattr(baseline, name))
        assert result.v_trace is result.g_trace is result.eligible_trace is None
        assert all(isinstance(value, float) and np.isfinite(value) and value >= 0
                   for value in result.timings.values())
        assert set(("setup_compile_s", "compile_warmup_s", "runtime_s", "simulation_s",
                    "input_generation_staging_s", "collection_s", "snapshots_s")) <= result.timings.keys()
    assert_exact(recorded.spikes, baseline.spikes)
    assert unrecorded.spikes.shape == streamed.spikes.shape == (0, 3)
    assert all(batch.dtype == np.int64 and batch.shape[1] == 3
               and len(np.unique(batch[:, 0])) <= 4 for batch in batches)
    assert_exact(np.concatenate(batches), baseline.spikes)


def test_generated_input_uses_seed_round_trial_and_chunk_independent_identity(metal, monkeypatch):
    graph = make_graph([1, 2], [], [], [])
    drive = Drive([0, 0, 1, 0], [3000, 0, 1700, 2400], seed=17)
    steps = 37
    results = []
    for round_idx in (0, ROUND):
        tape = np.stack([
            np.random.Generator(np.random.PCG64(np.random.SeedSequence([17, round_idx, trial])))
            .random((steps, 4)) < drive.rates_hz * Model().dt_s
            for trial in TRIALS
        ], axis=1).astype(np.uint8)
        staged_shapes = []
        original_generator = np.random.Generator

        class TrackingGenerator:
            def __init__(self, *args, **kwargs):
                self.generator = original_generator(*args, **kwargs)

            def random(self, shape):
                staged_shapes.append(shape)
                return self.generator.random(shape)

        with monkeypatch.context() as isolated:
            isolated.setattr(np.random, "Generator", TrackingGenerator)
            generated = metal(graph, steps, drive, trials=TRIALS, round_idx=round_idx, chunk_size=5)
        assert staged_shapes and all(0 < shape[0] <= 5 and shape[1] == 4 for shape in staged_shapes)
        assert sum(shape[0] for shape in staged_shapes) == steps * len(TRIALS)
        supplied = metal(graph, steps, drive, trials=TRIALS, round_idx=round_idx,
                         chunk_size=11, input_tape=tape)
        for name in STATE_FIELDS + ("spikes",):
            assert_exact(getattr(generated, name), getattr(supplied, name))
        results.append(generated)
    assert results[0].spikes.tobytes() != results[1].spikes.tobytes()


def test_empty_zero_step_and_no_input_shapes_need_no_device():
    from mlx_backend.engine import run_batch

    for neurons, steps, delay in ((0, 5, 18), (1, 0, 18), (0, 5, 2**31 - 1)):
        graph = make_graph(np.arange(neurons, dtype=np.int64), [], [], [])
        result = run_batch(graph, steps, Drive([], []), trials=(3, 7), trace=True,
                           model=Model(delay_steps=delay))
        assert result.step == steps
        assert result.v.shape == result.g.shape == result.lastspike.shape == (2, neurons)
        assert result.counts.shape == (2, neurons)
        assert result.delay_ring.shape == (delay, 2, neurons)
        assert result.spikes.shape == (0, 3)
        assert result.v_trace.shape == result.g_trace.shape == (steps, 2, neurons)
        assert result.eligible_trace.shape == (steps, 2, neurons)
        assert np.all(result.v == Model().v_rest_v)
        assert np.all(result.lastspike == -1e4) and not result.counts.any()
        assert result.not_refractory.all() and not result.delay_ring.any()


def test_no_input_single_neuron_stays_at_original_physical_rest(metal):
    result = metal(make_graph([7], [], [], []), 35, Drive([], []), trace=True)
    assert_exact(result.v, np.full((1, 1), -52 * 0.001, dtype=np.float64))
    assert_exact(result.g, np.zeros((1, 1), dtype=np.float64))
    assert_exact(result.lastspike, np.full((1, 1), -1e4, dtype=np.float64))
    assert not result.counts.any() and result.spikes.shape == (0, 3)


def test_overflow_and_sink_error_leave_next_run_fresh(metal):
    graph = make_graph([7], [], [], [])
    drive = Drive([0], [100])
    tape = np.array([1, 0, 1, 0, 1, 0], dtype=np.uint8).reshape(6, 1, 1)
    with pytest.raises(OverflowError):
        metal(graph, 6, drive, input_tape=tape, chunk_size=2, max_recorded_events=1)

    def fail_sink(events):
        raise OSError("injected sink failure")

    with pytest.raises(OSError, match="injected sink failure"):
        metal(graph, 6, drive, input_tape=tape, chunk_size=2, spike_sink=fail_sink)
    result = metal(graph, 6, drive, input_tape=tape, chunk_size=2)
    assert_exact(result.spikes, np.array([[1, 0, 0], [3, 0, 0], [5, 0, 0]], dtype=np.int64))
    assert_exact(result.counts, np.array([[3]], dtype=np.int64))


@pytest.mark.parametrize("options", [
    {"steps": -1}, {"steps": True}, {"chunk_size": 0}, {"chunk_size": 1.5},
    {"trials": ()}, {"trials": (True,)}, {"round_idx": -1},
    {"silenced": (1,)}, {"record": 1}, {"trace": 1},
    {"input_tape": np.zeros((3, 1, 1), dtype=np.float64)},
    {"input_tape": np.full((3, 1, 1), 2, dtype=np.uint8)},
    {"input_tape": np.zeros((3, 2, 1), dtype=np.uint8)},
])
def test_invalid_run_inputs_fail_before_native_compilation(options, monkeypatch):
    from mlx_backend.engine import run_batch

    real_import = builtins.__import__

    def reject_mlx(name, *args, **kwargs):
        if name == "mlx" or name.startswith("mlx."):
            raise AssertionError("invalid inputs reached native allocation")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_mlx)
    options = dict(options)
    steps = options.pop("steps", 3)
    with pytest.raises((TypeError, ValueError, OverflowError)):
        run_batch(make_graph([7], [], [], []), steps, Drive([0], [100]), **options)


def test_missing_metal_has_no_cpu_fallback(metal, monkeypatch):
    import brian2 as brian
    import mlx.core as mx

    def reject_cpu(*args, **kwargs):
        raise AssertionError("the native backend attempted CPU timestepping")

    monkeypatch.setattr(brian.Network, "run", reject_cpu)
    monkeypatch.setattr(brian, "run", reject_cpu)
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="Metal"):
        metal(make_graph([1], [], [], []), 3, Drive([], []))


def test_dense_event_extraction_is_bounded_sorted_and_complete(metal, monkeypatch):
    from mlx_backend import engine

    graph = make_graph(np.arange(5, dtype=np.int64), [], [], [])
    drive = Drive(np.arange(5, dtype=np.int64), np.full(5, 10000.0))
    baseline = metal(graph, 9, drive, trials=(0, 1, 2, 3), chunk_size=4)
    monkeypatch.setattr(engine, "_EVENT_CAP", 3)
    chunks = []
    streamed = metal(graph, 9, drive, trials=(0, 1, 2, 3), chunk_size=4,
                     spike_sink=lambda rows: chunks.append(rows.copy()))
    assert len(baseline.spikes) == 80
    assert chunks and all(len(rows) <= 3 for rows in chunks)
    assert_exact(np.concatenate(chunks), baseline.spikes)
    for name in STATE_FIELDS:
        assert_exact(getattr(streamed, name), getattr(baseline, name))


@pytest.mark.parametrize("delay", [1, 7, 18, 31, 32, 33, 65])
def test_delayed_events_and_physical_ring_across_word_and_queue_boundaries(metal, delay):
    graph = make_graph([1, 2], [0], [1], [10000])
    drive = Drive([0], [100])
    model = Model(delay_steps=delay)
    steps = 4 * delay + 9
    tape = np.zeros((steps, 2, 1), dtype=np.uint8)
    tape[::7, 0, 0] = 1
    tape[::11, 1, 0] = 1
    options = dict(model=model, trials=(3, 7), input_tape=tape)
    baseline = metal(graph, steps, drive, chunk_size=1, trace=True, **options)
    for trial in range(2):
        assert baseline.spikes[(baseline.spikes[:, 1] == trial)
                               & (baseline.spikes[:, 2] == 1), 0][0] == delay + 2
    expected_ring = np.zeros((delay, 2, 2), dtype=np.uint8)
    for tick, trial, neuron in baseline.spikes:
        if tick >= steps - delay:
            expected_ring[tick % delay, trial, neuron] = 1
    assert_exact(baseline.delay_ring, expected_ring)
    for chunk in (3, 32):
        actual = metal(graph, steps, drive, chunk_size=chunk, **options)
        for name in STATE_FIELDS + ("spikes",):
            assert_exact(getattr(actual, name), getattr(baseline, name))
