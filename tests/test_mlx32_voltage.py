"""Range-safe coordinates, absolute exports and independent phase controls."""

from dataclasses import asdict, replace

import numpy as np
import pytest

from mlx32.engine import _voltage_origin
from mlx32.model import Model
from mlx32_reference import ReferenceModel, run_reference


def test_default_range_certificate_uses_rest():
    model = Model()
    assert _voltage_origin(model, 69_948, 1) == model.v_rest_v


@pytest.mark.parametrize("model,incoming,channels", (
    (Model(v_rest_v=-3e38, v_reset_v=-3e38, threshold_v=3e38), 0, 0),
    (Model(weight_v=float(2**120)), 1, 0),
    (Model(dt_s=1e-12, tau_m_s=1.0, tau_g_s=2.0), 0, 0),
    (Model(), 0, 2**23),
), ids=("opposite_extremes", "unsafe_conductance", "rounded_unit_decay", "large_channel_multiplicity"))
def test_uncertified_ranges_keep_absolute_coordinates(model, incoming, channels):
    assert _voltage_origin(model, incoming, channels) == 0.0


@pytest.mark.metal
@pytest.mark.parametrize("steps", (0, 1, 31))
def test_rest_snapshots_remain_absolute_float32(native, steps):
    api, run_native, _ = native
    model = api.Model(v_reset_v=-0.060)
    graph = api.make_graph([10, 11], [], [], [])
    result = run_native(graph, steps, api.Drive([], []), model=model)
    expected = np.full(2, model.v_rest_v, dtype=np.float32)
    assert result.v.dtype == np.float32
    assert result.v.tobytes() == expected.tobytes()


@pytest.mark.metal
@pytest.mark.parametrize("rest", (-3e38, -0.0))
def test_absolute_fallback_preserves_original_rounding(native, rest):
    api, run_native, _ = native
    model = api.Model(v_rest_v=rest, v_reset_v=rest, threshold_v=3e38)
    graph = api.make_graph([10], [], [], [])
    drive = api.Drive([], [])
    zero = run_native(graph, 0, drive, model=model)
    one = run_native(graph, 1, drive, model=model)
    initial = np.float32(rest)
    affine, coupling, decay, _ = model.coefficients()
    expected = np.float32(affine + np.float32(
        np.float32(np.float32(0) * coupling) + np.float32(initial * decay)
    ))
    assert zero.v.tobytes() == np.array([initial], dtype=np.float32).tobytes()
    assert one.v.tobytes() == np.array([expected], dtype=np.float32).tobytes()


@pytest.mark.metal
def test_relative_snapshots_and_execution_invariance(native):
    api, run_native, run_batch_native = native
    model = api.Model(v_reset_v=-0.060)
    original = asdict(model)
    edges = ((0, 2, -10000), (1, 2, 5000))
    targets = (0, 0, 1, 0)
    steps, trial = 43, 3
    tape = np.zeros((steps, len(targets)), dtype=np.uint8)
    tape[0, (0, 2)] = 1
    tape[4, 1] = 1
    tape[22, (0, 1, 3)] = 1
    tape[42, (0, 1, 3)] = 1
    pre, post, contacts = zip(*edges)
    graph = api.make_graph([10, 11, 12], pre, post, contacts)
    drive = api.Drive(targets, [100] * len(targets))
    options = dict(model=model, trial=trial, input_tape=tape)
    baseline = run_native(graph, steps, drive, chunk_size=1, **options)

    reference = run_reference(
        3, edges, targets, tape, model=ReferenceModel(**original), dtype=np.float32, trial=trial,
    )
    events = np.column_stack((np.full(len(baseline.spikes), trial),
                              baseline.spikes[:, 1], baseline.spikes[:, 0]))
    np.testing.assert_array_equal(events, reference.events)
    np.testing.assert_array_equal(baseline.counts, reference.counts)
    np.testing.assert_allclose(baseline.v, reference.v, rtol=2e-5, atol=2e-7)
    np.testing.assert_allclose(baseline.g, reference.g, rtol=2e-5, atol=2e-7)
    assert baseline.g[2] < 0

    shifted = replace(model, v_rest_v=0.0, v_reset_v=model.v_reset_v - model.v_rest_v,
                      threshold_v=model.threshold_v - model.v_rest_v)
    relative = run_native(graph, steps, drive, model=shifted, trial=trial, input_tape=tape)
    expected_v = (relative.v.astype(np.float64) + model.v_rest_v).astype(np.float32)
    assert baseline.v.dtype == np.float32
    assert baseline.v.tobytes() == expected_v.tobytes()
    assert baseline.g.tobytes() == relative.g.tobytes()
    assert asdict(model) == original

    streamed = []
    variants = (
        run_native(graph, steps, drive, chunk_size=7, **options),
        run_native(graph, steps, drive, chunk_size=32, record=False, **options),
        run_native(graph, steps, drive, chunk_size=5,
                   spike_sink=lambda events: streamed.append(events.copy()), **options),
    )
    for result in variants:
        for field in ("v", "g", "refractory", "delay_ring", "counts"):
            assert getattr(result, field).tobytes() == getattr(baseline, field).tobytes(), field
    np.testing.assert_array_equal(variants[0].spikes, baseline.spikes)
    assert variants[1].spikes.shape == variants[2].spikes.shape == (0, 2)
    np.testing.assert_array_equal(np.concatenate(streamed), baseline.spikes)

    batched = run_batch_native(
        graph, steps, drive, model=model, trials=(trial, 17), chunk_size=9,
        input_tape=np.repeat(tape[:, None, :], 2, axis=1),
    )
    for position in range(2):
        for field in ("v", "g", "refractory", "counts"):
            assert getattr(batched, field)[position].tobytes() == getattr(baseline, field).tobytes(), field
        assert batched.delay_ring[:, position].tobytes() == baseline.delay_ring.tobytes()
        np.testing.assert_array_equal(batched.spikes[batched.spikes[:, 1] == position][:, (0, 2)],
                                      baseline.spikes)
