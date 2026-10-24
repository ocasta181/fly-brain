"""Default-model anchoring and exact native preservation of the retained fallback."""

from dataclasses import asdict

import numpy as np
import pytest

import mlx32.engine as engine
from mlx32.model import Model
from mlx32_reference import run_reference


# Retained verbatim from f74151907aefa6cf328bb2fda4b81804e276547f.
_RETAINED_STATE_SOURCE = """
    uint i = thread_position_in_grid.x;
    if (i >= n_neurons[0]) return;
    uint trial = thread_position_in_grid.y;
    uint j = trial * n_neurons[0] + i;
    uint stim_base = trial * uint(stim_row_shape[1]);
    int r = max(rfc[j] - 1, 0);
    bool not_ref = r == 0;
    float vi = v[j];
    float gi = g[j];
    if (not_ref) {
        vi = coefficients[0] + (gi * coefficients[1] + vi * coefficients[2]);
        gi = gi * coefficients[3];
    }
    bool sp = not_ref && vi > parameters[0];
    if (not_ref) {
        gi = gi + float(contrib[j]) * parameters[1];
        // A stable channel CSR preserves repeated additions and channel order.
        for (int c = channel_ptr[i]; c < channel_ptr[i + 1]; ++c) {
            if (stim_row[stim_base + uint(channel_order[c])]) vi = vi + parameters[2];
        }
    }
    if (sp) {
        vi = parameters[3];
        gi = 0.0f;
        r = reload[i];
    }
    v_out[j] = vi;
    g_out[j] = gi;
    rfc_out[j] = r;
    counts_out[j] = counts[j] + (sp ? 1 : 0);
    spike_out[j] = sp ? 1 : 0;
"""


def assert_bits(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


def test_default_certificate_pairs_rows_and_uses_channel_bound():
    model = Model()
    original = model.coefficients()
    original[0] = 0
    for incoming in (0, 69_948, 2**31 - 1):
        coefficients, span = engine._anchor_coefficients(model, model.v_rest_v, incoming, 1)
        assert span == 256
        assert coefficients.dtype == np.float32
        assert coefficients.shape == (772,)
        assert_bits(coefficients[:4], original)
    coefficients, span = engine._anchor_coefficients(model, model.v_rest_v, 0, 2**23)
    assert span == 1
    assert_bits(coefficients[:4], original)
    assert not coefficients[4:].any()


@pytest.mark.parametrize("failure", ("exception", "nonfinite", "range"))
def test_table_preparation_failure_retains_original_coefficients(monkeypatch, failure):
    def table(model):
        if failure == "exception":
            raise FloatingPointError("table cast failed")
        return np.full((256, 3), np.nan if failure == "nonfinite" else 1, dtype=np.float32)

    monkeypatch.setattr(engine, "_anchor_table", table)
    model = Model()
    origin = engine._voltage_origin(model, 69_948, 1)
    coefficients, span = engine._anchor_coefficients(model, origin, 69_948, 1)
    expected = model.coefficients()
    expected[0] = 0
    assert origin == model.v_rest_v
    assert span == 1
    assert_bits(coefficients[:4], expected)
    assert not coefficients[4:].any()


@pytest.mark.metal
@pytest.mark.parametrize("case", ("absolute", "relative", "near_equal", "overflow", "failed_default_certificate"))
def test_fallback_matches_retained_native_arithmetic(native, monkeypatch, case):
    import mlx.core as mx

    model = {
        "absolute": Model(v_rest_v=-0.0, v_reset_v=-0.0, threshold_v=3e38),
        "relative": Model(v_reset_v=-0.060),
        "near_equal": Model(tau_m_s=1.0, tau_g_s=1.0000000001),
        "overflow": Model(weight_v=3e38),
        "failed_default_certificate": Model(),
    }[case]

    def table(model):
        assert case == "failed_default_certificate", "Nondefault models must skip table preparation"
        return np.ones((256, 3), dtype=np.float32)

    monkeypatch.setattr(engine, "_anchor_table", table)
    origin = engine._voltage_origin(model, 7, 2)
    host_coefficients, span = engine._anchor_coefficients(model, origin, 7, 2)
    assert span == 1
    shape = (1, 8)
    stream = mx.new_stream(mx.gpu)
    with mx.stream(stream):
        candidate = engine._kernels(mx)[1]
        retained = mx.fast.metal_kernel(
            name="eon32_retained_f741519",
            input_names=["v", "g", "rfc", "counts", "contrib", "stim_row",
                         "channel_ptr", "channel_order", "reload", "n_neurons",
                         "coefficients", "parameters"],
            output_names=["v_out", "g_out", "rfc_out", "counts_out", "spike_out"],
            source=_RETAINED_STATE_SOURCE, header="#pragma clang fp contract(off)\n",
            atomic_outputs=False,
        )
        v = np.array([0x80000000, 0, 0x7f800000, 0xff800000, 0x7fc12345,
                      0x7f7fffff, 0xff7fffff, 0x3c23d70a], dtype=np.uint32).view(np.float32)
        g = np.array([0x80000000, 0x80000000, 0, 0x7fc54321, 0x7f800000,
                      0x7f7fffff, 0xff7fffff, 0x80000000], dtype=np.uint32).view(np.float32)
        actual = [mx.array(v.reshape(shape)), mx.array(g.reshape(shape)),
                  mx.array([[0, 1, 2, 3, 0, 1, 4, 0]], dtype=mx.int32),
                  mx.zeros(shape, dtype=mx.int32), mx.zeros(shape, dtype=mx.int32)]
        expected = actual[:4]
        coefficients = mx.array(host_coefficients)
        parameters = mx.array([model.threshold_v - origin, model.weight_v, model.input_v,
                               model.v_reset_v - origin], dtype=mx.float32)
        for frame in ([1, 0, 1], [0, 0, 0], [1, 1, 0], [0, 1, 1]):
            shared = [mx.array([[0, 0, 7, -7, 0, 1, -1, 0]], dtype=mx.int32),
                      mx.array([frame], dtype=mx.uint8),
                      mx.array([0, 2, 2, 3, 3, 3, 3, 3, 3], dtype=mx.int32),
                      mx.array([0, 1, 2], dtype=mx.int32),
                      mx.full((8,), model.refractory_steps, dtype=mx.int32),
                      mx.array([8], dtype=mx.uint32), coefficients, parameters]
            expected_outputs = retained(
                inputs=[*expected, *shared], output_shapes=[shape] * 5,
                output_dtypes=[mx.float32, mx.float32, mx.int32, mx.int32, mx.uint8],
                grid=(8, 1, 1), threadgroup=(256, 1, 1), stream=stream,
            )
            actual_outputs = candidate(
                inputs=[*actual, *shared, mx.array([span], dtype=mx.int32)],
                output_shapes=[shape] * 6,
                output_dtypes=[mx.float32, mx.float32, mx.int32, mx.int32, mx.int32, mx.uint8],
                grid=(8, 1, 1), threadgroup=(256, 1, 1), stream=stream,
            )
            mx.eval(*actual_outputs, *expected_outputs)
            for observed, original in zip(actual_outputs[:4] + actual_outputs[5:], expected_outputs):
                assert_bits(observed, original)
            assert not np.asarray(actual_outputs[4]).any()
            actual, expected = actual_outputs[:5], expected_outputs[:4]


@pytest.mark.metal
def test_age_boundary_refractory_delivery_reset_and_decode(native):
    import mlx.core as mx

    model = Model()
    host_coefficients, span = engine._anchor_coefficients(model, model.v_rest_v, 1, 1)
    assert span == 256
    shape = (1, 6)
    stream = mx.new_stream(mx.gpu)
    with mx.stream(stream):
        state_kernel = engine._kernels(mx)[1]
        decode_kernel = engine._terminal_kernel(mx)
        initial = np.array([[-0.01, -0.01, -0.01, 0.1, -0.0, np.nan]], dtype=np.float32)
        conductance = np.array([[0.02, 0.02, 0.02, 0.02, -0.0, np.nan]], dtype=np.float32)
        state = [mx.array(initial), mx.array(conductance),
                 mx.array([[0, 2, 0, 0, 3, 3]], dtype=mx.int32),
                 mx.zeros(shape, dtype=mx.int32),
                 mx.array([[254, 255, 255, 255, 0, 0]], dtype=mx.int32)]
        count_n, coefficients = mx.array([6], dtype=mx.uint32), mx.array(host_coefficients)
        shared = [mx.array([[0, 0, 1, 1, 0, 0]], dtype=mx.int32),
                  mx.array([[1, 1]], dtype=mx.uint8),
                  mx.array([0, 0, 0, 1, 2, 2, 2], dtype=mx.int32),
                  mx.array([0, 1], dtype=mx.int32),
                  mx.full((6,), model.refractory_steps, dtype=mx.int32), count_n, coefficients,
                  mx.array([model.threshold_v - model.v_rest_v, model.weight_v,
                            model.input_v, 0], dtype=mx.float32), mx.array([span], dtype=mx.int32)]
        *state, spikes = state_kernel(
            inputs=[*state, *shared], output_shapes=[shape] * 6,
            output_dtypes=[mx.float32, mx.float32, mx.int32, mx.int32, mx.int32, mx.uint8],
            grid=(6, 1, 1), threadgroup=(256, 1, 1), stream=stream,
        )
        decoded = decode_kernel(
            inputs=[state[0], state[1], state[4], count_n, coefficients],
            output_shapes=[shape] * 2, output_dtypes=[mx.float32] * 2,
            grid=(6, 1, 1), threadgroup=(256, 1, 1), stream=stream,
        )
        mx.eval(*state, spikes, *decoded)
        np.testing.assert_array_equal(np.asarray(state[4]), [[255, 255, 0, 0, 0, 0]])
        np.testing.assert_array_equal(np.asarray(spikes), [[0, 0, 0, 1, 0, 0]])
        assert_bits(np.asarray(state[0])[0, :2], initial[0, :2])
        assert_bits(np.asarray(state[1])[0, :2], conductance[0, :2])
        for index, age in ((0, 255), (1, 255), (2, 256)):
            a, b, d = host_coefficients[4 + 3 * (age - 1):4 + 3 * age]
            voltage = np.float32(np.float32(conductance[0, index] * b) + np.float32(initial[0, index] * a))
            current = np.float32(conductance[0, index] * d)
            if index == 2:
                voltage = np.float32(voltage + np.float32(model.input_v))
                current = np.float32(current + np.float32(model.weight_v))
            assert_bits(np.asarray(decoded[0])[0, index], voltage)
            assert_bits(np.asarray(decoded[1])[0, index], current)
        assert np.asarray(state[0])[0, 3] == np.asarray(state[1])[0, 3] == 0
        assert np.asarray(state[2])[0, 3] == model.refractory_steps
        assert_bits(np.asarray(decoded[0])[0, 4:], initial[0, 4:])
        assert_bits(np.asarray(decoded[1])[0, 4:], conductance[0, 4:])
        *state, spikes = state_kernel(
            inputs=[*state, *shared], output_shapes=[shape] * 6,
            output_dtypes=[mx.float32, mx.float32, mx.int32, mx.int32, mx.int32, mx.uint8],
            grid=(6, 1, 1), threadgroup=(256, 1, 1), stream=stream,
        )
        mx.eval(*state, spikes)
        np.testing.assert_array_equal(np.asarray(state[4]), [[0, 0, 0, 0, 0, 0]])
        np.testing.assert_array_equal(np.asarray(spikes), [[0, 0, 1, 0, 0, 0]])


@pytest.mark.metal
def test_default_long_anchor_matches_brian_and_execution_invariance(native):
    api, run_native, run_batch_native = native
    model = api.Model()
    original = asdict(model)
    steps, edges, targets = 277, ((0, 1, -10), (0, 2, 10000)), (0, 0)
    tape = np.zeros((steps, len(targets)), dtype=np.uint8)
    tape[0, 0] = tape[275, 1] = 1
    reference = run_reference(3, edges, targets, tape, dtype=np.float64)
    graph = api.make_graph([10, 11, 12], *zip(*edges))
    drive = api.Drive(targets, [100, 100])
    options = dict(model=model, input_tape=tape)
    baseline = run_native(graph, steps, drive, chunk_size=7, **options)
    np.testing.assert_array_equal(baseline.spikes, [[1, 0], [20, 2], [276, 0]])
    np.testing.assert_array_equal(baseline.spikes, reference.events[:, (2, 1)])
    np.testing.assert_array_equal(baseline.counts, reference.counts)
    np.testing.assert_allclose(baseline.v, reference.v, rtol=2e-5, atol=2e-7)
    np.testing.assert_allclose(baseline.g, reference.g, rtol=2e-5, atol=2e-7)
    fields = ("v", "g", "refractory", "delay_ring", "counts")
    streamed = []
    variants = (run_native(graph, steps, drive, chunk_size=32, **options),
                run_native(graph, steps, drive, chunk_size=17, record=False, **options),
                run_native(graph, steps, drive, chunk_size=9,
                           spike_sink=lambda events: streamed.append(events.copy()), **options))
    for result in variants:
        for field in fields:
            assert_bits(getattr(result, field), getattr(baseline, field))
    assert_bits(variants[0].spikes, baseline.spikes)
    assert variants[1].spikes.shape == variants[2].spikes.shape == (0, 2)
    assert_bits(np.concatenate(streamed), baseline.spikes)
    batched = run_batch_native(graph, steps, drive, model=model, trials=(0, 17), chunk_size=31,
                               input_tape=np.repeat(tape[:, None, :], 2, axis=1))
    for position in range(2):
        for field in fields:
            value = getattr(batched, field)
            assert_bits(value[:, position] if field == "delay_ring" else value[position],
                        getattr(baseline, field))
        assert_bits(batched.spikes[batched.spikes[:, 1] == position][:, (0, 2)], baseline.spikes)
    assert asdict(model) == original
