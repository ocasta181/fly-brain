"""Default-model anchoring and exact native span1 state traces."""

from dataclasses import asdict
from hashlib import sha256

import numpy as np
import pytest

import mlx32.engine as engine
from mlx32.model import Model
from mlx32_reference import run_reference


# Per-frame raw v, g, refractory, counts and spikes, in that order.
# Fixed traces preserve exceptional-value checks without a second state kernel.
_SPAN1_TRACE_SHA256 = {
    "absolute": (
        "6c0c21d15ce6436563f58d44a9e399e9e9355d44132c009f1857a5b85b8b06c4",
        "5f273fea442ed537db988396d380a49908cf83da7880e601794c87e2e112a1a8",
        "238e442671d300de61312ec79cbe29194f4168485bd61c800853eb94cd63dd85",
        "9d5d6915939f56ace56337a8303f031bab0482f1aeedaffed2ba1f829eaaa963",
    ),
    "relative": (
        "f771cbfabaf5d09b7a5fae6c12a96816ce670eda168b0a53175fb7d118a6e8f1",
        "f98be11af90bdf79ab0cadbbea815804afed8944ffa84b424313ba25e3c771df",
        "ee5b8f56ba8a2d4af3a2c5017dea835b00954fc65ac60ec87b948538f7357988",
        "e04f82bccf983c6a7a588d31301864f3e84cb7df04bd9afaa678de968d982932",
    ),
    "near_equal": (
        "629bfa613e6884d3a7baa40cf1658315758e12c76ac759008679d36e846af3c7",
        "a4bc060b6daf5c15a8a6bef87384b521bef32b7490acafd6d9de2a0b8cb93253",
        "683aedded99d0a36213a83c499ce8105aa0480f57436b2705678af50a2759498",
        "1f5889ae39ea1a15f56cc1f4759587677c90cc9ce69c8e6e9ea48cdd03ff764e",
    ),
    "overflow": (
        "629bfa613e6884d3a7baa40cf1658315758e12c76ac759008679d36e846af3c7",
        "a4bc060b6daf5c15a8a6bef87384b521bef32b7490acafd6d9de2a0b8cb93253",
        "683aedded99d0a36213a83c499ce8105aa0480f57436b2705678af50a2759498",
        "81232e0513d1542d9edc1584210a586a35a95928a39bd736cc1451236d2ab869",
    ),
    "failed_default_certificate": (
        "bf9bf1193ce696c88bff04d9c4dd8cfe980460bcea9dac476ad02b5f4fb28fdb",
        "e72e0c14c25f93f9a627f6310ce5edbe274e52f4f9bd9aae7437e3b7eee292c9",
        "c9423d1743851d184e2431277bb542784c7d810520f3b1d227504ee390c7e714",
        "51f09c3d155f39f1d9840b826494435d404461c050c478a0b71171a8bd1a5bdf",
    ),
}


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
@pytest.mark.parametrize("case", _SPAN1_TRACE_SHA256)
def test_span1_matches_exact_native_traces(native, monkeypatch, case):
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
        state_kernel = engine._kernels(mx)[1]
        v = np.array([0x80000000, 0, 0x7f800000, 0xff800000, 0x7fc12345,
                      0x7f7fffff, 0xff7fffff, 0x3c23d70a], dtype=np.uint32).view(np.float32)
        g = np.array([0x80000000, 0x80000000, 0, 0x7fc54321, 0x7f800000,
                      0x7f7fffff, 0xff7fffff, 0x80000000], dtype=np.uint32).view(np.float32)
        initial = [mx.array(v.reshape(shape)), mx.array(g.reshape(shape)),
                   mx.array([[0, 1, 2, 3, 0, 1, 4, 0]], dtype=mx.int32),
                   mx.zeros(shape, dtype=mx.int32), mx.zeros(shape, dtype=mx.int32)]
        actual = initial
        coefficients = mx.array(host_coefficients)
        parameters = mx.array([model.threshold_v - origin, model.weight_v, model.input_v,
                               model.v_reset_v - origin], dtype=mx.float32)
        frames = ([1, 0, 1], [0, 0, 0], [1, 1, 0], [0, 1, 1])
        contrib = mx.array([[[0, 0, 7, -7, 0, 1, -1, 0]]], dtype=mx.int32)
        shared = [mx.array([0, 2, 2, 3, 3, 3, 3, 3, 3], dtype=mx.int32),
                  mx.array([0, 1, 2], dtype=mx.int32),
                  mx.full((8,), model.refractory_steps, dtype=mx.int32),
                  mx.array([8], dtype=mx.uint32), coefficients, parameters,
                  mx.array([span], dtype=mx.int32)]
        history = []
        for frame, expected_sha256 in zip(frames, _SPAN1_TRACE_SHA256[case], strict=True):
            actual_outputs = state_kernel(
                inputs=[*actual, contrib, mx.array([[frame]], dtype=mx.uint8), *shared],
                output_shapes=[shape] * 5 + [(1, *shape)],
                output_dtypes=[mx.float32, mx.float32, mx.int32, mx.int32, mx.int32, mx.uint8],
                grid=(8, 1, 1), threadgroup=(256, 1, 1), stream=stream,
            )
            mx.eval(*actual_outputs)
            arrays = tuple(np.asarray(value) for value in actual_outputs)
            dtypes = (np.float32, np.float32, np.int32, np.int32, np.int32, np.uint8)
            for index, (value, dtype) in enumerate(zip(arrays, dtypes, strict=True)):
                assert value.dtype == np.dtype(dtype)
                assert value.shape == (shape if index < 5 else (1, *shape))
            raw = b"".join(value.tobytes() for value in arrays[:4] + (arrays[5][0],))
            assert sha256(raw).hexdigest() == expected_sha256
            assert not arrays[4].any()
            history.append(arrays[5])
            actual = actual_outputs[:5]
        blocked = state_kernel(
            inputs=[*initial, mx.concatenate([contrib] * len(frames), axis=0),
                    mx.array([[frame] for frame in frames], dtype=mx.uint8), *shared],
            output_shapes=[shape] * 5 + [(len(frames), *shape)],
            output_dtypes=[mx.float32, mx.float32, mx.int32, mx.int32, mx.int32, mx.uint8],
            grid=(8, 1, 1), threadgroup=(256, 1, 1), stream=stream,
        )
        mx.eval(*blocked)
        for value, expected in zip(blocked[:5], arrays[:5], strict=True):
            assert_bits(value, expected)
        assert_bits(blocked[5], np.concatenate(history, axis=0))


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
        initial_state = state
        count_n, coefficients = mx.array([6], dtype=mx.uint32), mx.array(host_coefficients)
        shared = [mx.array([[[0, 0, 1, 1, 0, 0]]], dtype=mx.int32),
                  mx.array([[[1, 1]]], dtype=mx.uint8),
                  mx.array([0, 0, 0, 1, 2, 2, 2], dtype=mx.int32),
                  mx.array([0, 1], dtype=mx.int32),
                  mx.full((6,), model.refractory_steps, dtype=mx.int32), count_n, coefficients,
                  mx.array([model.threshold_v - model.v_rest_v, model.weight_v,
                            model.input_v, 0], dtype=mx.float32), mx.array([span], dtype=mx.int32)]
        *state, first_history = state_kernel(
            inputs=[*state, *shared], output_shapes=[shape] * 5 + [(1, *shape)],
            output_dtypes=[mx.float32, mx.float32, mx.int32, mx.int32, mx.int32, mx.uint8],
            grid=(6, 1, 1), threadgroup=(256, 1, 1), stream=stream,
        )
        spikes = first_history[0]
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
        *state, second_history = state_kernel(
            inputs=[*state, *shared], output_shapes=[shape] * 5 + [(1, *shape)],
            output_dtypes=[mx.float32, mx.float32, mx.int32, mx.int32, mx.int32, mx.uint8],
            grid=(6, 1, 1), threadgroup=(256, 1, 1), stream=stream,
        )
        spikes = second_history[0]
        mx.eval(*state, spikes)
        np.testing.assert_array_equal(np.asarray(state[4]), [[0, 0, 0, 0, 0, 0]])
        np.testing.assert_array_equal(np.asarray(spikes), [[0, 0, 1, 0, 0, 0]])
        blocked = state_kernel(
            inputs=[*initial_state, mx.concatenate([shared[0]] * 2, axis=0),
                    mx.concatenate([shared[1]] * 2, axis=0), *shared[2:]],
            output_shapes=[shape] * 5 + [(2, *shape)],
            output_dtypes=[mx.float32, mx.float32, mx.int32, mx.int32, mx.int32, mx.uint8],
            grid=(6, 1, 1), threadgroup=(256, 1, 1), stream=stream,
        )
        mx.eval(*blocked)
        for value, expected in zip(blocked[:5], state, strict=True):
            assert_bits(value, expected)
        assert_bits(blocked[5], np.concatenate([np.asarray(first_history),
                                               np.asarray(second_history)], axis=0))


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
