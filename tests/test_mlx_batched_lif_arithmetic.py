"""Exact existing binary64 helpers on the long batched canonical LIF path."""
from state_assertions import assert_exact_state as assert_raw_state, assert_spikes_equal
from dataclasses import replace
import ctypes

import numpy as np
import pytest

from native_execution import require_metal
mx = require_metal('Metal is required', allow_module_level=True)
b = pytest.importorskip("brian2")

from configured_reference import reference
from mlx_backend import binary64, configured, math64, operator64
from mlx_backend.model import resolve_model
from mlx_backend.runner import _exact_default_model
from test_mlx_configured import graph, inputs

pytestmark = pytest.mark.metal




def test_long_batched_lif_preserves_cpu_bits_recording_and_boundaries(monkeypatch):
    monkeypatch.setitem(b.prefs, "codegen.target", "cython")
    model = resolve_model({"tau": 7*b.ms, "t_mbr": 24*b.ms}, cpu_target="cython")
    stimulus = inputs(model, steps=10003, trials=4, amplitudes=(75., -2.75, 68.75))
    events = stimulus.events.copy()
    events[2, 9982::3, 1] = 1
    events[3, 9979::2, 2] = 1
    stimulus = replace(stimulus, events=events)
    c = graph()
    expected = reference(c, stimulus, model)
    p = configured.prepare(c, stimulus, model)
    assert p.compact_idle
    factory = configured._neuron_kernel
    selected = []

    def observe(*args, **kwargs):
        selected.append(kwargs.get("batched_lif_arithmetic", False))
        return factory(*args, **kwargs)

    monkeypatch.setattr(configured, "_neuron_kernel", observe)
    actual, state = configured.run_prepared(p, stimulus, return_state=True,
        chunk_steps=9973, block_windows=13)
    assert all(selected)
    assert_spikes_equal(actual.spikes, expected.spikes)
    assert_raw_state(configured.host_state(state, p), expected.state)
    off, off_state = configured.run_prepared(p, stimulus, record=False,
        return_state=True, chunk_steps=9991)
    assert off.spikes.steps.size == 0
    assert_raw_state(configured.host_state(off_state, p), expected.state)
    for kind in ("history", "due"):
        recorded, disabled = getattr(state, kind), getattr(off_state, kind)
        assert recorded.keys() == disabled.keys()
        for key in recorded:
            np.testing.assert_array_equal(np.asarray(recorded[key]), np.asarray(disabled[key]))


@pytest.mark.parametrize("trials,steps,selected", [(4,10000,True),(4,1000,False),(1,10000,False)])
def test_long_batch_dispatch_preserves_accepted_short_and_single_paths(monkeypatch,trials,steps,selected):
    monkeypatch.setitem(b.prefs, "codegen.target", "cython")
    model = _exact_default_model(resolve_model())
    stimulus = inputs(model, steps=steps, trials=trials)
    p = configured.prepare(graph(), stimulus, model)
    factory = configured._neuron_kernel
    observed = []

    def observe(*args, **kwargs):
        observed.append(kwargs.get("batched_lif_arithmetic", False))
        return factory(*args, **kwargs)

    monkeypatch.setattr(configured, "_neuron_kernel", observe)
    state, _ = configured.advance(p, configured.initial_state(p, trials),
        mx.array(stimulus.events.reshape(-1)), steps, length=p.window)
    configured.evaluate_state(state)
    assert observed == [selected]


def test_reused_multiply_keeps_exact_finite_and_exceptional_bits():
    rng = np.random.default_rng(20261020)
    raw = rng.integers(0, np.iinfo(np.uint64).max, (8192,2), dtype=np.uint64)
    boundary = np.array([0,0x8000000000000000,1,0x8000000000000001,
        0x000fffffffffffff,0x0010000000000000,0x3ff0000000000000,
        0x3fefffffffffffff,0x7fefffffffffffff,0x7ff0000000000000,
        0xfff0000000000000,0x7ff8000012345678,0xfff0000000001234],np.uint64)
    grid = np.array(np.meshgrid(boundary,boundary)).reshape(2,-1).T.copy()
    raw = np.ascontiguousarray(np.concatenate((raw,grid)))
    kernel = mx.fast.metal_kernel(name="batched_lif_multiply_exact",input_names=["raw"],
        output_names=["fast","general"],
        header=binary64.MSL_HEADER+math64.MSL_MATH_HEADER+operator64._OPERATOR_FMA_HEADER,
        source="uint i=thread_position_in_grid.x;if(i<COUNT){fast[i]=_eon_operator_mul(raw[2*i],raw[2*i+1]);general[i]=d_mul(raw[2*i],raw[2*i+1]);}",
        compile_options={"math_mode":"safe"})
    fast, general = kernel(inputs=[mx.array(raw.reshape(-1))],template=[("COUNT",len(raw))],
        grid=(len(raw),1,1),threadgroup=(128,1,1),output_shapes=[(len(raw),)]*2,
        output_dtypes=[mx.uint64]*2)
    mx.eval(fast,general)
    fast, general = np.asarray(fast), np.asarray(general)
    np.testing.assert_array_equal(fast,general)
    with np.errstate(all="ignore"):
        cpu = raw[:,0].view(np.float64)*raw[:,1].view(np.float64)
    finite = ~np.isnan(cpu)
    np.testing.assert_array_equal(fast[finite],cpu.view(np.uint64)[finite])


def test_reused_fma_keeps_hardware_bits_and_general_exceptional_results():
    from test_mlx_floating_operators import polynomial_fma_cases
    from test_mlx_stable_math64 import fma_cases
    raw = np.ascontiguousarray(np.concatenate((polynomial_fma_cases(), fma_cases())))
    fma = ctypes.CDLL(None).fma
    fma.argtypes = [ctypes.c_double]*3
    fma.restype = ctypes.c_double
    cpu = np.array([fma(*(float(v) for v in row)) for row in raw.view(np.float64)],np.float64)
    kernel = mx.fast.metal_kernel(name="batched_lif_fma_exact",input_names=["raw"],
        output_names=["fast","general"],
        header=binary64.MSL_HEADER+math64.MSL_MATH_HEADER+operator64._OPERATOR_FMA_HEADER,
        source="uint i=thread_position_in_grid.x;if(i<COUNT){fast[i]=_eon_operator_polynomial_fma(raw[3*i],raw[3*i+1],raw[3*i+2]);general[i]=d_fma(raw[3*i],raw[3*i+1],raw[3*i+2]);}",
        compile_options={"math_mode":"safe"})
    fast, general = kernel(inputs=[mx.array(raw.reshape(-1))],template=[("COUNT",len(raw))],
        grid=(len(raw),1,1),threadgroup=(128,1,1),output_shapes=[(len(raw),)]*2,
        output_dtypes=[mx.uint64]*2)
    mx.eval(fast,general)
    fast, general = np.asarray(fast), np.asarray(general)
    np.testing.assert_array_equal(fast,general)
    valid = ~np.isnan(cpu)
    np.testing.assert_array_equal(fast[valid],cpu.view(np.uint64)[valid])
