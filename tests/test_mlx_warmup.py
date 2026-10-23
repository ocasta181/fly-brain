"""Fresh-state correctness and bounded reuse of native compilation probes."""

from dataclasses import replace

import numpy as np
import pytest
from native_execution import require_metal


@pytest.mark.metal
def test_warmup_reuse_preserves_independent_cpu_state_and_separates_shapes():
    mx = require_metal('Metal is required')
    import brian2 as b
    from configured_reference import reference
    from mlx_backend import configured
    from mlx_backend.model import DEFAULT_EQUATIONS, resolve_model
    from test_mlx_configured import graph, inputs
    model = resolve_model({"eqs": DEFAULT_EQUATIONS+"phase:integer\n",
        "eq_th": "True", "eq_rst": "v=v_rst;g=0*mV;phase+=1", "t_rfc": 0*b.ms})
    c, stimulus = graph(), inputs(model, targets=(), steps=5, trials=1)
    expected = reference(c, stimulus, model)
    p = configured.prepare(c, stimulus, model)
    configured._WARMUPS.clear()

    def check(state, copies=1):
        actual = configured.host_state(state, p)
        assert actual.keys() == expected.state.keys()
        for name, values in expected.state.items():
            right = np.repeat(values, copies, axis=0)
            raw = lambda array: array.view(np.uint64) if array.dtype == np.float64 else array
            np.testing.assert_array_equal(raw(actual[name]), raw(right), err_msg=name)

    cold, cold_state = configured.run_prepared(p, stimulus, record=False, return_state=True)
    assert cold.timings["jit_and_warmup_s"] > 0
    check(cold_state)
    warm, warm_state = configured.run_prepared(p, stimulus, record=False, return_state=True)
    assert warm.timings["jit_and_warmup_s"] == 0
    check(warm_state)
    recorded, state = configured.run_prepared(p, stimulus, record=True, return_state=True)
    assert recorded.timings["jit_and_warmup_s"] > 0
    np.testing.assert_array_equal(recorded.spikes.neurons, expected.spikes.neurons)
    np.testing.assert_array_equal(recorded.spikes.steps, expected.spikes.steps)
    check(state)
    shorter = replace(stimulus, events=stimulus.events[:, :3, :])
    assert configured.run_prepared(p, shorter, record=False).timings["jit_and_warmup_s"] > 0
    doubled = replace(stimulus, events=np.concatenate((stimulus.events, stimulus.events)),
                      trial_indices=(0, 1))
    multiple, state = configured.run_prepared(p, doubled, record=False, return_state=True)
    assert multiple.timings["jit_and_warmup_s"] > 0
    check(state, copies=2)
    zero = replace(stimulus, events=stimulus.events[:, :0, :])
    empty, state = configured.run_prepared(p, zero, record=False, return_state=True)
    assert empty.timings["jit_and_warmup_s"] == 0
    assert np.all(configured.host_state(state, p)["phase"] == 0)
    assert len(configured._WARMUPS) == 4
    assert all(key[0]() is p.program for key in configured._WARMUPS)
