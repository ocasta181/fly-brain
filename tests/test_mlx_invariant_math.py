"""Invariant model functions preserve independent Brian execution semantics."""
from state_assertions import assert_raw_state, assert_spikes_equal
from native_execution import forbid_call, require_metal

import numpy as np
import pytest

from mlx_backend.model import DEFAULT_EQUATIONS, resolve_model


@pytest.mark.metal
def test_invariant_exprel_boundaries_match_cpu_without_host_simulation(monkeypatch):
    mx = require_metal('Metal is required')
    b = pytest.importorskip("brian2")
    from configured_reference import reference
    from mlx_backend import configured
    from test_mlx_configured import graph, inputs

    monkeypatch.setitem(b.prefs, "codegen.target", "numpy")
    boundary = np.float64(1e-16)
    cases = [-1000., -1., -1e-17, 0., 1e-17,
             np.nextafter(boundary, 0.), boundary, np.nextafter(boundary, np.inf),
             .125, 700., 718.]
    params = {f"argument{j}": value for j, value in enumerate(cases)}
    params.update({
        "eqs": DEFAULT_EQUATIONS + "\n".join(f"sample{j}:1" for j in range(len(cases))),
        "eq_th": "t>=0*ms",
        "eq_rst": "v=v_rst;g=0*mV;" + ";".join(
            f"sample{j}=exprel(argument{j})" for j in range(len(cases))),
        "t_rfc": 0*b.ms,
    })
    model = resolve_model(params)
    c, s = graph(), inputs(model, targets=(), steps=5, trials=2)
    expected = reference(c, s, model)

    forbid_call(b.Network, 'run', 'Native execution advanced a CPU Brian network', monkeypatch)
    prepared = configured.prepare(c, s, model)
    for recording in (True, False):
        actual, state = configured.run_prepared(prepared, s, record=recording,
                                               return_state=True, chunk_steps=3)
        if recording:
            assert actual.spikes.steps.size == 30
            assert_spikes_equal(actual.spikes, expected.spikes)
        else:
            assert actual.spikes.steps.size == 0
        states = configured.host_state(state, prepared)
        assert_raw_state(states, expected.state)
