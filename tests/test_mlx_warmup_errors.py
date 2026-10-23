"""Synthetic warm-up inputs cannot invent an actual simulation error."""
from native_execution import forbid_call
from collections import OrderedDict
from dataclasses import replace

import brian2 as b
import numpy as np
import pytest

from component_cpu_reference import component_inputs
from configured_reference import reference
from mlx_backend import configured
from mlx_backend.model import DEFAULT_EQUATIONS, DEFAULT_RESET, resolve_model
from mlx_backend.runtime_errors import is_model_exception


def _inputs():
    model = resolve_model({
        'eqs': DEFAULT_EQUATIONS.replace('(unless refractory)', '')
            + 'x:integer\nseen:1\n',
        'eq_th': 'True',
        'eq_rst': 'seen=1/int(v>v_0);x+=1;' + DEFAULT_RESET,
        'v_0': -50*b.mV, 'v_rst': -50*b.mV, 't_rfc': 0*b.ms,
    }, cpu_target='cython')
    graph, stimulus = component_inputs(2, 4, model)
    return model, graph, stimulus


@pytest.mark.metal
@pytest.mark.parametrize('record', [False, True])
@pytest.mark.parametrize('event', [0, 1])
def test_synthetic_probe_preserves_actual_cpu_success_and_failure(event, record, monkeypatch, tmp_path):
    monkeypatch.setitem(b.prefs, 'codegen.target', 'cython')
    monkeypatch.setitem(b.prefs, 'codegen.runtime.cython.cache_dir', str(tmp_path/'cpu'))
    monkeypatch.setattr(configured, '_WARMUPS', OrderedDict())
    model, graph, stimulus = _inputs()
    stimulus = replace(stimulus, events=np.full((1, 4, 2), event, np.uint8),
                       targets=(0, 1), rates_hz=(1., 1.),
                       amplitudes_mv=(1., 1.), sha256=None)
    expected_error = None
    try:
        expected = reference(graph, stimulus, model, cpu_target='cython')
    except b.BrianObjectException as failure:
        expected_error = failure.__cause__
        assert isinstance(expected_error, ZeroDivisionError)
    assert (expected_error is None) == bool(event)
    prepared = configured.prepare(graph, stimulus, model)

    forbid_call(b.Network, 'run', 'CPU simulation called during native execution', monkeypatch)
    evaluate = configured.evaluate_state
    probe_errors = []

    def observe(state):
        try:
            return evaluate(state)
        except Exception as error:
            probe_errors.append(error)
            raise

    monkeypatch.setattr(configured, 'evaluate_state', observe)
    if expected_error is not None:
        with pytest.raises(type(expected_error)) as actual:
            configured.run_prepared(prepared, stimulus, record=record)
        assert str(actual.value) == str(expected_error)
    else:
        result, state = configured.run_prepared(
            prepared, stimulus, record=record, return_state=True)
        actual = configured.host_state(state, prepared)
        for name, wanted in expected.state.items():
            assert actual[name].dtype == wanted.dtype
            dtype = f'u{wanted.dtype.itemsize}' if wanted.dtype.kind == 'f' else wanted.dtype
            np.testing.assert_array_equal(actual[name].view(dtype), wanted.view(dtype), err_msg=name)
        for name in ('trials', 'neurons', 'steps'):
            wanted = getattr(expected.spikes, name) if record else np.empty(0, np.int64)
            np.testing.assert_array_equal(getattr(result.spikes, name), wanted)
    assert probe_errors and is_model_exception(probe_errors[0])


@pytest.mark.metal
@pytest.mark.parametrize('error_type', [RuntimeError, ZeroDivisionError])
def test_disposable_probe_does_not_hide_unmarked_failure(error_type, monkeypatch):
    monkeypatch.setattr(configured, '_WARMUPS', OrderedDict())
    model, graph, stimulus = _inputs()
    prepared = configured.prepare(graph, stimulus, model)
    error = error_type('unmarked probe failure')

    def fail(state):
        raise error

    monkeypatch.setattr(configured, 'evaluate_state', fail)
    with pytest.raises(error_type) as caught:
        configured.run_prepared(prepared, stimulus)
    assert caught.value is error
