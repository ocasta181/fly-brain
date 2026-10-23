"""Arrival-scan preparation must retain the checked integer replay frontier."""
from native_execution import forbid_call
from dataclasses import replace

import brian2 as b
import numpy as np
import pytest

from component_cpu_reference import component_inputs
from configured_reference import reference
from mlx_backend import configured
from mlx_backend.model import DEFAULT_EQUATIONS, DEFAULT_RESET, resolve_model
from test_mlx_compiled_precision_configured import check


@pytest.mark.metal
def test_arrival_scan_retries_checked_temporaries_with_fresh_cached_inputs(tmp_path, monkeypatch):
    monkeypatch.setitem(b.prefs, 'codegen.target', 'cython')
    monkeypatch.setitem(b.prefs, 'core.default_integer_dtype', np.int64)
    monkeypatch.setitem(b.prefs, 'codegen.runtime.cython.cache_dir', str(tmp_path / 'cpu'))
    model = resolve_model(dict(
        eqs=DEFAULT_EQUATIONS + 'x:integer\ny:integer\np:integer\nq:integer\nseen:1\n',
        eq_th='True', t_rfc=0*b.ms, t_dly=.2*b.ms,
        eq_rst=DEFAULT_RESET + ';x=2;y=3;p+=1;q+=1;seen+=abs(x)**(p+60)-abs(y)**(q+38)',
    ), cpu_target='cython')
    graph, stimulus = component_inputs(4, 8, model)
    graph = replace(graph, sources=np.arange(4, dtype=np.int32),
                    destinations=np.array([1, 2, 3, 0], np.int32),
                    counts=np.ones(4), weights_mv=np.ones(4))
    stimulus = replace(stimulus, targets=(0, 1, 2, 3), rates_hz=(1.,)*4,
                       amplitudes_mv=(1.,)*4,
                       events=np.ones((1, 8, 4), np.uint8), sha256=None)
    expected = reference(graph, stimulus, model, cpu_target='cython')
    compile_ = configured._compiled_program
    capacities = []

    def small_capacity(model, *args, **kwargs):
        capacity = kwargs.pop('native_precision_limbs', 2)
        capacities.append(capacity)
        return compile_(model, *args, native_precision_limbs=capacity, **kwargs)

    monkeypatch.setattr(configured, '_compiled_program', small_capacity)
    prepared = configured.prepare(graph, stimulus, model, component=True)
    assert prepared.scan_arrivals and prepared.program.native_precision_limbs == 2
    assert prepared.components is None

    forbid_call(b.Network, 'run', 'Native arrival scan executed a CPU timestep', monkeypatch)
    for record in (False, True, False):
        result, state = configured.run_prepared(
            prepared, stimulus, record=record, return_state=True, chunk_steps=3,
        )
        check(configured.host_state(state, prepared), expected.state)
        assert state.step == 8 and state.native_precision_limbs >= 4
        if record:
            for name in ('trials', 'neurons', 'steps'):
                np.testing.assert_array_equal(getattr(result.spikes, name), getattr(expected.spikes, name))
        else:
            assert not result.spikes.steps.size
    assert max(capacities) >= 4
