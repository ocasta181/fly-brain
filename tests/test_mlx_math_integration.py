"""Math helpers compose in the same configured component and CPU model."""
from native_execution import forbid_call
from dataclasses import replace

import brian2 as b
import numpy as np
import pytest

from component_cpu_reference import component_inputs
from configured_reference import reference
from mlx_backend import configured
from mlx_backend.model import DEFAULT_EQUATIONS, DEFAULT_RESET, resolve_model
from test_mlx_dataset_preparation_reuse import assert_state


@pytest.mark.metal
@pytest.mark.parametrize('strict', [False, True])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_selected_math_helpers_share_headers_and_recording_paths(dtype, strict, tmp_path, monkeypatch,
                                                               *, integer_phase=True, indexed=True):
    monkeypatch.setitem(b.prefs, 'core.default_float_dtype', dtype)
    monkeypatch.setitem(b.prefs, 'codegen.target', 'cython')
    if strict:
        monkeypatch.setitem(b.prefs, 'codegen.cpp.extra_compile_args',
                            ['-O3', '-fno-fast-math', '-ffp-contract=off'])
    monkeypatch.setitem(b.prefs, 'codegen.runtime.cython.cache_dir', str(tmp_path / 'cpu'))
    fields = ('phase:integer\n' if integer_phase else 'phase:1\n')
    fields += 'angle:1\nsine:1\ninverse_sine:1\ninverse_cosine:1\ntangent:1\nstable:1\njoint:1\n'
    reset = (DEFAULT_RESET + ';phase+=1;angle=' + ('(i+phase)' if indexed else 'phase') + '/64;'
             'sine=sin(angle);inverse_sine=arcsin(sine);inverse_cosine=arccos(sine);'
             'tangent=arctan(angle);stable=exprel(angle);'
             'joint=sin(inverse_cosine)+arctan(stable)+expm1(angle)'
             '+exp(angle)+log1p(angle)+log(1+angle)')
    model = resolve_model(dict(eqs=DEFAULT_EQUATIONS + fields, eq_th='True',
                               eq_rst=reset, t_rfc=0*b.ms, t_dly=.2*b.ms), cpu_target='cython')
    graph, stimulus = component_inputs(4, 9, model)
    graph = replace(graph, sources=np.arange(4, dtype=np.int32),
                    destinations=np.array([1, 2, 3, 0], np.int32),
                    counts=np.ones(4), weights_mv=np.ones(4))
    stimulus = replace(stimulus, targets=(0, 1, 2, 3), rates_hz=(1.,)*4,
                       amplitudes_mv=(1.,)*4, events=np.ones((1, 9, 4), np.uint8), sha256=None)
    expected = reference(graph, stimulus, model, cpu_target='cython')
    prepared = configured.prepare(graph, stimulus, model, component=True)

    forbid_call(b.Network, 'run', 'Native math integration executed a CPU timestep', monkeypatch)
    for record in (False, True):
        result, state = configured.run_prepared(prepared, stimulus, record=record,
                                                return_state=True, chunk_steps=5)
        assert_state(configured.host_state(state, prepared), expected.state)
        if record:
            for name in ('trials', 'neurons', 'steps'):
                np.testing.assert_array_equal(getattr(result.spikes, name), getattr(expected.spikes, name))
        else:
            assert not result.spikes.steps.size


@pytest.mark.metal
@pytest.mark.parametrize('indexed', [False, True])
@pytest.mark.parametrize('strict', [False, True])
def test_float32_math_combination_without_integer_state(indexed, strict, tmp_path, monkeypatch):
    test_selected_math_helpers_share_headers_and_recording_paths(
        np.float32, strict, tmp_path, monkeypatch, integer_phase=False, indexed=indexed,
    )
