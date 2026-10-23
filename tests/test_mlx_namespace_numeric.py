"""Accepted nonfinite namespaces retain their values and independent CPU behavior."""
from contextlib import nullcontext

import numpy as np
import pytest

from mlx_backend.model import resolve_model


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_namespace_nan_payloads_have_distinct_immutable_identities(monkeypatch, dtype):
    import brian2 as b
    monkeypatch.setitem(b.prefs, 'codegen.target', 'numpy')
    raw_dtype = np.uint32 if dtype == np.float32 else np.uint64
    raw = ([0x7f800001, 0x7f800002, 0xff800001, 0x7fc12345]
           if dtype == np.float32 else
           [0x7ff0000000000001, 0x7ff0000000000002,
            0xfff0000000000001, 0x7ff8123456789abc])
    models = [resolve_model({'custom_scalar': np.array(bits, raw_dtype).view(dtype)[()]})
              for bits in raw]
    assert len({model.signature for model in models}) == len(raw)
    for bits, model in zip(raw, models):
        restored = np.asarray(model.brian_namespace()['custom_scalar'])
        assert restored.dtype == np.dtype(dtype)
        assert int(restored.view(raw_dtype)) == bits
    if dtype == np.float64:
        # Python floats can also carry distinct signed NaN payloads.
        python_models = [resolve_model({'custom_scalar': float(np.array(bits, raw_dtype).view(dtype))})
                         for bits in raw]
        assert len({model.signature for model in python_models}) == len(raw)
        for bits, model in zip(raw, python_models):
            assert int(np.asarray(model.brian_namespace()['custom_scalar']).view(raw_dtype)) == bits


@pytest.mark.metal
@pytest.mark.parametrize('state_dtype', [np.float32, np.float64])
@pytest.mark.parametrize('payload', [0x7f800001, 0x7f800002, 0xff800001])
def test_original_api_nonfinite_namespace_matches_cpu(tmp_path, state_dtype, payload):
    import brian2 as b
    from mlx_backend.brian_device import original_api
    from original_api_reference import (
        assert_snapshot_equal, build_original_api_fixture, cpu_runtime,
    )
    values = {
        'custom_small': np.array(payload, np.uint32).view(np.float32)[()],
        'custom_wide': np.array(0xfff0000000000123, np.uint64).view(np.float64)[()],
        'custom_inf': float('inf'), 'custom_negative_inf': float('-inf'),
    }
    parameters = {
        **values,
        'eqs': 'v:volt\ng:volt\nrfc:second\ncopied:1\nlogged:1\nwide:1\npositive:1\nnegative:1\n',
        'eq_th': 'True', 't_rfc': 0*b.ms, 'r_poi': 0*b.Hz, 'r_poi2': 0*b.Hz,
        'eq_rst': 'v=v_rst;g=0*mV;copied=custom_small;logged=log10(custom_small);'
                  'wide=custom_wide;positive=custom_inf;negative=custom_negative_inf',
    }
    expected = []
    for native in (False, True):
        with cpu_runtime(float_dtype=state_dtype):
            saved = b.prefs.codegen.loop_invariant_optimisations
            b.prefs.codegen.loop_invariant_optimisations = False
            try:
                with original_api(workers=False) if native else nullcontext() as device:
                    fixture = build_original_api_fixture(tmp_path/str(native), parameters=parameters,
                                                         exc=(), exc2=(), silenced=())
                    for index, duration in enumerate((.1, .2)):
                        observed = fixture.run_ms(duration)
                        if native:
                            assert_snapshot_equal(observed, expected[index])
                        else:
                            expected.append(observed)
                    if native:
                        assert device.native_runs == 2 and device.native_steps == 3
            finally:
                b.prefs.codegen.loop_invariant_optimisations = saved


@pytest.mark.metal
@pytest.mark.parametrize('state_dtype', [np.float32, np.float64])
@pytest.mark.parametrize('target', ['numpy', 'cython'])
def test_mixed_functions_use_reference_namespace_constant_width(monkeypatch, state_dtype, target):
    import brian2 as b
    from component_cpu_reference import component_inputs
    from configured_reference import reference
    from mlx_backend import configured
    from test_mlx_inverse_trigonometry32 import assert_state

    monkeypatch.setitem(b.prefs, 'core.default_float_dtype', state_dtype)
    monkeypatch.setitem(b.prefs, 'codegen.target', target)
    model = resolve_model({
        'eqs': 'v:volt\ng:volt\nrfc:second\na:1\nc:1\ny:1\nh:1\n',
        'eq_th': 'True', 't_rfc': 0*b.ms, 'endpoint': np.float32(-1.),
        'eq_rst': 'v=v_rst;g=0*mV;a=arcsin(endpoint);c=arccos(endpoint);'
                  'y=arctan(endpoint);h=log10(3+endpoint)+sinh(endpoint)/16+tanh(endpoint)/32',
    })
    graph, stimulus = component_inputs(2, 2, model)
    expected = reference(graph, stimulus, model, cpu_target=target)
    prepared = configured.prepare(graph, stimulus, model)
    for recording in (True, False):
        result, state = configured.run_prepared(prepared, stimulus, record=recording,
                                                return_state=True, chunk_steps=1)
        assert_state(configured.host_state(state, prepared), expected.state)
        assert result.spikes.steps.size == (4 if recording else 0)
