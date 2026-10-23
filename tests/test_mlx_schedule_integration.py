"""Schedules retain packed typed state and the actual original-object API."""
from state_assertions import assert_recorded_spikes_equal, assert_spikes_equal
from contextlib import nullcontext
import gc

import numpy as np
import pytest

b = pytest.importorskip('brian2')
from native_execution import require_metal
mx = require_metal('Metal is required', allow_module_level=True)
from configured_reference import reference
from mlx_backend import configured
from mlx_backend.brian_device import original_api
from mlx_backend.model import DEFAULT_EQUATIONS, resolve_model
from test_mlx_configured import graph, inputs
from original_api_reference import (assert_snapshot_equal, assert_rng_state_equal,
    build_original_api_fixture, capture_rng_state, cpu_runtime)

pytestmark = pytest.mark.metal
R_FIRST = ('resets','groups','thresholds','synapses')
S_FIRST = ('synapses','groups','thresholds','resets')


def _equal(actual, expected):
    for name, value in expected.items():
        other = actual[name]
        if value.dtype.kind == 'f':
            dtype = np.uint64 if value.dtype.itemsize == 8 else np.uint32
            value, other = value.view(dtype), other.view(dtype)
        np.testing.assert_array_equal(other,value,err_msg=name)


@pytest.mark.parametrize('target,phases,delay',[
    ('cython',R_FIRST,0), ('numpy',S_FIRST,0), ('cython',S_FIRST,1),
    ('numpy',('thresholds','synapses','resets','groups'),7)])
def test_packed_schedule_states_spikes_staging_and_recording(monkeypatch,target,phases,delay):
    monkeypatch.setitem(b.prefs,'codegen.target',target)
    monkeypatch.setitem(b.prefs,'core.network.default_schedule',['start',*phases,'end'])
    model=resolve_model({'t_dly':delay*.1*b.ms,'t_rfc':.35*b.ms,
        'eqs':DEFAULT_EQUATIONS+'count : integer\n',
        'eq_rst':'v=v_rst;g=0*mV;count+=1'})
    c,s=graph(),inputs(model,steps=33)
    expected=reference(c,s,model,record_state=True)
    monkeypatch.setattr(b.Network,'run',lambda *a,**k:pytest.fail('Native code ran CPU network'))
    p=configured.prepare(c,s,model)
    assert p.window==max(1,min(32,model.arrival_delay_steps))
    state=configured.initial_state(p,2)
    assert isinstance(state.values,configured._PackedValues)
    recorded=[]
    for begin,end in ((0,9),(9,17),(17,33)):
        events=mx.array(np.ascontiguousarray(s.events[:,begin:end]).reshape(-1))
        for step in range(begin,end):
            state,bits=configured.advance(p,state,events,33,1,event_begin=begin,event_steps=end-begin)
            configured.evaluate_state(state)
            _equal(configured.host_state(state,p),{n:a[:,step] for n,a in expected.traces.items()})
            for row in np.flatnonzero(np.asarray(bits)&1):
                recorded.append((s.trial_indices[row//p.neurons],row%p.neurons,step))
    assert sorted(recorded)==sorted(zip(expected.spikes.trials,expected.spikes.neurons,expected.spikes.steps))
    actual,quiet=configured.run_prepared(p,s,record=False,return_state=True,chunk_steps=7)
    assert not actual.spikes.steps.size
    _equal(configured.host_state(quiet,p),expected.state)
    actual=configured.run_prepared(p,s,record=True,chunk_steps=7,block_windows=3)
    assert_spikes_equal(actual.spikes, expected.spikes)


@pytest.mark.parametrize('target,phases,delay,count_only',[
    ('cython',R_FIRST,.1,False), ('numpy',S_FIRST,0,False),
    ('cython',S_FIRST,.1,True)])
def test_actual_float32_schedule_rng_continuation_and_counts(tmp_path,monkeypatch,target,phases,delay,count_only):
    from mlx_backend.brian_poisson import NativePoisson
    monkeypatch.setattr(NativePoisson,'MAX_TAPE_DRAWS',12)
    expected=[]
    parameters={'t_dly':delay*b.ms,'t_rfc':.35*b.ms,'r_poi':3700*b.Hz,'r_poi2':7200*b.Hz}
    for native in (False,True):
        gc.collect()
        with cpu_runtime(seed=773, target=target, float_dtype=np.float32):
            b.prefs.core.network.default_schedule=['start',*phases,'end']
            with original_api(workers=False) if native else nullcontext() as device:
                fixture=build_original_api_fixture(tmp_path/str(native),parameters=parameters,
                    exc=(1,1),exc2=(3,),silenced=(),initial_state={
                    'v':np.array([-44.,-43.,-48.,-46.])*b.mV,
                    'g':np.array([1.,-3.,4.,2.])*b.mV})
                if count_only:
                    fixture.monitor=b.SpikeMonitor(fixture.neurons,record=False)
                    fixture.network=b.Network(fixture.neurons,fixture.synapses,fixture.monitor,*fixture.inputs)
                for index,duration in enumerate((.1,.15,.2,.25,.1)):
                    if index==2:b.prefs.core.default_float_dtype=np.float64
                    if index==3:fixture.inputs[1].active=False
                    observed=fixture.run_ms(duration)
                    if native:
                        assert_snapshot_equal(observed,expected[index])
                        session=device._sessions[fixture.network]
                        assert session.prepared.model.state_dtype=='float32'
                        assert isinstance(session.state.values,configured._PackedValues)
                        assert session.prepared.window==1
                    else:expected.append(observed)
                if native:assert_rng_state_equal(capture_rng_state(),rng)
                else:rng=capture_rng_state()
        del fixture
        gc.collect()


@pytest.mark.parametrize("target,dtype,phases", [
    ("numpy", np.float32, R_FIRST),
    ("cython", np.float64, S_FIRST),
])
def test_schedule_and_arctan_math_headers_match_cpu(monkeypatch, target, dtype, phases):
    """New phase ordering retains composed math and every intermediate state."""
    monkeypatch.setitem(b.prefs, "codegen.target", target)
    monkeypatch.setitem(b.prefs, "core.default_float_dtype", dtype)
    monkeypatch.setitem(b.prefs, "core.network.default_schedule", ["start", *phases, "end"])
    model = resolve_model({
        "eqs": DEFAULT_EQUATIONS+"argument:1\nobserved:1\ncount:integer\n",
        "eq_th": "True", "t_dly": .1*b.ms, "t_rfc": .35*b.ms,
        "eq_rst": "v=v_rst;g=0*mV;argument+=1/32;"
                  "observed=arctan(argument)+tanh(argument);count+=1",
    })
    c, s = graph(), inputs(model, steps=17)
    expected = reference(c, s, model, record_state=True, cpu_target=target)
    p = configured.prepare(c, s, model)
    state = configured.initial_state(p, 2)
    events = mx.array(np.ascontiguousarray(s.events).reshape(-1))
    for step in range(17):
        state, _ = configured.advance(p, state, events, 17, 1)
        configured.evaluate_state(state)
        _equal(configured.host_state(state, p),
               {name: values[:, step] for name, values in expected.traces.items()})
    for record in (False, True):
        result, state = configured.run_prepared(p, s, record=record,
            return_state=True, chunk_steps=5, block_windows=3)
        _equal(configured.host_state(state, p), expected.state)
        assert_recorded_spikes_equal(result, expected, record)
