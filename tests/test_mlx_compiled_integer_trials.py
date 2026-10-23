"""Independent actual Cython trials versus batched native integer state."""
from native_execution import forbid_call
from dataclasses import replace
import brian2 as b
import numpy as np
import pytest
from original_api_reference import cpu_runtime
from test_mlx_compiled_integer_adapter import _typed_fixture,private_compiler_cache


@pytest.mark.metal
def test_actual_cython_mixed_integer_trials_recording_and_split_continuation(monkeypatch):
    import mlx.core as mx
    from mlx_backend import configured
    from mlx_backend.brian_device import MLXDevice
    from component_cpu_reference import component_inputs
    n=9;steps=3;trial_ids=(2**40+3,2**40+9)
    types=dict(x=np.dtype('int64'),y=np.dtype('uint64'),flag=np.dtype('bool'))
    reset='x+=3;y+=x*5;flag=not flag'
    initial=[dict(x=np.arange(n,dtype=np.int64)*(1 if trial==0 else -1),
        y=np.full(n,(1<<64)-15+trial,np.uint64),flag=np.full(n,trial==0,bool)) for trial in range(2)]
    expected=[];model=None
    for values in initial:
        with cpu_runtime(target='cython', integer_dtype=np.int64):
            cpu=_typed_fixture(types,reset,values,threshold='i%3!=1')
            expected.append([cpu.run_ms(.1),cpu.run_ms(.2)])
            if model is None:model,_=MLXDevice._model(cpu.neurons,cpu.synapses,cpu.network,{})
    from brian2.codegen.runtime.cython_rt import CythonCodeObject
    forbid_call(CythonCodeObject, 'run', 'Native batched trials used a CPU timestep', monkeypatch)
    graph,stimulus=component_inputs(n,steps,model)
    stimulus=replace(stimulus,events=np.zeros((2,steps,0),np.uint8),trial_indices=trial_ids)
    prepared=configured.prepare(graph,stimulus,model);state=configured.initial_state(prepared,2)
    values=dict(state.values)
    for name in initial[0]:values[name]=mx.array(np.stack([row[name] for row in initial]).reshape(-1))
    state=replace(state,values=configured._pack_values(prepared.program,values))
    start=0;recorded=[]
    for index,length in enumerate((1,2)):
        state,bits=configured.advance(prepared,state,mx.array([],mx.uint8),steps,length)
        configured.evaluate_state(state);actual=configured.host_state(state,prepared)
        for name,value in actual.items():
            wanted=np.stack([row[index].state[name] for row in expected])
            raw=lambda a:a.view(np.uint64 if a.dtype.itemsize==8 else np.uint32) if a.dtype.kind=='f' else a
            assert value.dtype==wanted.dtype
            np.testing.assert_array_equal(raw(value),raw(wanted),err_msg=name)
        words=np.asarray(bits)[:2*n].reshape(2,n)
        for trial,identity in enumerate(trial_ids):
            for step in range(length):
                recorded.extend((identity,int(row),start+step) for row in np.flatnonzero((words[trial]>>step)&1))
        reference=[(identity,int(row),int(round(time/model.dt_s)))
            for trial,identity in enumerate(trial_ids)
            for row,time in zip(expected[trial][index].spike_i,expected[trial][index].spike_t_s)]
        assert sorted(recorded)==sorted(reference)
        assert state.step==start+length
        start+=length
