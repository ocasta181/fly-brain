"""Direct configured streaming/export uses the same accepted precision frontier."""
from state_assertions import assert_exact_state as check
import brian2 as b
import numpy as np
import pytest
from mlx_backend import configured
from mlx_backend.model import DEFAULT_EQUATIONS,DEFAULT_RESET,resolve_model
from original_api_reference import cpu_runtime
from component_cpu_reference import ComponentCpu,component_inputs




@pytest.mark.metal
@pytest.mark.parametrize('record',[False,True])
def test_direct_native_precision_retry_preserves_chunks_and_exports(record,tmp_path,monkeypatch):
    with cpu_runtime(target='cython', integer_dtype=np.int64):
        model=resolve_model(dict(eqs=DEFAULT_EQUATIONS+'x:integer\ny:integer\np:integer\nq:integer\nseen:1\n',
            eq_th='True',eq_rst=DEFAULT_RESET+';x=2;y=3;p+=1;q+=1;seen+=abs(x)**p-abs(y)**q',t_rfc=0*b.ms),cpu_target='cython')
        cpu=ComponentCpu(4,64,model,'cython',tmp_path/'cpu',prefix='precision_export')
        monitor=b.SpikeMonitor(cpu.neurons,record=record);cpu.network.add(monitor);cpu.network.store('initial')
        _,_,expected=cpu.run()
        expected_i=np.asarray(monitor.i[:]).copy() if record else np.empty(0,np.int32)
        expected_t=np.rint(np.asarray(monitor.t[:]/b.second)/model.dt_s).astype(np.int64) if record else np.empty(0,np.int64)
        graph,stimulus=component_inputs(4,64,model)
        compile_=configured._compiled_program;capacities=[]
        def explicit_precision(model,*args,**kwargs):
            capacity=kwargs.get('native_precision_limbs',2);capacities.append(capacity)
            return compile_(model,*args,native_precision_limbs=capacity)
        monkeypatch.setattr(configured,'_compiled_program',explicit_precision)
        prepared=configured.prepare(graph,stimulus,model)
        class Sink:
            def __init__(self):self.parts=[]
            def append(self,events):
                self.parts.append(tuple(np.array(value,copy=True) for value in (events.trials,events.neurons,events.steps)))
                return 0.
        sink=Sink()
        result,state=configured.run_prepared(prepared,stimulus,record=record,spike_sink=sink if record else None,
            return_state=True,chunk_steps=32,block_windows=2)
        check(configured.host_state(state,prepared),expected)
        assert state.step==64 and state.native_precision_limbs>=4
        assert max(capacities)>=4
        if record:
            assert len(sink.parts)==2 # one accepted export per immutable input chunk
            trials,neurons,steps=[np.concatenate([part[column] for part in sink.parts]) for column in range(3)]
            np.testing.assert_array_equal(trials,np.zeros(len(expected_i),np.int64))
            np.testing.assert_array_equal(neurons,expected_i)
            np.testing.assert_array_equal(steps,expected_t)
        assert len(result.spikes.neurons)==0
