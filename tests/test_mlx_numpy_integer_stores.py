"""Actual NumPy array-store loops use the selected spike vector's ranks."""
from native_execution import forbid_call, forbid_model_cpu_steps, require_metal
from dataclasses import replace

import numpy as np
import pytest

pytestmark = pytest.mark.metal


@pytest.mark.parametrize('dtype', ['int8','uint8','int16','uint16','int32','uint32','int64','uint64'])
def test_numpy_float64_integer_reset_store_context(monkeypatch,tmp_path,dtype,source_dtype=np.float64,
                                                  store_expression='f'):
    import brian2 as b
    from mlx_backend import configured
    from mlx_backend.model import resolve_model
    from mlx_backend.models import Connectome,Stimulus
    mx = require_metal('Metal required')
    monkeypatch.setitem(b.prefs,'codegen.target','numpy')
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',source_dtype)
    monkeypatch.setitem(b.prefs,'core.default_integer_dtype',np.dtype(dtype).type)
    monkeypatch.setitem(b.prefs,'codegen.loop_invariant_optimisations',False)
    n,trials,steps=49,2,3
    equations='v:volt\ng:volt\nrfc:second\nx:integer\ny:integer\nf:1\nflag:boolean'
    reset=f'x={store_expression};y=x+1;f=f+1'
    model=resolve_model({'eqs':equations,'eq_th':'flag','eq_rst':reset,'t_rfc':0*b.ms})
    values=np.resize(np.array([4294967297.,-4294967297.,2.**63-1024.,2.**64,-2.**64,
                              1.5,-1.5,0.,-0.,65537.,-65537.],source_dtype),(trials,n))
    flags=np.stack((np.arange(n)%4!=1,np.arange(n)%3!=1))
    names=('x','y','f','flag','lastspike','not_refractory')
    expected={name:[] for name in names}
    for trial in range(trials):
        clock=b.Clock(dt=model.dt_s*b.second)
        group=b.NeuronGroup(n,equations,threshold=model.eq_th,reset=reset,refractory='rfc',
            method='linear',namespace=model.brian_namespace(),clock=clock)
        assert np.dtype(group.variables['f'].dtype)==np.dtype(source_dtype)
        group.v=model.v_0*b.volt;group.g=0*b.volt;group.rfc=0*b.second
        group.f=values[trial];group.flag=flags[trial]
        monitor=b.StateMonitor(group,names,record=True,when='end')
        b.Network(group,monitor).run(steps*clock.dt)
        (tmp_path/f'cpu-reset-{trial}.txt').write_text(str(group.resetter['spike'].codeobj.code))
        for name in names:expected[name].append(np.asarray(getattr(monitor,name)).T.copy())
    expected={name:np.stack(rows) for name,rows in expected.items()}
    empty=np.array([],np.int32)
    graph=Connectome(np.arange(n),empty,empty,np.array([],float),np.array([],float))
    stimulus=Stimulus(np.zeros((trials,steps,0),np.uint8),(),(),(2**40+1,2**40+7),0,0,'integer-stores',model.dt_ms)
    forbid_call(b.Network, 'run', 'Native integer store used CPU timesteps', monkeypatch)
    prepared=configured.prepare(graph,stimulus,model)
    if store_expression=='f':
        needs_rank=(np.dtype(dtype).itemsize<8 if source_dtype==np.float64 else dtype=='uint16')
        assert prepared.program.collective_reset==needs_rank
    initial=configured.initial_state(prepared,trials)
    rawtype=np.uint64 if source_dtype==np.float64 else np.uint32
    initial=replace(initial,values={**initial.values,'f':mx.array(values.view(rawtype).reshape(-1)),
        'flag':mx.array(flags.reshape(-1))})
    state=initial
    def compare(state):
        actual=configured.host_state(state,prepared)
        for name in names:
            a,e=actual[name],expected[name][:,state.step-1]
            assert a.dtype==e.dtype,(dtype,name)
            if a.dtype.kind=='f':
                bits=np.uint64 if a.dtype.itemsize==8 else np.uint32
                a,e=a.view(bits),e.view(bits)
            np.testing.assert_array_equal(a,e,err_msg=f'{dtype}/{name}/step{state.step}')
    for _ in range(steps):
        state,_=configured.advance(prepared,state,mx.zeros((1,),mx.uint8),steps,1)
        configured.evaluate_state(state);compare(state)
    monkeypatch.setattr(configured,'initial_state',lambda *args,**kwargs:initial)
    for recording in (False,True):
        result,state=configured.run_prepared(prepared,stimulus,record=recording,return_state=True)
        compare(state)
        assert result.spikes.steps.size==(int(flags.sum())*steps if recording else 0)


@pytest.mark.parametrize('dtype', ['int8','uint8','int16','uint16','int32','uint32','int64','uint64'])
def test_numpy_float32_integer_reset_store_context(monkeypatch,tmp_path,dtype):
    test_numpy_float64_integer_reset_store_context(monkeypatch,tmp_path,dtype,source_dtype=np.float32)


@pytest.mark.parametrize('dtype', ['int8','uint8','int16','uint16','int32','uint32','int64','uint64'])
def test_numpy_half_integer_reset_store_context(monkeypatch,tmp_path,dtype):
    test_numpy_float64_integer_reset_store_context(monkeypatch,tmp_path,dtype,source_dtype=np.float32,
                                                  store_expression='sin(flag)-3')


def test_numpy_runtime_precision_integer_reset_store_context(monkeypatch,tmp_path):
    test_numpy_float64_integer_reset_store_context(monkeypatch,tmp_path,'uint16',source_dtype=np.float32,
                                                  store_expression='f+t*1e45*Hz')


@pytest.mark.parametrize('source_dtype',[np.float32,np.float64])
def test_numpy_boolean_sqrt_integer_store_preserves_half_rounding(monkeypatch,tmp_path,source_dtype):
    test_numpy_float64_integer_reset_store_context(monkeypatch,tmp_path,'uint16',source_dtype=source_dtype,
                                                  store_expression='sqrt(flag)+0.9999')


@pytest.mark.parametrize('dtype',['int8','uint8','int16','uint16','int32','uint32'])
def test_numpy_scalar_float_integer_reset_store_context(monkeypatch,tmp_path,dtype):
    test_numpy_float64_integer_reset_store_context(monkeypatch,tmp_path,dtype,
                                                  store_expression='4294967297.')


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),1e100],ids=['nan','infinity','finite-overflow'])
@pytest.mark.parametrize('spikes',[False,True],ids=['empty-spikes','spikes'])
def test_python_scalar_store_failure_and_repaired_api_continuation(tmp_path,monkeypatch,bad,spikes,recording=True,first_bad=17.,default_dtype=np.float64,phases=None,power_fault=False,base_name='k'):
    from contextlib import nullcontext
    import brian2 as b
    from mlx_backend.brian_device import original_api
    from mlx_backend.model import DEFAULT_EQUATIONS
    from original_api_reference import (build_original_api_fixture,cpu_runtime,assert_snapshot_equal,
        capture_rng_state,assert_rng_state_equal)
    expected=[];rng=[];cause=None
    for native in (False,True):
        with cpu_runtime():
            b.prefs.codegen.target='numpy';b.prefs.core.default_integer_dtype=np.int8
            b.prefs.core.default_float_dtype=default_dtype
            monkeypatch.setitem(b.prefs,'codegen.loop_invariant_optimisations',False)
            params=dict(eqs=DEFAULT_EQUATIONS+'x:integer\ny:integer\n',
                eq_th=str(spikes),eq_rst='v=v_rst;x=first_bad;y=bad;g+=1*mV',
                first_bad=first_bad,bad=bad,r_poi=4900*b.Hz,r_poi2=3200*b.Hz,t_rfc=0*b.ms)
            if power_fault:
                params['eqs']+=base_name+':integer\nnegative:integer\n'
                params['eq_rst']='v=v_rst;x=bad;y='+base_name+'**negative;g+=1*mV'
            with original_api(workers=False) if native else nullcontext():
                initial=dict(x=[7,7,7,7],y=[8,8,8,8])
                if power_fault:initial.update({base_name:[2,3,5,7],'negative':[-1]*4})
                fixture=build_original_api_fixture(tmp_path/str(native),parameters=params,
                    initial_state=initial,exc=(0,0),exc2=(2,),silenced=())
                assert np.dtype(fixture.neurons.variables['v'].dtype)==np.dtype(default_dtype)
                assert np.dtype(fixture.neurons.variables['x'].dtype)==np.dtype('int8')
                if phases is not None:fixture.network.schedule=['start',*phases,'end']
                if not recording:
                    fixture.network.remove(fixture.monitor)
                    fixture.monitor=b.SpikeMonitor(fixture.neurons,record=False)
                    fixture.network.add(fixture.monitor)
                if native:
                    forbid_model_cpu_steps(fixture, monkeypatch, 'CPU timestep executed', patch_preparation=True)
                zero=fixture.run_ms(0.)
                with pytest.raises(b.BrianObjectException) as error:fixture.run_ms(.1)
                current=(type(error.value.__cause__),str(error.value.__cause__))
                if power_fault:assert current[0] is (ValueError if spikes else OverflowError)
                if native:
                    assert current==cause
                    assert_snapshot_equal(zero,expected[0]);assert_snapshot_equal(fixture.capture(),expected[1])
                    assert_rng_state_equal(capture_rng_state(),rng[0])
                else:
                    cause=current;expected.extend((zero,fixture.capture()));rng.append(capture_rng_state())
                    (tmp_path/'actual-reset.py').write_text(str(fixture.neurons.resetter['spike'].codeobj.code.run))
                fixture.neurons.namespace['bad']=3.5
                fixture.neurons.namespace['first_bad']=17.
                if power_fault:fixture.neurons.negative_=1
                observed=fixture.run_ms(.15)
                if native:
                    assert_snapshot_equal(observed,expected[2]);assert_rng_state_equal(capture_rng_state(),rng[1])
                else:expected.append(observed);rng.append(capture_rng_state())


@pytest.mark.parametrize('dtype',['int8','uint8','int16','uint16','int32','uint32','int64','uint64'])
@pytest.mark.parametrize('optimise',[False,True])
def test_scalar_object_provenance_survives_namespace_hoist_alias_and_clock(monkeypatch,tmp_path,dtype,optimise,distinct_source=False):
    import brian2 as b
    from mlx_backend import configured
    from mlx_backend.model import resolve_model
    from mlx_backend.models import Connectome,Stimulus
    mx=pytest.importorskip('mlx.core')
    monkeypatch.setitem(b.prefs,'codegen.target','numpy')
    monkeypatch.setitem(b.prefs,'core.default_integer_dtype',np.dtype(dtype).type)
    monkeypatch.setitem(b.prefs,'codegen.loop_invariant_optimisations',optimise)
    fields=('a','b','c','d','negative','f','h','low','high','fraction')
    if dtype in ('uint32','uint64'):fields+=('unsigned_high',)
    equations='v:volt\ng:volt\nrfc:second\nflag:boolean\n'+''.join(name+':integer\n' for name in fields)
    reset='tmp=4294967297.;alias=tmp;tmp2=tmp+1;a=4294967297.;b=narrow_constant;c=floor(4294967297.);d=4294967297.+t*Hz;negative=-65535.;f=alias;h=tmp2;low=-9223372036854775808.;high=9223372036854774784.;fraction=-1.5'
    if dtype in ('uint32','uint64'):reset+=';unsigned_high=9223372036854775808.'
    if distinct_source:reset=reset.replace('c=floor(4294967297.)','c=floor(4294967297.)+0')
    params=dict(eqs=equations,eq_th='flag',eq_rst=reset,narrow_constant=np.float32(4294967297.))
    model=resolve_model(params)
    n,steps=5,3
    flags=np.array([True,False,True,True,False]);expected={name:[] for name in fields}
    group=b.NeuronGroup(n,equations,threshold='flag',reset=reset,refractory='rfc',
        namespace=params,dt=model.dt_s*b.second)
    group.flag=flags;monitor=b.StateMonitor(group,fields,record=True,when='end')
    b.Network(group,monitor).run(steps*model.dt_s*b.second)
    (tmp_path/'actual-reset.py').write_text(str(group.resetter['spike'].codeobj.code.run))
    for name in fields:expected[name]=np.asarray(getattr(monitor,name)).T.copy()
    empty=np.array([],np.int32);graph=Connectome(np.arange(n),empty,empty,np.array([],float),np.array([],float))
    stimulus=Stimulus(np.zeros((1,steps,0),np.uint8),(),(),(91,),0,0,'scalar-provenance',model.dt_ms)
    forbid_call(b.Network, 'run', 'Native scalar stores used CPU timesteps', monkeypatch)
    prepared=configured.prepare(graph,stimulus,model)
    initial=configured.initial_state(prepared,1)
    initial=replace(initial,values={**initial.values,'flag':mx.array(flags)})
    state=initial
    for step in range(steps):
        state,_=configured.advance(prepared,state,mx.zeros((1,),mx.uint8),steps,1)
        configured.evaluate_state(state);actual=configured.host_state(state,prepared)
        for name in fields:
            np.testing.assert_array_equal(actual[name][0],expected[name][step],err_msg=f'{dtype}/{optimise}/{name}/{step}')


def test_scalar_store_failure_count_only_and_competing_store_order(tmp_path,monkeypatch):
    test_python_scalar_store_failure_and_repaired_api_continuation(
        tmp_path,monkeypatch,float('nan'),True,recording=False,first_bad=float('inf'))


@pytest.mark.parametrize('dtype',['int8','uint64'])
def test_fresh_optimized_numpy_scalar_store_source(monkeypatch,tmp_path,dtype):
    test_scalar_object_provenance_survives_namespace_hoist_alias_and_clock(
        monkeypatch,tmp_path,dtype,True,distinct_source=True)


@pytest.mark.parametrize('spikes',[False,True])
def test_single_precision_scalar_store_failure_nondefault_schedule(tmp_path,monkeypatch,spikes):
    test_python_scalar_store_failure_and_repaired_api_continuation(tmp_path,monkeypatch,
        float('nan'),spikes,default_dtype=np.float32,
        phases=('groups','thresholds','resets','synapses'))


@pytest.mark.parametrize('spikes',[False,True])
def test_expression_and_scalar_store_priority_noncolliding_control(tmp_path,monkeypatch,spikes):
    test_python_scalar_store_failure_and_repaired_api_continuation(tmp_path,monkeypatch,
        float('inf'),spikes,power_fault=True,base_name='base_value')
