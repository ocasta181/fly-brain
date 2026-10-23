"""Independent original Brian compiled-integer blocks versus actual Metal."""
from native_execution import forbid_cpu_steps, forbidden_call
from pathlib import Path

import brian2 as b
import numpy as np
import pytest

from mlx_backend.model import DEFAULT_EQUATIONS, DEFAULT_RESET
from original_api_protocol import check_original_api_runs, observe_run
from original_api_reference import build_original_api_fixture, cpu_runtime

pytestmark = pytest.mark.metal


@pytest.fixture(autouse=True)
def private_compiler_cache(monkeypatch):
    monkeypatch.setitem(b.prefs, "codegen.runtime.cython.cache_dir", str(
        Path(__file__).resolve().parents[1]/"executions/cython-compiled-integers-default"))


def _options(reset, *, x=None, threshold="True"):
    return dict(parameters=dict(eqs=DEFAULT_EQUATIONS+"x:integer\ny:integer\nflag:boolean\n",
        eq_th=threshold, eq_rst=DEFAULT_RESET+";"+reset, t_rfc=0*b.ms, w_syn=0*b.mV),
        exc=(), exc2=(), silenced=(), initial_state=dict(
            x=np.array([-5, np.iinfo(np.int64).min, np.iinfo(np.int64).max, 0], np.int64)
            if x is None else x, y=np.full(4, 17, np.int64), flag=np.ones(4, bool)))


def _forbid_cpu_steps(fixture, monkeypatch):
    forbid_cpu_steps(fixture.network, monkeypatch, 'A CPU timestep executed in the compiled-integer test', patch_preparation=True)


def _typed_fixture(types, reset, initial, *, threshold='True',record=True,equations=None,default_reset=True,namespace=None,
                   codeobj_class=None):
    from original_api_reference import OriginalAPIFixture,load_original_model
    types=dict(types) # NeuronGroup adds refractory dtypes to its supplied dict.
    original=load_original_model();params=dict(original.default_params)
    equations=equations or ''.join(name+(':boolean\n' if np.dtype(dtype).kind=='b' else
                            ':integer\n' if np.dtype(dtype).kind in 'iu' else ':1\n')
                      for name,dtype in types.items())
    params.update(eqs=DEFAULT_EQUATIONS+equations,eq_th=threshold,
                  eq_rst=(DEFAULT_RESET+';' if default_reset else '')+reset,t_rfc=0*b.ms)
    params.update(namespace or {})
    n=len(next(iter(initial.values())))
    codeobj={} if codeobj_class is None else dict(codeobj_class=codeobj_class)
    group=b.NeuronGroup(n,params['eqs'],threshold=threshold,reset=params['eq_rst'],
        refractory='rfc',method='linear',namespace=params,dtype=types,**codeobj)
    group.v=params['v_0'];group.g=0*b.mV;group.rfc=0*b.ms
    for name,value in initial.items():setattr(group,name,value)
    syn=b.Synapses(group,group,'w:volt',on_pre='g_post+=w',delay=params['t_dly'])
    syn.connect(i=[0],j=[0]);syn.w=0*b.volt
    monitor=b.SpikeMonitor(group,record=record)
    return OriginalAPIFixture(original,params,(),group,syn,monitor,[],b.Network(group,syn,monitor))


def test_actual_cython_abs_minimum_preserves_error_and_completed_prefix(tmp_path, monkeypatch):
    options=_options("y=abs(x)")
    def exercise(fixture, mode):
        failed=observe_run(fixture,.1,raises=b.BrianObjectException)
        if mode == 'cpu':
            assert isinstance(failed.error.__cause__, OverflowError)
            (tmp_path/"actual-cython-reset.txt").write_text(str(fixture.neurons.resetter["spike"].codeobj.code))
            assert failed.snapshot.state["y"].tolist()==[5,17,17,17]
        yield failed
        fixture.neurons.x[1]=6
        yield observe_run(fixture,.2)
    check_original_api_runs(lambda mode:build_original_api_fixture(tmp_path/mode,**options),
        exercise,monkeypatch,guard=_forbid_cpu_steps,target='cython',integer_dtype=np.int64)


def test_actual_cython_integer_component_matches_all_state_bits(monkeypatch):
    from component_cpu_reference import ComponentCpu,component_inputs
    from mlx_backend import configured
    from mlx_backend.model import resolve_model
    with cpu_runtime(integer_dtype=np.int64):
        names=[f'counter{j}' for j in range(12)]
        primes=[3,5,7,11,13,17,19,23,29,31,37,41]
        reset=';'.join(f'{name} += {names[(j+1)%12]}*{primes[j]}+{17+2*j}'
                       for j,name in enumerate(names))
        model=resolve_model(dict(eqs=DEFAULT_EQUATIONS+''.join(name+':integer\n' for name in names),
            eq_th='True',eq_rst=DEFAULT_RESET+';'+reset,t_rfc=0*b.ms),cpu_target='cython')
        graph,stimulus=component_inputs(7,5,model)
        cpu=ComponentCpu(7,5,model,'cython',
            Path(__file__).resolve().parents[1]/'executions/cython-compiled-integer-component')
        _,_,expected=cpu.run()
        prepared=configured.prepare(graph,stimulus,model)
        forbidden = forbidden_call('A CPU timestep executed in the native integer component')
        with monkeypatch.context() as patch:
            patch.setattr(b.Network,'run',forbidden)
            _,state=configured.run_prepared(prepared,stimulus,return_state=True,record=False)
            actual=configured.host_state(state,prepared)
        assert actual.keys()==expected.keys()
        for name,value in expected.items():
            assert actual[name].dtype==value.dtype
            assert actual[name].shape==value.shape
            assert actual[name].tobytes()==value.tobytes(),name


def test_actual_cython_four_accepted_widths_sparse_mask_and_continuation(monkeypatch):
    widths=('int32','int64','uint8','uint64')
    types={name:np.dtype(dtype) for j,dtype in enumerate(widths) for name in (f'x{j}',f'y{j}')}
    types['flag']=np.dtype('bool')
    initial={}
    for j,dtype in enumerate(widths):
        limits=np.iinfo(dtype)
        initial[f'x{j}']=np.resize(np.array([limits.min,limits.max,0,1,limits.max-1,limits.min+1],dtype),9)
        initial[f'y{j}']=np.full(9,17,dtype)
    reset=';'.join(f'y{j}=x{j}+x{j};x{j}+=1' for j in range(4))+';flag=(x1+1)>x1'
    def exercise(fixture, mode):
        for duration in (.1,.2):
            yield observe_run(fixture,duration)
    check_original_api_runs(lambda mode:_typed_fixture(types,reset,initial,threshold='i%3!=1'),
        exercise,monkeypatch,guard=_forbid_cpu_steps,target='cython',integer_dtype=np.int64)


@pytest.mark.parametrize('dtype',['int32','int64','uint8','uint64'])
def test_actual_cython_abs_keeps_frontend_operation_and_conversion(dtype,monkeypatch):
    types=dict(x=np.dtype(dtype),y=np.dtype(dtype))
    bounds=np.iinfo(dtype)
    initial=dict(x=np.array([bounds.min,bounds.max,0,1],dtype),y=np.full(4,17,dtype))
    def exercise(fixture, mode):
        yield observe_run(fixture,.1,raises=b.BrianObjectException,optional=True)
    check_original_api_runs(lambda mode:_typed_fixture(types,'y=abs(x)',initial),exercise,
        monkeypatch,guard=_forbid_cpu_steps,target='cython',integer_dtype=np.int64)


@pytest.mark.parametrize('float_dtype',['float32','float64'])
@pytest.mark.parametrize('expression',['int(f)','f'])
def test_actual_cython_floating_to_integer_conversion(float_dtype,expression,monkeypatch):
    types=dict(f=np.dtype(float_dtype),y=np.dtype('int64'))
    initial=dict(f=np.array([3.75,-3.75,2.**31,2.**32+1,-2.**31-1,2.**63,-2.**63,
        np.nan,np.inf,-np.inf,2.**64,2.**63-1024],float_dtype))
    initial.update(v=np.linspace(-.04,-.05,12)*b.volt,g=np.linspace(.001,.003,12)*b.volt)
    def exercise(fixture, mode):
        result=observe_run(fixture,.1)
        if mode == 'cpu' and expression=='f':
            assert fixture.neurons.resetter['spike'].codeobj.compiled_code['run'] is None
        yield result
    check_original_api_runs(lambda mode:_typed_fixture(types,'y='+expression,initial),exercise,
        monkeypatch,guard=_forbid_cpu_steps,target='cython',integer_dtype=np.int64)


def test_actual_cython_missing_threshold_module_keeps_eventspace(monkeypatch):
    types=dict(x=np.dtype('int32'),y=np.dtype('int64'))
    initial=dict(x=np.arange(9,dtype=np.int32))
    def build():
        fixture=_typed_fixture(types,'y+=1',initial,threshold='int(x)>0')
        eventspace=fixture.neurons.variables['_spikespace'].get_value()
        eventspace[:3]=[0,2,4];eventspace[-1]=3
        return fixture
    def exercise(fixture, mode):
        first=observe_run(fixture,.3)
        yield first
        yield observe_run(fixture,.2)
        if mode == 'cpu':
            assert fixture.neurons.thresholder['spike'].codeobj.compiled_code['run'] is None
            np.testing.assert_array_equal(first.snapshot.spike_i,np.tile([0,2,4],3))
    check_original_api_runs(lambda mode:build(),exercise,monkeypatch,
        guard=_forbid_cpu_steps,target='cython',integer_dtype=np.int64)


def test_actual_cpp_accepted_widths_optimizer_and_native_runtime(tmp_path):
    _actual_cpp_component(tmp_path,'widths')


def test_actual_cpp_boolean_store_retains_char_before_integer_consumer(tmp_path):
    _actual_cpp_component(tmp_path,'boolean')


def test_actual_cpp_float_casts_keep_all_seven_accepted_integer_widths(tmp_path):
    _actual_cpp_component(tmp_path,'casts')


def test_actual_cpp_out_of_range_constant_casts_retain_selected_compiler_behavior(tmp_path):
    _actual_cpp_component(tmp_path,'constant_casts')


def test_actual_cpp_boolean_store_respects_unsigned_char_policy(tmp_path,monkeypatch):
    from brian2.codegen.cpp_prefs import get_compiler_and_args
    monkeypatch.setitem(b.prefs,'codegen.cpp.extra_compile_args',
                        [*get_compiler_and_args()[1],'-funsigned-char'])
    _actual_cpp_component(tmp_path,'boolean',unsigned_char=True)


def _actual_cpp_component(tmp_path,mode,*,unsigned_char=False):
    import json,subprocess,sys
    from dataclasses import replace
    from hashlib import sha256
    from types import MappingProxyType
    import mlx.core as mx
    from mlx_backend import configured
    from mlx_backend.model import ModelAnalysis,ModelVariable,resolve_model
    from component_cpu_reference import component_inputs
    from probe_compiled_integer_cpp import case
    driver=Path(__file__).with_name('probe_compiled_integer_cpp.py')
    from mlx_backend.integer_policy import snapshot_integer_policy
    with cpu_runtime():policy=snapshot_integer_policy('cpp')
    identity=sha256(driver.read_bytes()+repr((policy.identity,mode)).encode()).hexdigest()
    directory=Path(__file__).resolve().parents[1]/'executions/compiled-integers/cpp-reference'/identity
    if not (directory/'identity.txt').exists():
        result=subprocess.run([sys.executable,str(driver),'--output',str(directory),'--case',mode,
                               *(['--unsigned-char'] if unsigned_char else [])],
            capture_output=True,text=True,timeout=70)
        assert result.returncode==0,result.stdout+'\n'+result.stderr
        (directory/'identity.txt').write_text(identity)
    assert (directory/'identity.txt').read_text()==identity
    metadata=json.loads((directory/'metadata.json').read_text())
    assert metadata['target']=='CPPStandaloneCodeObject'
    params,_,initial=case(mode)
    variables={name:ModelVariable(**dict(row,dimensions=tuple(row['dimensions']),flags=tuple(row['flags'])))
               for name,row in metadata['variables'].items()}
    analysis=ModelAnalysis(*metadata['abstract'],MappingProxyType(variables),metadata['brian_version'],
                           metadata['temporary_dtype'],'cpp')
    with cpu_runtime(integer_dtype=np.int64):
        model=resolve_model(params,cpu_target='cpp')
        model=replace(model,analysis=analysis,reference_compile_args=tuple(metadata['flags']),
                      signature=sha256((model.signature+repr(analysis)).encode()).hexdigest())
        graph,stimulus=component_inputs(metadata['neurons'],3,model)
        prepared=configured.prepare(graph,stimulus,model)
        state=configured.initial_state(prepared)
        values=dict(state.values)
        for name,value in initial.items():
            values[name]=mx.array(value.view(np.uint32 if value.dtype.itemsize==4 else np.uint64)
                                  if value.dtype.kind=='f' else value).view(
                configured._storage_dtype(prepared.program.variables[name].storage_dtype))
        state=replace(state,values=configured._pack_values(prepared.program,values))
        expected=np.load(directory/'final.npz')
        events=mx.array(stimulus.events.reshape(-1))
        words=[]
        for length in (1,2):
            state,bits=configured.advance(prepared,state,events,3,length)
            configured.evaluate_state(state);words.append(np.asarray(bits))
        for name,observed in configured.host_state(state,prepared).items():
            wanted=np.broadcast_to(expected[name][None,:],observed.shape)
            if observed.dtype.kind=='f':
                raw=np.uint32 if observed.dtype.itemsize==4 else np.uint64
                observed=observed.view(raw);wanted=wanted.view(raw)
            elif observed.dtype.kind=='b':
                observed=observed.view(np.uint8);wanted=wanted.view(np.uint8)
            np.testing.assert_array_equal(observed,wanted,err_msg=name)
        assert state.step==metadata['clock_timestep']==3
        rows=np.array([0,2,3,5,6,8]) if mode in ('widths','timed1','timed2') else np.arange(metadata['neurons'])
        np.testing.assert_array_equal(expected['spike_i'],np.tile(rows,3))
        actual_i=[];actual_steps=[]
        for word,length,start in zip(words,(1,2),(0,1)):
            for step in range(length):
                selected=np.flatnonzero((word[:metadata['neurons']]>>step)&1)
                actual_i.extend(selected);actual_steps.extend([start+step]*len(selected))
        np.testing.assert_array_equal(actual_i,expected['spike_i'])
        np.testing.assert_array_equal(actual_steps,np.rint(expected['spike_t']/model.dt_s).astype(np.int64))
