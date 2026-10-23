"""Bit-exact arctangent against independent CPU and actual Metal execution."""
from state_assertions import (
    assert_exact_state as _assert_state,
    assert_recorded_spikes_equal,
    assert_spikes_equal,
)
import ast
import ctypes
from pathlib import Path
import subprocess

import numpy as np
import pytest
from native_execution import require_metal
from pinned_apple_math import require_apple_arm64_reference

from mlx_backend import arctangent64, binary64, math64


def cases():
    centers = [0., -0., 2.**-1074, 2.**-1022, .5, .75, 1., 4/3, 5/3, 2.,
               2.**53, float.fromhex('0x1.d12ed0af1a27fp-27'),
               float.fromhex('0x1.4000000000027p-1'),
               float.fromhex('0x1.c0000000f4213p-1'),
               float.fromhex('0x1.2aaaaaaaaaa96p+0'),
               float.fromhex('0x1.8000000000003p+0'),
               float.fromhex('0x1.d555555461337p+0')]
    centers += [np.ldexp(1., exponent) for exponent in range(-1074, 1024, 7)]
    values = np.asarray([y for x in centers for sign in (-1, 1)
                         for y in (np.nextafter(sign*x, -np.inf), sign*x,
                                   np.nextafter(sign*x, np.inf))], np.float64)
    special = np.array([0, 1, 0x8000000000000000, 0x8000000000000001,
        0x000fffffffffffff, 0x800fffffffffffff, 0x7fefffffffffffff,
        0xffefffffffffffff, 0x7ff0000000000000, 0xfff0000000000000,
        0x7ff0000000000001, 0xfff0000000000001, 0x7ff8123456789abc,
        0xfff8123456789abc, 0x7fffffffffffffff, 0xffffffffffffffff], np.uint64)
    rng = np.random.default_rng(20261018)
    random = rng.integers(0, np.iinfo(np.uint64).max, 16384, dtype=np.uint64)
    dense = rng.uniform(-4, 4, 16384).view(np.uint64)
    return np.unique(np.r_[values.view(np.uint64), special, random, dense])


def expected(raw, target='numpy'):
    require_apple_arm64_reference()
    if target == 'numpy':
        from brian2.core.functions import DEFAULT_FUNCTIONS
        with np.errstate(all='ignore'):
            return np.asarray(DEFAULT_FUNCTIONS['arctan'].pyfunc(raw.view(np.float64))).view(np.uint64)
    library = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
    fn = library.atan
    fn.argtypes, fn.restype = [ctypes.c_double], ctypes.c_double
    return np.asarray([fn(x) for x in raw.view(np.float64)]).view(np.uint64)


def assert_bits(actual, reference, inputs):
    bad = np.flatnonzero(actual != reference)
    if len(bad):
        i = bad[0]
        pytest.fail(f'arctan({int(inputs[i]):#018x}): actual={int(actual[i]):#018x}, '
                    f'CPU={int(reference[i]):#018x}; {len(bad)}/{len(inputs)} differ')


@pytest.fixture(scope='module')
def host(tmp_path_factory):
    require_apple_arm64_reference()
    module = ast.parse(Path(__file__).with_name('test_mlx_math64.py').read_text())
    prefix = next(ast.literal_eval(n.value) for n in module.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'HOST_PREFIX' for t in n.targets))
    directory = tmp_path_factory.mktemp('arctangent-host')
    source, library = directory/'atan.cpp', directory/'atan.so'
    source.write_text(prefix+'\n#define constant const\n'+binary64.MSL_HEADER+
        math64.MSL_MATH_HEADER+arctangent64.metal_arctangent_header({'arctan'})+r'''
extern "C" void run_atan(const ulong*x,ulong*y,int count) {
    for(int i=0;i<count;++i)y[i]=_eon_arctan64(x[i]);
}
''')
    build = subprocess.run(['clang++', '-std=c++17', '-shared', '-fPIC', '-O2',
        '-fno-fast-math', '-ffp-contract=off', str(source), '-o', str(library)],
        capture_output=True, text=True, timeout=30)
    assert build.returncode == 0, build.stderr
    result = ctypes.CDLL(str(library))
    pointer = np.ctypeslib.ndpointer(dtype=np.uint64, flags='C_CONTIGUOUS')
    result.run_atan.argtypes, result.run_atan.restype = [pointer,pointer,ctypes.c_int], None
    return result


def test_direct_host_matches_independent_cpu_bits(host):
    raw = cases(); actual = np.empty_like(raw)
    host.run_atan(raw, actual, len(raw))
    for target in ('numpy', 'cython'):
        assert_bits(actual, expected(raw,target),raw)


@pytest.mark.metal
def test_direct_metal_matches_independent_cpu_bits():
    mx = require_metal('Metal required')
    raw = cases()
    kernel = mx.fast.metal_kernel(name='direct_arctangent64', input_names=['x'],
        output_names=['out'], source='uint i=thread_position_in_grid.x;if(i<COUNT)out[i]=_eon_arctan64(x[i]);',
        header=binary64.MSL_HEADER+math64.MSL_MATH_HEADER+
               arctangent64.metal_arctangent_header({'arctan'}),
        compile_options={'math_mode':'safe'})
    output, = kernel(inputs=[mx.array(raw)], template=[('COUNT',len(raw))],
        grid=(len(raw),1,1), threadgroup=(128,1,1), output_shapes=[raw.shape],
        output_dtypes=[mx.uint64])
    mx.eval(output)
    for target in ('numpy','cython'):
        assert_bits(np.asarray(output), expected(raw,target),raw)


def test_selector_and_cpu_target_identity(monkeypatch):
    import brian2 as b
    from mlx_backend.model import DEFAULT_EQUATIONS, resolve_model
    from mlx_backend.compiler import compile_model
    monkeypatch.setitem(b.prefs, 'core.default_float_dtype', np.float64)
    assert arctangent64.metal_arctangent_header(()) == ''
    assert arctangent64.metal_arctangent_header({'sin','cos','tan'}) == ''
    models = []
    for target in ('numpy','cython','auto'):
        monkeypatch.setitem(b.prefs,'codegen.target',target)
        model = resolve_model({'eqs':DEFAULT_EQUATIONS+'observer:1\n',
            'eq_rst':'v=v_rst;g=0*mV;observer=arctan(observer+v/mV)'})
        models.append(model)
        assert '_eon_arctan64' in compile_model(model).reset_code
        assert model.reference_codegen_target == ('numpy' if target=='numpy' else 'cython')
    assert models[0].signature != models[1].signature
    assert models[1].signature != models[2].signature
    explicit = dict(models[1].integer_policy.identity)
    automatic = dict(models[2].integer_policy.identity)
    assert explicit.pop("requested_target") == "cython"
    assert automatic.pop("requested_target") == "auto"
    assert explicit == automatic
    assert models[1].reference_codegen_target == models[2].reference_codegen_target == "cython"


def model_configuration(divisor=1024):
    import brian2 as b
    from mlx_backend.model import DEFAULT_EQUATIONS, resolve_model
    return resolve_model({'eqs':DEFAULT_EQUATIONS+'observer:1\nphase:integer\n',
        'eq_th':'True','t_rfc':0*b.ms,
        'eq_rst':f'v=v_rst;g=0*mV;observer+=arctan((i-phase)/{divisor});phase+=1'})


@pytest.mark.metal
@pytest.mark.parametrize('target',['numpy','cython'])
def test_runtime_reset_observers_recording_and_continuation(monkeypatch,tmp_path,target):
    mx = require_metal('Metal required')
    import brian2 as b
    from mlx_backend import configured
    from component_cpu_reference import component_inputs
    from configured_reference import reference
    monkeypatch.setitem(b.prefs,'codegen.target',target)
    monkeypatch.setitem(b.prefs,'codegen.runtime.cython.cache_dir',str(tmp_path/'cython-default'))
    model=model_configuration(2)
    graph,stimulus=component_inputs(7,21,model)
    wanted=reference(graph,stimulus,model,record_state=True)
    prepared=configured.prepare(graph,stimulus,model)
    assert '_eon_arctan64' in prepared.program.reset_code
    for recording in (True,False):
        result,state=configured.run_prepared(prepared,stimulus,record=recording,
            return_state=True,chunk_steps=7,block_windows=1)
        assert_recorded_spikes_equal(result, wanted, recording)
        _assert_state(configured.host_state(state,prepared),wanted.state)
    # Read every step so arithmetic consumption and repeated continuation are observable.
    state=configured.initial_state(prepared)
    events=mx.array(np.zeros(1,np.uint8))
    for step in range(21):
        state,_=configured.advance(prepared,state,events,21,1)
        configured.evaluate_state(state)
        observed=configured.host_state(state,prepared)
        _assert_state(observed,{name:values[:,step] for name,values in wanted.traces.items()})


@pytest.mark.metal
@pytest.mark.parametrize('target',['numpy','cython'])
@pytest.mark.parametrize('operator',['>','>='])
def test_runtime_threshold_ties(monkeypatch,tmp_path,target,operator):
    mx = require_metal('Metal required')
    import brian2 as b
    from configured_reference import reference
    from component_cpu_reference import component_inputs
    from mlx_backend import configured
    from mlx_backend.model import DEFAULT_EQUATIONS,resolve_model
    monkeypatch.setitem(b.prefs,'codegen.target',target)
    monkeypatch.setitem(b.prefs,'codegen.runtime.cython.cache_dir',str(tmp_path/'cython-default'))
    model=resolve_model({'eqs':DEFAULT_EQUATIONS+'argument:1\n',
        'eq_th':f'(t<dt) or (arctan(argument){operator}cutoff)',
        'eq_rst':'v=v_rst;g=0*mV;argument=1','t_rfc':0*b.ms,
        'cutoff':float(np.arctan(1.))})
    graph,stimulus=component_inputs(1,7,model)
    wanted=reference(graph,stimulus,model)
    prepared=configured.prepare(graph,stimulus,model)
    assert '_eon_arctan64' in prepared.program.threshold_code
    result,state=configured.run_prepared(prepared,stimulus,return_state=True,chunk_steps=3)
    assert result.spikes.steps.size==(1 if operator=='>' else 7)
    assert_spikes_equal(result.spikes, wanted.spikes)
    _assert_state(configured.host_state(state,prepared),wanted.state)


@pytest.mark.metal
@pytest.mark.parametrize('target',['numpy','cython'])
def test_runtime_dynamic_linear_state_arctan(monkeypatch,tmp_path,target):
    mx = require_metal('Metal required')
    import brian2 as b
    from configured_reference import reference
    from component_cpu_reference import component_inputs
    from mlx_backend import configured
    from mlx_backend.model import DEFAULT_EQUATIONS,resolve_model
    monkeypatch.setitem(b.prefs,'codegen.target',target)
    monkeypatch.setitem(b.prefs,'codegen.runtime.cython.cache_dir',str(tmp_path/'cython-default'))
    model=resolve_model({'eqs':DEFAULT_EQUATIONS+'da/dt=-arctan(i+phase)*a/dt:1\nphase:integer\n',
        'eq_th':'True','eq_rst':'v=v_rst;g=0*mV;a+=1;phase+=1','t_rfc':0*b.ms})
    graph,stimulus=component_inputs(3,9,model)
    wanted=reference(graph,stimulus,model,record_state=True)
    prepared=configured.prepare(graph,stimulus,model)
    assert '_eon_arctan64' in prepared.program.state_code
    result,state=configured.run_prepared(prepared,stimulus,return_state=True,chunk_steps=3)
    _assert_state(configured.host_state(state,prepared),wanted.state)
    assert_spikes_equal(result.spikes, wanted.spikes)


@pytest.mark.metal
def test_runtime_actual_cpp_standalone(monkeypatch,tmp_path):
    mx = require_metal('Metal required')
    import brian2 as b
    from brian2.devices.device import get_device
    from component_cpu_reference import component_inputs
    from mlx_backend import configured
    monkeypatch.setitem(b.prefs,'codegen.target','auto')
    monkeypatch.setitem(b.prefs,'codegen.runtime.cython.cache_dir',str(tmp_path/'cython-default'))
    previous=get_device();b.set_device('cpp_standalone',build_on_run=False)
    b.device.reinit();b.device.activate(build_on_run=False)
    b.prefs.devices.cpp_standalone.extra_make_args_unix=['-j2']
    try:
        model=model_configuration(2);assert model.reference_codegen_target=='cpp'
        graph,stimulus=component_inputs(7,9,model)
        prepared=configured.prepare(graph,stimulus,model)
        clock=b.Clock(dt=model.dt_s*b.second)
        neurons=b.NeuronGroup(7,model.eqs,method='linear',threshold=model.eq_th,
            reset=model.eq_rst,refractory='rfc',namespace=model.brian_namespace(),clock=clock,dtype=np.float64)
        neurons.v=model.v_0*b.volt;neurons.g=0*b.volt;neurons.rfc=0*b.second
        monitor=b.SpikeMonitor(neurons)
        network=b.Network(neurons,monitor);network.schedule=list(model.schedule)
        network.run(9*model.dt_s*b.second,namespace={})
        b.device.build(directory=str(tmp_path/'cpp-default'),run=True,with_output=False)
        result,state=configured.run_prepared(prepared,stimulus,return_state=True)
        np.testing.assert_array_equal(result.spikes.neurons,np.asarray(monitor.i[:]))
        np.testing.assert_array_equal(result.spikes.steps,np.rint(np.asarray(monitor.t_[:])/model.dt_s).astype(np.int64))
        actual=configured.host_state(state,prepared)
        wanted={name:np.asarray(getattr(neurons,name+'_')[:]).copy().reshape(1,-1) for name in actual}
        _assert_state(actual,wanted)
    finally:b.set_device(previous)


@pytest.mark.metal
@pytest.mark.parametrize('target',['numpy','cython'])
def test_runtime_special_input_bits_from_timed_array(monkeypatch,tmp_path,target):
    mx = require_metal('Metal required')
    import brian2 as b
    from configured_reference import reference
    from component_cpu_reference import component_inputs
    from mlx_backend import configured
    from mlx_backend.model import DEFAULT_EQUATIONS,resolve_model
    monkeypatch.setitem(b.prefs,'codegen.target',target)
    monkeypatch.setitem(b.prefs,'codegen.runtime.cython.cache_dir',str(tmp_path/'cython-default'))
    raw=np.array([0,0x8000000000000000,1,0x8000000000000001,
        0x000fffffffffffff,0x800fffffffffffff,0x0010000000000000,
        0x7fefffffffffffff,0xffefffffffffffff,0x7ff0000000000000,
        0xfff0000000000000,0x7ff0000000000001,0xfff0000000000001,
        0x7ff8123456789abc,0xfff8123456789abc,
        *np.array([-.5,-.75,-1.,-4/3,-5/3,-2.,.5,.75,1.,4/3,5/3,2.]).view(np.uint64)],np.uint64)
    source=b.TimedArray(raw.view(np.float64).reshape(1,-1),dt=.1*b.ms)
    model=resolve_model({'eqs':DEFAULT_EQUATIONS+'observer:1\nchecksum:1\n',
        'eq_th':'True','t_rfc':0*b.ms,
        'eq_rst':'v=v_rst;g=0*mV;observer=arctan(source(t,i));checksum+=observer',
        'source':source})
    graph,stimulus=component_inputs(len(raw),3,model)
    wanted=reference(graph,stimulus,model)
    prepared=configured.prepare(graph,stimulus,model)
    _,state=configured.run_prepared(prepared,stimulus,record=False,return_state=True)
    _assert_state(configured.host_state(state,prepared),wanted.state)


def test_compile_flags_are_in_arctan_identity(monkeypatch):
    import brian2 as b
    from mlx_backend.compiler import compile_model,UnsupportedModelError
    from mlx_backend.model import DEFAULT_EQUATIONS,resolve_model
    monkeypatch.setitem(b.prefs,'codegen.target','cython')
    default=model_configuration()
    monkeypatch.setitem(b.prefs,'codegen.cpp.extra_compile_args',['-O3','-fno-fast-math'])
    strict=model_configuration()
    assert default.signature!=strict.signature
    assert default.reference_compile_args!=strict.reference_compile_args
    assert strict.reference_compile_args==('-O3','-fno-fast-math')
    assert '_eon_arctan64' in compile_model(strict).reset_code
