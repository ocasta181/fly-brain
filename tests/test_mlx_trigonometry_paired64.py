"""Separate binary64 outputs preserve the actual selected CPU pair ABI."""
import ctypes
import json
import subprocess

import numpy as np
import pytest


def paired64_values():
    centers=np.array([0,1,0xfffffffffffff,0x10000000000000,
        0x3fe8000000000000,0x3fe921fb54442d18,0x412000013be57a40,
        0x7feffffffffffffe],np.uint64)
    boundary=np.r_[centers,centers+np.uint64(1),centers[1:]-np.uint64(1)]
    rng=np.random.default_rng(20261019)
    random=rng.integers(0,1<<64,16384,dtype=np.uint64)
    random=random[(random&np.uint64(0x7ff0000000000000))!=np.uint64(0x7ff0000000000000)]
    small=rng.uniform(-.8,.8,4096)
    quadrants=np.arange(-256,257,dtype=np.float64)*np.pi/2
    near=np.r_[quadrants,np.nextafter(quadrants,-np.inf),np.nextafter(quadrants,np.inf)]
    return np.r_[boundary,boundary^np.uint64(1<<63),random,small.view(np.uint64),near.view(np.uint64)]


@pytest.mark.metal
def test_paired64_actual_cpu_abi():
    mx=pytest.importorskip('mlx.core')
    from mlx_backend import binary64,math64
    from mlx_backend.trigonometry64 import metal_trigonometry_header
    class Pair(ctypes.Structure):
        _fields_=[('sine',ctypes.c_double),('cosine',ctypes.c_double)]
    function=getattr(ctypes.CDLL(None),'__sincos_stret')
    function.argtypes=[ctypes.c_double];function.restype=Pair
    values=paired64_values()
    expected=np.array([(p.sine,p.cosine) for p in (function(float(x)) for x in values.view(np.float64))],np.float64).view(np.uint64)
    kernel=mx.fast.metal_kernel(name='paired64_live_abi',input_names=['raw'],output_names=['out'],
        source='uint i=thread_position_in_grid.x; auto pair=_eon_trig64_sincos_scalar(raw[i]); out[2*i]=pair.sine; out[2*i+1]=pair.cosine;',
        header=binary64.MSL_HEADER+math64.MSL_MATH_HEADER+metal_trigonometry_header({'sin','cos'}),compile_options={'math_mode':'safe'})
    out,=kernel(inputs=[mx.array(values)],grid=(values.size,1,1),threadgroup=(128,1,1),
        output_shapes=[(values.size,2)],output_dtypes=[mx.uint64]);mx.eval(out)
    actual=np.asarray(out)
    mismatch=np.argwhere(actual!=expected)
    assert not len(mismatch),[(hex(int(values[i])),int(j),hex(int(actual[i,j])),hex(int(expected[i,j]))) for i,j in mismatch[:10]]


@pytest.mark.metal
@pytest.mark.parametrize('target',['numpy','cython'])
@pytest.mark.parametrize('case',['same','pure','negative','changed','sequential','conditional','shared','clock32','mixed','shared64'])
def test_original_paired64_separate_outputs(tmp_path,monkeypatch,target,case):
    import brian2 as b
    from original_api_protocol import check_original_api_runs, observe_run
    from original_api_reference import build_original_api_fixture
    reset={'same':'s=sin(x);c=cos(x)',
           'pure':'s=sin(x);c=cos(x)',
           'negative':'s=sin(x);c=cos(-x)',
           'changed':'s=sin(x);c=cos(x+.125)',
           'sequential':'s=sin(x);x=x+.125;c=cos(x)',
           'conditional':'s=sin(x);c=cos(x)',
           'shared':'s=sin(x);c=cos(x)',
           'clock32':'s=sin(x+t/ms);c=cos(x+t/ms)',
           'mixed':'s=sin(x+drive(t,i));c=cos(x+drive(t,i))',
           'shared64':'s=sin(t/ms);c=cos(t/ms)'}[case]
    def build(mode):
        parameters=dict(
            eqs='v:volt\ng:volt\nrfc:second\ns:1\nc:1\nx:1'+(' (shared)' if case=='shared' else ''),
            eq_th='sin(x)>0 and cos(x)>0' if case=='conditional' else 'True',
            eq_rst='v=v_rst;g=0*mV;'+reset+('' if case in ('shared','pure') else ';x=-x'),
            t_rfc=0*b.ms,t_dly=.2*b.ms)
        if case=='mixed':
            parameters['drive']=b.TimedArray(np.full((6,4),.0625,np.float64),dt=.1*b.ms)
        return build_original_api_fixture(tmp_path/str(mode == 'native'),parameters=parameters,
            initial_state={'x':.75 if case=='shared' else [-.75,-.125,.125,.75]},exc=(),exc2=(),silenced=())
    def exercise(fixture, mode):
        for duration in (0.,.2,.3):
            yield observe_run(fixture,duration,rng=True)
        codeobj=fixture.neurons.resetter['spike'].codeobj
        (tmp_path/('native-reset.txt' if mode == 'native' else 'cpu-reset.txt')).write_text(str(codeobj.code))
        if mode == 'cpu' and target=='cython':
            module=codeobj.compiled_code['run'].__file__
            imports=subprocess.check_output(['nm','-u',module],text=True)
            (tmp_path/'cpu-module.json').write_text(json.dumps({'module':module,'imports':imports}))
            if case in ('same','pure','negative','shared','shared64'):assert '___sincos_stret' in imports
            if case=='clock32':assert '___sincosf_stret' in imports
            if case=='mixed':assert '_sin\n' in imports and '_cos\n' in imports
    check_original_api_runs(build,exercise,monkeypatch,seed=953,target=target,
        float_dtype=np.float32 if case in ('mixed','clock32','shared64') else np.float64)
