"""One ufunc per shader: exact narrow results and independent NumPy flags."""
from native_execution import forbid_cpu_code_objects, require_metal
import json
import os
from pathlib import Path
import time

import numpy as np
import pytest
from numpy_callback import RestoreCallback

pytestmark = pytest.mark.metal


FUNCTIONS=('exp','log','expm1','log1p','sqrt','abs','sign','floor','ceil',
           'arcsin','arccos','arctan','log10','sinh','cosh','tanh','sin','cos','tan')


def _cases(dtype):
    wordtype=np.uint16 if dtype==np.float16 else np.uint32
    small=np.nextafter(dtype(0),dtype(1));tiny=np.finfo(dtype).tiny
    values=np.array([-np.inf,-1000,-100,-90,-10,-2,-1,-tiny,-small,-0.,0.,small,
        tiny,1e-20,1e-10,.25,1,2,10,90,100,1000,np.finfo(dtype).max,np.inf,np.nan],dtype=dtype)
    raw=values.view(wordtype)
    extra=([0x7c01,0x7e01,0xfc01,0x03ff,0x0400,0x0401] if dtype==np.float16 else
           [0x7f812345,0x7fc12345,0xff812345,0x007fffff,0x00800000,0x00800001])
    return np.concatenate((raw,np.array(extra,wordtype)))


def _reference(name,raw,dtype):
    expected=[];flags=[];previous=np.geterrcall()
    with RestoreCallback(previous):
        for word in raw:
            errors=[];np.seterrcall(lambda text,bits:errors.append(bits))
            with np.errstate(all='call'):
                out=getattr(np,name)(np.array([word],raw.dtype).view(dtype))
            expected.append(int(out.view(raw.dtype)[0]));flags.append(errors[-1] if errors else 0)
    return np.array(expected,raw.dtype),np.array(flags,np.uint32)


def _metal():
    mx = require_metal('Actual Metal required')
    return mx


@pytest.mark.parametrize('name',FUNCTIONS)
def test_narrow_ufunc_result_and_status(name):
    from mlx_backend import binary64,binary32,binary16,math64
    from mlx_backend.numpy_narrow_status import original_header,metal_header
    mx=_metal();started=time.monotonic()
    raw32,raw16=_cases(np.float32),_cases(np.float16)
    expected,flags32=_reference(name,raw32,np.float32)
    _,flags16=_reference(name,raw16,np.float16)
    header=(binary64.MSL_HEADER+binary32.MSL_HEADER+binary16.MSL_HEADER
            +binary16.MSL_F32_HEADER+math64.MSL_MATH_HEADER
            +original_header((name,))+metal_header((name,)))
    if name=='sign':header+=metal_header((name,))  # Shared helpers compose once.
    kernel=mx.fast.metal_kernel(name='numpy_narrow_status_'+name,input_names=['raw32','raw16'],
        output_names=['result','flags32','flags16'],header=header,source=f'''
        uint i=thread_position_in_grid.x,flags=0u;
        if(i>={len(raw32)}u)return;
        result[i]=_eon_numpy_{name}32_status(raw32[i],flags);
        flags32[i]=flags;flags16[i]=_eon_numpy_{name}16_flags(ushort(raw16[i]));
        ''',compile_options={'math_mode':'safe'})
    result,actual32,actual16=kernel(inputs=[mx.array(raw32),mx.array(raw16.astype(np.uint32))],
        grid=(len(raw32),1,1),threadgroup=(32,1,1),
        output_shapes=[raw32.shape]*3,output_dtypes=[mx.uint32]*3)
    result,actual32,actual16=map(np.asarray,(result,actual32,actual16))
    errors=[]
    for label,raw,got,want in (('result32',raw32,result,expected),('flags32',raw32,actual32,flags32),('flags16',raw16,actual16,flags16)):
        errors.extend(dict(scope=label,input=hex(int(raw[i])),actual=hex(int(got[i])),expected=hex(int(want[i]))) for i in np.flatnonzero(got!=want))
    report=dict(function=name,rows32=len(raw32),rows16=len(raw16),elapsed_seconds=time.monotonic()-started,mismatches=len(errors),first_mismatches=errors[:20])
    path=os.environ.get('EON_NUMPY_NARROW_PROBE')
    if path:Path(path+'-'+name+'.json').write_text(json.dumps(report,indent=2)+'\n')
    assert not errors,report


def test_narrow_cast_status_and_quiet_comparisons():
    from mlx_backend import binary64,binary32,binary16,math64
    from mlx_backend.numpy_narrow_status import metal_header
    mx=_metal()
    raw=np.array([0,0x3ff0000000000000,0x3810000000000000,0x380fffffffffffff,
        0x36a0000000000000,0x3690000000000000,0x47efffffe0000000,0x47ffffffe0000000,
        0x3f10000000000000,0x3f0fffffffffffff,0x3e70000000000000,0x3e60000000000000,
        0x40effc0000000000,0x40fffc0000000000,0x7ff0000000000000,
        0x7ff0000012345678,0xfff0000087654321,0x7ff8000012345678],np.uint64)
    expected=[];previous=np.geterrcall()
    with RestoreCallback(previous):
        for dtype in (np.float16,np.float32):
            row=[]
            for word in raw:
                calls=[];np.seterrcall(lambda text,bits:calls.append(bits))
                with np.errstate(all='call'):np.array([word],np.uint64).view(np.float64).astype(dtype)
                row.append(calls[-1] if calls else 0)
            expected.append(row)
    header=binary64.MSL_HEADER+binary32.MSL_HEADER+binary16.MSL_HEADER+math64.MSL_MATH_HEADER+metal_header(())
    kernel=mx.fast.metal_kernel(name='numpy_narrow_cast_flags',input_names=['raw'],output_names=['status'],header=header,
        source=f'uint i=thread_position_in_grid.x;if(i<{len(raw)}u){{status[i]=_eon_numpy_cast_status(raw[i],16u);status[i+{len(raw)}u]=_eon_numpy_cast_status(raw[i],32u);}}',compile_options={'math_mode':'safe'})
    actual=kernel(inputs=[mx.array(raw)],grid=(len(raw),1,1),threadgroup=(32,1,1),output_shapes=[(2,len(raw))],output_dtypes=[mx.uint32])[0]
    np.testing.assert_array_equal(np.asarray(actual),expected)


def test_reached_brian_int32_array_cast_status_uses_prefix_and_tail():
    from brian2.codegen.generators.numpy_generator import int_func
    from mlx_backend import binary64,binary32,binary16,math64
    from mlx_backend.numpy_narrow_status import metal_header
    mx=_metal();inputs=[];positions=[];lengths=[];widths=[];expected=[]
    with RestoreCallback(np.geterrcall()):
        for dtype in (np.float64,np.float32,np.float16):
            for length,position in ((1,0),(9,0),(9,8)):
                for value in (0.,2**31-1.,2**31,2**32,-2**31,-2**31-.75,
                              -2**31-1.,2**63,-2**63,np.inf,np.nan):
                    with np.errstate(all='ignore'):
                        vector=np.zeros(length,dtype);vector[position]=value
                    calls=[];np.seterrcall(lambda text,bits:calls.append(bits))
                    with np.errstate(all='call'):int_func(vector)
                    inputs.append(np.asarray(vector[position],np.float64).view(np.uint64).item())
                    positions.append(position);lengths.append(length);widths.append(np.dtype(dtype).itemsize*8)
                    expected.append(calls[-1] if calls else 0)
    header=binary64.MSL_HEADER+binary32.MSL_HEADER+binary16.MSL_HEADER+math64.MSL_MATH_HEADER+metal_header(())
    kernel=mx.fast.metal_kernel(name='numpy_brian_int32_cast_flags',input_names=['raw','positions','lengths','widths'],output_names=['status'],header=header,
        source=f'uint i=thread_position_in_grid.x;if(i<{len(inputs)}u)status[i]=_eon_numpy_int32_cast_status(raw[i],positions[i],lengths[i],widths[i]);',compile_options={'math_mode':'safe'})
    actual=kernel(inputs=[mx.array(np.asarray(inputs,np.uint64)),mx.array(positions,mx.int64),mx.array(lengths,mx.int64),mx.array(widths,mx.uint32)],
        grid=(len(inputs),1,1),threadgroup=(64,1,1),output_shapes=[(len(inputs),)],output_dtypes=[mx.uint32])[0]
    np.testing.assert_array_equal(np.asarray(actual),expected)


def test_persistent_integer_store_status_matches_selected_numpy_cast_loop():
    """Small witnesses for each distinct supported real/complex cast loop."""
    import warnings
    from mlx_backend import binary64,binary32,binary16,math64
    from mlx_backend.numpy_narrow_status import metal_header
    cases=(
        ('float64','int16',32,0,2**40),
        ('float64','int16',33,32,2**40),
        ('complex128','int16',8,0,2**40),
        ('complex128','int16',9,0,2**40),
        ('complex128','int16',9,8,2**40),
        ('float64','uint32',8,0,2**40),
        ('complex128','uint32',8,0,2**40),
        ('complex128','uint32',9,0,2**40),
        ('float32','uint16',16,0,-1.),
        ('float32','uint16',17,16,-1.),
        ('complex64','uint16',16,0,-1.),
        ('complex64','uint16',17,0,-1.),
        ('float16','uint16',16,0,-1.),
        ('complex64','uint8',9,0,-1.),
        ('float64','uint64',1,0,-.75),
        ('complex128','uint64',1,0,-1.),
        ('float64','uint64',1,0,2**64),
        ('complex128','int64',1,0,-2**63),
        ('float32','int64',1,0,2**63),
        ('float16','int32',1,0,np.inf),
        ('complex128','int32',1,0,np.nan),
    )
    raw=[];bits=[];unsigned=[];positions=[];lengths=[];source_bits=[];complex_sources=[];expected=[]
    with RestoreCallback(np.geterrcall()):
        for source,destination,length,position,value in cases:
            dtype=np.dtype(source);complex_source=dtype.kind=='c'
            with np.errstate(all='ignore'):
                values=np.zeros(length,dtype);values[position]=complex(value,1.) if complex_source else value
            calls=[];np.seterrcall(lambda text,flags:calls.append(flags))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',np.ComplexWarning)
                with np.errstate(all='call'):values.astype(destination)
            component=values.real if complex_source else values
            raw.append(np.asarray(component[position],np.float64).view(np.uint64).item())
            target=np.dtype(destination)
            bits.append(target.itemsize*8);unsigned.append(target.kind=='u')
            positions.append(position);lengths.append(length);source_bits.append(component.dtype.itemsize*8)
            complex_sources.append(complex_source);expected.append(calls[-1] if calls else 0)
    mx=_metal()
    header=binary64.MSL_HEADER+binary32.MSL_HEADER+binary16.MSL_HEADER+math64.MSL_MATH_HEADER+metal_header(())
    kernel=mx.fast.metal_kernel(name='numpy_persistent_integer_cast_flags',
        input_names=['raw','bits','unsigned_flags','positions','lengths','source_bits','complex_sources'],
        output_names=['status'],header=header,source=f'''
        uint i=thread_position_in_grid.x;if(i>={len(cases)}u)return;
        status[i]=_eon_numpy_integer_cast_status(raw[i],bits[i],bool(unsigned_flags[i]),
            positions[i],lengths[i],source_bits[i],bool(complex_sources[i]));
        ''',compile_options={'math_mode':'safe'})
    actual=kernel(inputs=[mx.array(np.asarray(raw,np.uint64)),mx.array(bits,mx.uint32),
        mx.array(unsigned,mx.uint32),mx.array(positions,mx.int64),mx.array(lengths,mx.int64),
        mx.array(source_bits,mx.uint32),mx.array(complex_sources,mx.uint32)],
        grid=(len(cases),1,1),threadgroup=(32,1,1),output_shapes=[(len(cases),)],output_dtypes=[mx.uint32])[0]
    np.testing.assert_array_equal(np.asarray(actual),expected)


def test_original_integer_store_invalid_preserves_cpu_prefix(tmp_path,monkeypatch):
    """Actual Eon q:integer/reset q=x, with one failing signed32 tail store."""
    import brian2 as b
    from mlx_backend.brian_device import original_api
    from mlx_backend.model import DEFAULT_EQUATIONS,DEFAULT_RESET
    from original_api_reference import build_original_api_fixture,cpu_runtime, original_runtime,assert_snapshot_equal
    options=dict(parameters=dict(eqs=DEFAULT_EQUATIONS+'x:1\nq:integer\n',
        eq_th='True',eq_rst=DEFAULT_RESET+';q=x',t_rfc=0*b.ms,w_syn=0*b.mV),
        exc=(),exc2=(),silenced=(),initial_state=dict(x=[0.,2**31,1.,-2**31-1.],q=[7,7,7,7]))
    with cpu_runtime(integer_dtype=np.int32):
        cpu=build_original_api_fixture(tmp_path/'cpu',**options)
        with np.errstate(invalid='raise'):
            with pytest.raises(b.BrianObjectException) as raised:cpu.run_ms(.1)
        assert isinstance(raised.value.__cause__,FloatingPointError)
        message=str(raised.value.__cause__);expected=cpu.capture()
    with original_runtime(original_api, integer_dtype=np.int32):
        from mlx_backend import configured
        prepare=configured.prepare
        identities={}
        def observe_prepare(*args,**kwargs):
            prepared=prepare(*args,**kwargs)
            identities.setdefault(prepared.model.numpy_policy_ir,set()).add(prepared.model.signature)
            return prepared
        monkeypatch.setattr(configured,'prepare',observe_prepare)
        native=build_original_api_fixture(tmp_path/'native',**options)
        forbid_cpu_code_objects(monkeypatch, 'CPU numerical timestep in native store test')
        with np.errstate(invalid='raise'):
            with pytest.raises(b.BrianObjectException) as raised:native.run_ms(.1)
        assert isinstance(raised.value.__cause__,FloatingPointError)
        assert str(raised.value.__cause__)==message
        assert_snapshot_equal(native.capture(),expected)
        # Integer execution binding must preserve the policy mode, so
        # ordered replay cannot reuse its speculative fused program.
        assert identities.get(False)
        assert identities.get(True)
        assert identities[False].isdisjoint(identities[True])
