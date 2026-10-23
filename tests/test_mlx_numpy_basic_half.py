"""NumPy's basic unary functions retain their actual half-precision results."""
from native_execution import forbid_cpu_steps as guard_cpu_steps, require_metal
import numpy as np
import pytest

from trigonometry_checks import OriginalComparison


pytestmark=pytest.mark.metal


@pytest.mark.parametrize('function',['exp','log','expm1','log1p'])
@pytest.mark.parametrize('dtype',['float32','float64'])
def test_boolean_basic_ufunc_retains_half_precision(tmp_path,function,dtype,temporary=False):
    mx = require_metal('Metal required')
    values=np.array([False,True,True,False])
    assert getattr(np,function)(values).dtype==np.float16
    options=dict(state_inputs=True,phase_dtype='boolean',phase_symbol='state_flag',
        dtype=dtype,optimise=False,result_expression=f'{function}(state_flag)+.1',run_steps=(0,1,2))
    if temporary:
        options['result_expression']=(f'0;predicate=t>=.1*ms;thread={function}(predicate);sample=thread+.1'
            if temporary=='clock' else f'{function}(state_flag);thread=sample;sample=thread+.1')
    OriginalComparison('numpy',function,values,tmp_path,**options).check_native()


@pytest.mark.parametrize('function',['exp','log','expm1','log1p'])
@pytest.mark.parametrize('dtype',['float32','float64'])
def test_half_temporary_uses_backend_data_name(tmp_path,function,dtype):
    test_boolean_basic_ufunc_retains_half_precision(tmp_path,function,dtype,temporary=True)


@pytest.mark.parametrize('function',['exp','log','expm1','log1p'])
@pytest.mark.parametrize('dtype',['float32','float64'])
def test_half_scalar_temporary_uses_backend_data_name(tmp_path,function,dtype):
    test_boolean_basic_ufunc_retains_half_precision(tmp_path,function,dtype,temporary='clock')


@pytest.mark.parametrize('function',['exp','log','expm1','log1p'])
def test_all_half_encodings_execute_on_metal(function):
    mx = require_metal('Metal required')
    from mlx_backend import binary64,binary16
    from mlx_backend.numpy_half import Renderer,Expression
    # Shuffle the whole domain so an indexing/stride error cannot match by
    # accidentally returning the upload's original row ordering.
    raw=np.random.default_rng(65536).permutation(65536).astype(np.uint16)
    inputs=raw.view(np.float16)
    with np.errstate(all='ignore'):
        expected=getattr(np,function)(inputs).astype(np.float64).view(np.uint64)
    renderer=Renderer({}, {},set(),set(),'reset')
    argument=Expression('h_to_d(raw[i])','float',False,True,half=True)
    result=renderer._half_ufunc(function,argument)
    arrays=renderer.half_input_arrays
    source='uint i=thread_position_in_grid.x;if(i<N){out[i]='+result+';}'
    kernel=mx.fast.metal_kernel(name='numpy_basic_half_'+function,
        input_names=['raw',*arrays],output_names=['out'],source=source,
        header=binary64.MSL_HEADER+binary16.MSL_HEADER,
        compile_options={'math_mode':'safe'})
    observed,=kernel(inputs=[mx.array(raw),*(mx.array(a.values.view(np.uint64)) for a in arrays.values())],
        template=[('N',len(raw))],grid=(len(raw),1,1),threadgroup=(128,1,1),
        output_shapes=[raw.shape],output_dtypes=[mx.uint64])
    mx.eval(observed)
    np.testing.assert_array_equal(np.asarray(observed),expected)
    assert len(arrays)==1 and sum(a.values.nbytes for a in arrays.values())==512*1024


@pytest.mark.parametrize('function',['exp','log','expm1','log1p'])
@pytest.mark.parametrize('integer_dtype',['int8','uint8'])
def test_actual_integer8_basic_ufunc_continuation(tmp_path,function,integer_dtype):
    mx = require_metal('Metal required')
    info=np.iinfo(integer_dtype)
    inputs=np.arange(info.min,info.max+1,dtype=np.int64).astype(integer_dtype)
    options=dict(state_inputs=True,phase_dtype='integer',phase_symbol='half_source',
        dtype='float64',integer_dtype=integer_dtype,optimise=False,
        result_expression=f'{function}(half_source)+.1',run_steps=(1,0,2))
    OriginalComparison('numpy',function,inputs,tmp_path,**options).check_native()


@pytest.mark.parametrize('record',[False,True])
def test_half_tables_share_timed_array_poisson_namespace(monkeypatch,tmp_path,record):
    from contextlib import nullcontext
    import brian2 as b
    from mlx_backend.brian_device import original_api
    from mlx_backend.model import DEFAULT_EQUATIONS
    from original_api_reference import (build_original_api_fixture,cpu_runtime,
        assert_snapshot_equal,capture_rng_state,assert_rng_state_equal)
    mx = require_metal('Metal required')
    monkeypatch.setitem(b.prefs,'codegen.loop_invariant_optimisations',False)
    expected=[];expected_rng=[]
    for native in (False,True):
        with cpu_runtime(target='numpy'):
            drive=b.TimedArray(np.array([.125,-.25,.5,-1.,2.]),dt=.1*b.ms)
            params=dict(eqs=DEFAULT_EQUATIONS+'flag:boolean\na:1\nb:1\nc:1\nd:1\n',
                eq_th='True',t_rfc=0*b.ms,drive=drive,r_poi=4900*b.Hz,r_poi2=3200*b.Hz,
                eq_rst='v=v_rst;g=drive(t)*mV;a=exp(sqrt(flag));b=log1p(sinh(flag));'
                       'c=expm1(arctan(flag));d=log(cosh(flag));flag=not flag')
            with original_api(workers=False) if native else nullcontext() as device:
                fixture=build_original_api_fixture(tmp_path/str(native),parameters=params,
                    initial_state={'flag':[False,True,True,False]},exc=(0,0),exc2=(2,),silenced=())
                if not record:
                    fixture.network.remove(fixture.monitor)
                    fixture.monitor=b.SpikeMonitor(fixture.neurons,record=False)
                    fixture.network.add(fixture.monitor)
                if native:
                    guard_cpu_steps(fixture.network, monkeypatch, 'A CPU timestep executed', runners=lambda : (fixture.neurons.state_updater, fixture.neurons.thresholder['spike'], fixture.neurons.resetter['spike'], *fixture.synapses._pathways), patch_preparation=True)
                for index,duration in enumerate((0.,.15,.25)):
                    if index==2:drive.values[2:]+=.25
                    observed=fixture.run_ms(duration);rng=capture_rng_state()
                    if native:
                        assert_snapshot_equal(observed,expected[index])
                        assert_rng_state_equal(rng,expected_rng[index])
                        prepared=device._sessions[fixture.network].prepared
                        assert len(prepared.program.input_arrays)==5
                        assert len(prepared.input_arrays)==1
                    else:expected.append(observed);expected_rng.append(rng)
