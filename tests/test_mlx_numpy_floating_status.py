"""Actual Eon floating-error semantics; native arithmetic, never CPU timesteps."""
from native_execution import forbid_cpu_code_objects, forbid_call, require_metal
import numpy as np
import pytest
from numpy_callback import RestoreCallback

b = pytest.importorskip('brian2')
mx = require_metal('Metal required', allow_module_level=True)

from mlx_backend.brian_device import original_api
from mlx_backend.model import DEFAULT_EQUATIONS, DEFAULT_RESET
from original_api_reference import build_original_api_fixture, cpu_runtime, original_runtime, assert_snapshot_equal

pytestmark = pytest.mark.metal


def options():
    return dict(parameters=dict(eqs=DEFAULT_EQUATIONS+'x:1\ny:1\np:1\n',
        eq_th='True', eq_rst=DEFAULT_RESET+';p=x**y', t_rfc=0*b.ms, w_syn=0*b.mV),
        exc=(), exc2=(), silenced=(),
        initial_state=dict(x=[.3,0.,3.,2.],y=[2.,-1.,2.,2.]))


def test_original_power_raise_preserves_prefix_and_repair(tmp_path, monkeypatch):
    with cpu_runtime():
        cpu = build_original_api_fixture(tmp_path/'cpu', **options())
        with np.errstate(divide='raise'):
            with pytest.raises(b.BrianObjectException) as raised:
                cpu.run_ms(.1)
        assert isinstance(raised.value.__cause__, FloatingPointError)
        message = str(raised.value.__cause__)
        failed = cpu.capture()
        with np.errstate(all='ignore'):
            repaired = cpu.run_ms(.1)
    with original_runtime(original_api) as device:
        native = build_original_api_fixture(tmp_path/'native', **options())
        forbid_cpu_code_objects(monkeypatch, 'Native policy used a CPU numerical timestep')
        with np.errstate(divide='raise'):
            with pytest.raises(b.BrianObjectException) as raised:
                native.run_ms(.1)
        assert isinstance(raised.value.__cause__, FloatingPointError)
        assert str(raised.value.__cause__) == message
        assert_snapshot_equal(native.capture(), failed)
        with np.errstate(all='ignore'):
            assert_snapshot_equal(native.run_ms(.1), repaired)
        assert device.native_runs == 1


def test_native_binary64_arithmetic_flags_match_independent_numpy():
    from mlx_backend.binary64 import MSL_HEADER
    from mlx_backend.math64 import MSL_MATH_HEADER
    from mlx_backend.numpy_error_policy import metal_header
    tiny = np.finfo(np.float64).tiny
    small = np.nextafter(np.float64(0), np.float64(1))
    maximum = np.finfo(np.float64).max
    cases = [(0,maximum,maximum),(0,np.inf,-np.inf),(0,small,small),
             (1,np.inf,np.inf),(1,maximum,-maximum),
             (2,tiny,.5),(2,small,.5),(2,small,1.),(2,maximum,2.),(2,np.inf,0.),
             (3,1.,0.),(3,0.,0.),(3,np.inf,np.inf),(3,small,2.),(3,tiny,2.),(3,maximum,.5),
             (2,np.nextafter(tiny,0.),np.nextafter(1.,2.))]
    functions = (np.add,np.subtract,np.multiply,np.divide)
    expected = []
    with RestoreCallback(np.geterrcall()):
        for op,a,z in cases:
            errors = []
            np.seterrcall(lambda message,flags: errors.append(flags))
            with np.errstate(all='call'):
                functions[op](np.array([a]),np.array([z]))
            expected.append(0 if not errors else errors[-1])
    kernel = mx.fast.metal_kernel(name='numpy_policy_binary64_flags',
        input_names=['operations','left','right'],output_names=['flags'],
        header=MSL_HEADER+MSL_MATH_HEADER+metal_header(),source='''
        uint i=thread_position_in_grid.x,op=operations[i];ulong a=left[i],b=right[i];
        ulong r=op==0?d_add(a,b):op==1?d_sub(a,b):op==2?d_mul(a,b):d_div(a,b);
        flags[i]=_eon_numpy_binary_status(op,a,b,r,64u);
        ''',compile_options={'math_mode':'safe'})
    result=kernel(inputs=[mx.array([c[0] for c in cases],mx.uint32),
        mx.array(np.array([c[1] for c in cases],np.float64).view(np.uint64)),
        mx.array(np.array([c[2] for c in cases],np.float64).view(np.uint64))],
        grid=(len(cases),1,1),threadgroup=(len(cases),1,1),
        output_shapes=[(len(cases),)],output_dtypes=[mx.uint32])[0]
    np.testing.assert_array_equal(np.asarray(result),expected)


def test_original_power_call_delivered_at_native_operation_boundary(tmp_path,monkeypatch):
    with RestoreCallback(np.geterrcall()):
        with cpu_runtime():
            cpu=build_original_api_fixture(tmp_path/'cpu',**options())
            expected_calls=[]
            np.seterrcall(lambda message,flags:expected_calls.append((message,flags)))
            with np.errstate(divide='call'):expected=cpu.run_ms(.1)
        with original_runtime(original_api):
            native=build_original_api_fixture(tmp_path/'native',**options())
            from brian2.codegen.runtime.numpy_rt.numpy_rt import NumpyCodeObject
            forbid_call(NumpyCodeObject, 'run', 'CPU numerical timestep', monkeypatch)
            calls=[]
            np.seterrcall(lambda message,flags:calls.append((message,flags)))
            with np.errstate(divide='call'):actual=native.run_ms(.1)
            assert calls==expected_calls==[('divide by zero',1)]
            assert_snapshot_equal(actual,expected)


@pytest.mark.parametrize('mode',['callback-error','policy-change','error-free-policy'])
def test_original_policy_action_prefix_and_changed_policy(tmp_path,monkeypatch,mode):
    previous=np.geterrcall()
    params=options()
    params['parameters']['eqs']+='q:1\n'
    params['parameters']['eq_rst']+=';q=x**y'
    def run(fixture):
        calls=[];error=None
        def callback(message,flags):
            calls.append((message,flags))
            if mode=='callback-error':raise RuntimeError('action failed')
            np.seterr(divide='ignore')
        np.seterrcall(callback)
        with np.errstate(divide='ignore' if mode=='error-free-policy' else 'call',
                         under='call' if mode=='error-free-policy' else 'ignore'):
            try:fixture.run_ms(.1)
            except b.BrianObjectException as value:error=(type(value.__cause__),str(value.__cause__))
        return fixture.capture(),calls,error
    with RestoreCallback(previous):
        with cpu_runtime():
            cpu=build_original_api_fixture(tmp_path/'cpu',**params)
            expected,calls,error=run(cpu)
        with original_runtime(original_api):
            native=build_original_api_fixture(tmp_path/'native',**params)
            from brian2.codegen.runtime.numpy_rt.numpy_rt import NumpyCodeObject
            forbid_call(NumpyCodeObject, 'run', 'CPU numerical timestep', monkeypatch)
            actual,actual_calls,actual_error=run(native)
            assert actual_calls==calls
            assert actual_error==error
            assert_snapshot_equal(actual,expected)


def test_exprel_unselected_branches_do_not_emit_native_errors(tmp_path,monkeypatch):
    params=options()
    params['parameters']['eq_rst']=DEFAULT_RESET+';p=exprel(x)'
    params['initial_state']['x']=[0.,1e-20,800.,-1.]
    with cpu_runtime():
        cpu=build_original_api_fixture(tmp_path/'cpu',**params)
        with np.errstate(all='raise'):expected=cpu.run_ms(.1)
    with original_runtime(original_api):
        native=build_original_api_fixture(tmp_path/'native',**params)
        from brian2.codegen.runtime.numpy_rt.numpy_rt import NumpyCodeObject
        forbid_call(NumpyCodeObject, 'run', 'CPU numerical timestep', monkeypatch)
        with np.errstate(all='raise'):actual=native.run_ms(.1)
        assert_snapshot_equal(actual,expected)


@pytest.mark.parametrize('function',['exp','exprel'])
def test_original_unary_actions_follow_native_operation_order(tmp_path,monkeypatch,function):
    params=options()
    params['parameters']['eq_rst']=DEFAULT_RESET+f';p={function}(x)'
    params['initial_state']['x']=([710.,-1000.,0.,1.] if function=='exp'
                                  else [0.,1e-20,710.,800.])
    previous=np.geterrcall()
    def run(fixture):
        calls=[]
        np.seterrcall(lambda text,flags:calls.append((text,flags)))
        with np.errstate(all='call'):
            value=fixture.run_ms(.1)
        return value,calls
    with RestoreCallback(previous):
        with cpu_runtime():
            cpu=build_original_api_fixture(tmp_path/'cpu',**params)
            expected,expected_calls=run(cpu)
        with original_runtime(original_api):
            native=build_original_api_fixture(tmp_path/'native',**params)
            forbid_cpu_code_objects(monkeypatch, 'CPU numerical timestep')
            actual,calls=run(native)
            assert calls==expected_calls
            assert_snapshot_equal(actual,expected)


def test_original_invariant_numpy_call_runs_at_each_boundary(tmp_path,monkeypatch):
    params=options()
    params['parameters'].update(eq_rst=DEFAULT_RESET+';p=exp(k)',k=np.float64(710.))
    previous=np.geterrcall()
    def run(fixture):
        calls=[]
        np.seterrcall(lambda text,flags:calls.append((text,flags,
            int(fixture.neurons.clock.variables['timestep'].get_value()[0]))))
        with np.errstate(all='call'):
            value=fixture.run_ms(.2)
        return value,calls
    with RestoreCallback(previous):
        with cpu_runtime():
            cpu=build_original_api_fixture(tmp_path/'cpu',**params)
            expected,expected_calls=run(cpu)
        with original_runtime(original_api):
            native=build_original_api_fixture(tmp_path/'native',**params)
            forbid_cpu_code_objects(monkeypatch, 'CPU numerical timestep')
            actual,calls=run(native)
            assert calls==expected_calls
            assert_snapshot_equal(actual,expected)


@pytest.mark.parametrize('case',['clock-scalar','float32-store','array-int','python-int','empty-scalar',
                                'timestep-cast','float64-widen'])
def test_original_cast_and_scalar_policy_boundaries(tmp_path,monkeypatch,case):
    import warnings
    params=options()
    policy=dict(all='raise')
    dtype=np.float64
    if case=='clock-scalar':
        params['parameters'].update(eq_rst=DEFAULT_RESET+';p=dt*k*z/second',k=1e308,z=1e308)
    elif case=='float32-store':
        params['parameters'].update(eq_rst=DEFAULT_RESET+';p=k',k=1e100)
        dtype=np.float32
    elif case=='array-int':
        params['parameters']['eq_rst']=DEFAULT_RESET+';p=floor(x)'
        params['initial_state']['x']=[np.nan,2.,3.,4.]
    elif case=='python-int':
        params['parameters'].update(eq_rst=DEFAULT_RESET+';p=int(k)',k=np.nan)
        policy=dict(all='ignore')
    elif case=='timestep-cast':
        params['parameters']['eq_rst']=DEFAULT_RESET+';p=timestep(x*ms,dt)'
        params['initial_state']['x']=[1e30,2.,3.,4.]
    elif case=='float64-widen':
        params['parameters'].update(eqs=DEFAULT_EQUATIONS+'x:second\n',
                                   eq_rst=DEFAULT_RESET+';lastspike=x')
        raw=np.array([0x7f812345,0x3f800000,0,0],np.uint32)
        # Attach dimensions through an ndarray view; unit multiplication
        # would quiet the signaling NaN before the intended widening cast.
        params['initial_state']=dict(x=b.Quantity(raw.view(np.float32),dim=b.second.dim,copy=False))
        dtype=np.float32
    else:
        params['parameters'].update(eq_th='False',eq_rst=DEFAULT_RESET+';p=exp(k)',k=710.)
    def observe(fixture):
        if case=='float64-widen':
            np.testing.assert_array_equal(fixture.neurons.variables['x'].get_value().view(np.uint32),raw)
        error=None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with np.errstate(**policy):
                try:fixture.run_ms(.1)
                except b.BrianObjectException as value:
                    error=(type(value.__cause__),str(value.__cause__))
        return fixture.capture(),error,[str(item.message) for item in caught
                                       if issubclass(item.category,RuntimeWarning)]
    with cpu_runtime():
        b.prefs.core.default_float_dtype=dtype
        expected=observe(build_original_api_fixture(tmp_path/'cpu',**params))
        if case in ('timestep-cast','float64-widen'):
            assert expected[1]==(FloatingPointError,'invalid value encountered in cast')
    with cpu_runtime():
        b.prefs.core.default_float_dtype=dtype
        with original_api(workers=False):
            fixture=build_original_api_fixture(tmp_path/'native',**params)
            forbid_cpu_code_objects(monkeypatch, 'CPU numerical timestep')
            actual=observe(fixture)
            assert actual[1:]==expected[1:]
            assert_snapshot_equal(actual[0],expected[0])
