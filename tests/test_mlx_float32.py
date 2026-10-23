"""Float32 native semantics, checked against independent Brian execution."""
from state_assertions import assert_array_bits_equal as assert_bits, assert_spikes_equal

import numpy as np
import pytest
from native_execution import require_metal
from pathlib import Path

from mlx_backend.model import resolve_model


@pytest.fixture
def brian(monkeypatch):
    b = pytest.importorskip("brian2")
    monkeypatch.setitem(b.prefs, "core.default_float_dtype", np.float32)
    monkeypatch.setitem(b.prefs, "core.default_integer_dtype", np.int32)
    monkeypatch.setitem(b.prefs, "codegen.target", "numpy")
    monkeypatch.setattr(b.defaultclock, "dt", .1*b.ms)
    monkeypatch.setitem(b.prefs, "codegen.runtime.cython.cache_dir", str(
        Path(__file__).resolve().parents[1] / "executions/cython-default-flags"))
    return b


def test_snapshot_keeps_clock_and_lastspike_binary64(brian):
    model = resolve_model({})
    assert model.state_dtype == "float32"
    for name in ("v", "g", "rfc"):
        assert model.analysis.variables[name].dtype == "float32"
    for name in ("t", "dt", "lastspike"):
        assert model.analysis.variables[name].dtype == "float64"


@pytest.mark.parametrize('target',['numpy','cython'])
def test_float32_snapshot_compilation_and_identity_survive_preference_changes(brian, monkeypatch,target):
    from mlx_backend.compiler import compile_model

    monkeypatch.setitem(brian.prefs,'codegen.target',target)
    model = resolve_model({})
    initial = compile_model(model)
    signature = model.signature
    if target=='cython':
        monkeypatch.setitem(brian.prefs,'codegen.cpp.extra_compile_args', ['-O3','-fno-fast-math'])
        different=resolve_model({})
        assert model.signature!=different.signature
        monkeypatch.setenv('CXX','false')
        monkeypatch.setenv('CFLAGS','-fno-fast-math')
    monkeypatch.setitem(brian.prefs, "core.default_float_dtype", np.float64)
    monkeypatch.setitem(brian.prefs, "core.default_integer_dtype", np.int64)
    monkeypatch.setitem(brian.prefs, "codegen.loop_invariant_optimisations", False)
    changed_target='cython' if target=='numpy' else 'numpy'
    monkeypatch.setitem(brian.prefs, "codegen.target", changed_target)
    monkeypatch.setitem(brian.prefs, "legacy.refractory_timing", True)
    later = compile_model(model)
    assert (later.state_code, later.scalar_code, later.reset_code, later.threshold_code) == (
        initial.state_code, initial.scalar_code, initial.reset_code, initial.threshold_code)
    assert model.signature == signature
    assert later.variables["v"].dtype == "float32"
    assert later.variables["lastspike"].dtype == "float64"
    assert brian.prefs.core.default_float_dtype is np.float64
    assert brian.prefs.core.default_integer_dtype is np.int64
    assert brian.prefs.legacy.refractory_timing is True
    assert brian.prefs.codegen.loop_invariant_optimisations is False
    assert brian.prefs.codegen.target == changed_target


def test_float32_codegen_and_optimisation_have_distinct_model_identity(brian, monkeypatch):
    baseline = resolve_model()
    monkeypatch.setitem(brian.prefs, "codegen.target", "cython")
    compiled = resolve_model()
    monkeypatch.setitem(brian.prefs, "codegen.loop_invariant_optimisations", False)
    unoptimised = resolve_model()
    assert len({baseline.signature, compiled.signature, unoptimised.signature}) == 3


@pytest.mark.metal
@pytest.mark.parametrize('target',['numpy','cython'])
def test_explicit_float32_namespace_keeps_target_scalar_promotions(brian,monkeypatch,target):
    from mlx_backend.model import DEFAULT_EQUATIONS
    from test_mlx_configured import inputs
    monkeypatch.setitem(brian.prefs,'codegen.target',target)
    model=resolve_model({'eqs':DEFAULT_EQUATIONS+'a : 1\n','eq_th':'t < dt',
        'eq_rst':'a=s*16777217;a-=16777216','s':np.float32(1.),'t_rfc':0*brian.ms})
    assert isinstance(model.source_namespace['s'],np.float32)
    ordinary=resolve_model({'eqs':DEFAULT_EQUATIONS+'a : 1\n','eq_th':'t < dt',
        'eq_rst':'a=s*16777217;a-=16777216','s':1.,'t_rfc':0*brian.ms})
    assert model.signature!=ordinary.signature
    _,_,_,expected=paired(model,s=inputs(model,targets=(),steps=2,trials=1))
    # Assignment rounds before the following in-place subtraction on both
    # targets, even when the scalar multiplication itself was binary64.
    np.testing.assert_array_equal(expected.state['a'],0)


@pytest.mark.metal
def test_numpy_float32_namespace_maximum_preserves_vector_precision(brian):
    from mlx_backend.model import DEFAULT_EQUATIONS
    from test_mlx_configured import inputs
    model=resolve_model({'eqs':DEFAULT_EQUATIONS+'a : 1\n','eq_th':'t < dt',
        'eq_rst':'a=a+smax; a=a+smax; a=a-smax','smax':np.float32(np.finfo(np.float32).max),
        't_rfc':0*brian.ms})
    _,_,_,expected=paired(model,s=inputs(model,targets=(),steps=2,trials=1))
    assert np.all(np.isposinf(expected.state['a']))


@pytest.mark.metal
def test_native_binary32_rounding_and_arithmetic():
    mx = require_metal('Metal is required')
    from mlx_backend.binary64 import MSL_HEADER as DOUBLE
    from mlx_backend.binary32 import MSL_HEADER as SINGLE

    rng = np.random.default_rng(1402)
    # Predefined exact bit requirements, including every rounding boundary
    # class, signed zero, subnormals, normal carry and overflow.
    bits = np.r_[np.array([0, 1, 2, 0x7fffff, 0x800000, 0x800001,
                          0x3f7fffff, 0x3f800000, 0x7f7fffff], np.uint32),
                 rng.integers(0, 0x7f800000, 256, dtype=np.uint32)]
    values = bits.view(np.float32).astype(np.float64)
    next_values = (bits+1).view(np.float32).astype(np.float64)
    finite = np.isfinite(next_values)
    midpoints = values[finite] + (next_values[finite]-values[finite])/2
    values = np.r_[values, midpoints, np.nextafter(midpoints, -np.inf),
                   np.nextafter(midpoints, np.inf),
                   [2.**-150, np.nextafter(2.**-150, np.inf),
                    2.**128-2.**103, np.inf]]
    values = np.r_[values, -values]
    kernel = mx.fast.metal_kernel(
        name="float32_exact_rounding", input_names=["values"], output_names=["rounded", "widened"],
        header=DOUBLE+SINGLE, source="""
        uint i = thread_position_in_grid.x;
        if(i >= COUNT) return;
        uint f = f_from_d(values[i]);
        rounded[i] = f;
        widened[i] = f_to_d(f);
        """, compile_options={"math_mode": "safe"})
    with np.errstate(over="ignore", invalid="ignore"):
        expected = values.astype(np.float32)
    rounded, widened = kernel(inputs=[mx.array(values.view(np.uint64))],
        template=[("COUNT", len(values))], grid=(len(values), 1, 1),
        threadgroup=(128, 1, 1), output_shapes=[values.shape]*2,
        output_dtypes=[mx.uint32, mx.uint64])
    np.testing.assert_array_equal(np.asarray(rounded), expected.view(np.uint32))
    np.testing.assert_array_equal(np.asarray(widened), expected.astype(np.float64).view(np.uint64))

    left = rng.integers(0, 0x7f800000, 2048, dtype=np.uint32)
    right = rng.integers(1, 0x7f800000, 2048, dtype=np.uint32)
    left[::2] |= 0x80000000
    right[::3] |= 0x80000000
    left[:8] = [0, 1, 0x7fffff, 0x800000, 0x3f800000, 0x80000000, 0x7f7fffff, 0xff7fffff]
    right[:8] = [1, 0x3f000000, 0x800001, 0x3f800000, 0x3f800001, 0x80000001, 0x3f000000, 0xbf800000]
    arithmetic = mx.fast.metal_kernel(
        name="float32_exact_operations", input_names=["a", "b"], output_names=["add", "sub", "mul", "div"],
        header=DOUBLE+SINGLE, source="""
        uint i=thread_position_in_grid.x;
        if(i >= COUNT) return;
        add[i]=f_add(a[i], b[i]); sub[i]=f_sub(a[i], b[i]);
        mul[i]=f_mul(a[i], b[i]); div[i]=f_div(a[i], b[i]);
        """, compile_options={"math_mode": "safe"})
    outputs = arithmetic(inputs=[mx.array(left), mx.array(right)], template=[("COUNT", len(left))],
        grid=(len(left), 1, 1), threadgroup=(128, 1, 1), output_shapes=[left.shape]*4,
        output_dtypes=[mx.uint32]*4)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        for output, operation in zip(outputs, (np.add, np.subtract, np.multiply, np.divide)):
            expected = operation(left.view(np.float32), right.view(np.float32))
            np.testing.assert_array_equal(np.asarray(output), expected.view(np.uint32))

    # Integer-to-float32 conversion must not double-round through float64.
    midpoint = 2**62 + 2**38
    integers = np.array([-(2**63), -midpoint-1, -midpoint, -midpoint+1,
                         0, 2**24+1, midpoint-1, midpoint, midpoint+1, 2**63-1], np.int64)
    convert = mx.fast.metal_kernel(name="float32_integer_rounding", input_names=["a"], output_names=["b"],
        header=DOUBLE+SINGLE, source="uint i=thread_position_in_grid.x; if(i < COUNT) b[i]=f_from_int(a[i]);",
        compile_options={"math_mode": "safe"})
    output, = convert(inputs=[mx.array(integers)], template=[("COUNT", len(integers))],
        grid=(len(integers), 1, 1), threadgroup=(32, 1, 1), output_shapes=[integers.shape], output_dtypes=[mx.uint32])
    # NumPy's ARM vector cast double-rounds the SIMD prefix via float64.
    # Scalar casts agree with independently compiled Brian Cython conversion.
    expected = np.array([np.float32(value) for value in integers], np.float32)
    np.testing.assert_array_equal(np.asarray(output), expected.view(np.uint32))


@pytest.mark.metal
@pytest.mark.parametrize('dtype',['float32','float64'])
def test_native_fused_rounding_matches_independent_system_fma(dtype):
    import ctypes
    mx=pytest.importorskip('mlx.core')
    from mlx_backend.binary64 import MSL_HEADER as DOUBLE
    from mlx_backend.binary32 import MSL_HEADER as SINGLE

    bits_type='uint32' if dtype=='float32' else 'uint64'
    ctype=ctypes.c_float if dtype=='float32' else ctypes.c_double
    function=getattr(ctypes.CDLL(None),'fmaf' if dtype=='float32' else 'fma')
    function.restype,function.argtypes=ctype,[ctype]*3
    limit=0x7f800000 if dtype=='float32' else 0x7ff0000000000000
    rng=np.random.default_rng(1402)
    bits=rng.integers(0,limit,(3,2048),dtype=bits_type)
    bits[0,::2] |= np.array(1<<(np.dtype(dtype).itemsize*8-1),dtype=bits_type)
    values=bits.view(dtype)
    exponent=12 if dtype=='float32' else 26
    precision=23 if dtype=='float32' else 52
    tiny=2.**(-100 if dtype=='float32' else -500)
    # Cancellation, overflow cancellation and a midpoint nudged by a term
    # too small for an intermediate binary64 sum to retain.
    special=np.array([[1+2.**-precision,1-2.**-precision,-1],
        [np.finfo(dtype).max,2,-np.finfo(dtype).max],
        [1+2.**-exponent,1+2.**-(exponent if dtype=='float32' else exponent+1),tiny],
        [1+2.**-exponent,1+2.**-(exponent if dtype=='float32' else exponent+1),-tiny],
        [0.,-0.,-0.]],dtype=dtype).T
    values=np.concatenate((special,values),axis=1)
    expected=np.array([function(*map(float,row)) for row in values.T],dtype=dtype)
    primitive='f_fma' if dtype=='float32' else 'clang_d_fma'
    kernel=mx.fast.metal_kernel(name='independent_fused_'+dtype,input_names=['a','b','c'],output_names=['out'],
        header=DOUBLE+SINGLE,source=f'uint i=thread_position_in_grid.x; if(i<COUNT)out[i]={primitive}(a[i],b[i],c[i]);',
        compile_options={'math_mode':'safe'})
    actual,=kernel(inputs=[mx.array(row.view(bits_type)) for row in values],template=[('COUNT',len(expected))],
        grid=(len(expected),1,1),threadgroup=(128,1,1),output_shapes=[expected.shape],
        output_dtypes=[mx.uint32 if dtype=='float32' else mx.uint64])
    np.testing.assert_array_equal(np.asarray(actual),expected.view(bits_type))


@pytest.mark.metal
@pytest.mark.parametrize("target", ["numpy", "cython"])
def test_native_float32_first_cpu_pair(brian, monkeypatch, target):
    mx = require_metal('Metal is required')
    from test_mlx_configured import graph, inputs, assert_matches_cpu

    monkeypatch.setitem(brian.prefs, "codegen.target", target)
    model = resolve_model({})
    assert model.state_dtype == "float32"
    assert model.float32_codegen == target
    c, s = graph(), inputs(model, steps=41, trials=1)
    # Keep the original direct run's default staging and recording sizes.
    assert_matches_cpu(c, s, model, chunk_steps=18432, block_windows=64)


@pytest.mark.metal
@pytest.mark.parametrize('tau,t_mbr',[(.005,.02),(.007,.024)])
@pytest.mark.parametrize('subnormal',[False,True])
def test_cython_float32_randomized_live_updater(brian,monkeypatch,tau,t_mbr,subnormal):
    import mlx.core as mx
    from dataclasses import replace
    from mlx_backend import configured
    from mlx_backend.models import Connectome
    from test_mlx_configured import inputs

    monkeypatch.setitem(brian.prefs,'codegen.target','cython')
    model=resolve_model({'tau':tau*brian.second,'t_mbr':t_mbr*brian.second,'eq_th':'False'})
    n,steps=64,9
    rng=np.random.default_rng(1402)
    initial={'v':rng.uniform(-.08,-.01,n).astype(np.float32),
             'g':rng.uniform(-.2,.2,n).astype(np.float32)}
    if subnormal:
        boundary=np.array([1,0x80000001,2,0x80000002,0x7fffff,0x807fffff,
                           0x800000,0x80800000,0,0x80000000],np.uint32)
        initial['g'][:len(boundary)]=boundary.view(np.float32)
    clock=brian.Clock(dt=model.dt_s*brian.second,name='float32_component_clock*')
    group=brian.NeuronGroup(n,model.eqs,method='linear',threshold=model.eq_th,
        reset=model.eq_rst,refractory='rfc',namespace=model.brian_namespace(),clock=clock,
        name='float32_component_neurons*')
    group.v=initial['v']*brian.volt;group.g=initial['g']*brian.volt;group.rfc=0*brian.second
    monitor=brian.StateMonitor(group,['v','g'],record=True,when='end')
    brian.Network(group,monitor).run(steps*clock.dt,namespace={})
    empty=np.empty(0,np.int32)
    c=Connectome(np.arange(n),empty,empty,empty,np.empty(0))
    s=inputs(model,steps=steps,trials=1,targets=())
    prepared=configured.prepare(c,s,model)
    state=configured.initial_state(prepared)
    state=replace(state,values=dict(state.values,
        **{name:mx.array(value.view(np.uint32)) for name,value in initial.items()},
        rfc=mx.zeros((n,),mx.uint32)))
    events=mx.zeros((1,),mx.uint8)
    for step in range(steps):
        state,_=configured.advance(prepared,state,events,steps,length=1)
        configured.evaluate_state(state)
        for name,value in configured.host_state(state,prepared).items():
            if name in initial:
                assert_bits(value[0],np.asarray(getattr(monitor,name+'_'))[:,step],f'{step} {name}')




def paired(model, *, c=None, s=None, silenced=(), chunk_steps=73):
    from configured_reference import reference
    from mlx_backend import configured
    from test_mlx_configured import graph, inputs

    c = graph() if c is None else c
    s = inputs(model, steps=131, trials=3, amplitudes=(60.125, -2.75, 72.5)) if s is None else s
    expected = reference(c, s, model, silenced)
    prepared = configured.prepare(c, s, model, silenced)
    result, state = configured.run_prepared(prepared, s, return_state=True,
        chunk_steps=chunk_steps, block_windows=3)
    assert "Apple" in result.device_name
    assert_spikes_equal(result.spikes, expected.spikes)
    for name, value in configured.host_state(state, prepared).items():
        assert_bits(value, expected.state[name], name)
    return result, state, prepared, expected


@pytest.mark.metal
@pytest.mark.parametrize("delay", [0, 1, 18, 33])
def test_numpy_float32_delays_trials_signed_coincident_and_fractional_inputs(brian, delay):
    model = resolve_model({"t_dly": delay*.1*brian.ms})
    paired(model)


@pytest.mark.metal
@pytest.mark.parametrize("params,dt", [
    ({"v_0": -.061, "v_rst": -.057, "v_th": -.043}, .2),
    ({"tau": .003, "t_mbr": .017, "t_rfc": .00029995}, .13),
    ({"w_syn": .000387, "f_poi": 123, "t_dly": .00335}, .23),
])
def test_numpy_float32_constants_clock_and_silencing(brian, params, dt):
    parameters = {k: v*(brian.volt if k.startswith("v_") or k == "w_syn" else brian.second)
                  if k != "f_poi" else v for k, v in params.items()}
    paired(resolve_model(parameters, dt_ms=dt), silenced=(2,), chunk_steps=17)


@pytest.mark.metal
def test_numpy_float32_assignment_rounding_threshold_and_typed_extra_state(brian):
    from mlx_backend.model import DEFAULT_EQUATIONS
    from test_mlx_configured import inputs

    model = resolve_model({
        "eqs": DEFAULT_EQUATIONS + "a : 1\nb : 1\ncounter : integer\nflag : boolean\n",
        "eq_th": "(t < dt) or (a > 1-epsilon)", "epsilon": 2.**-25,
        "eq_rst": "v=v_rst; g=0*mV; a=1; b=16777216; b+=1; b-=16777216; counter+=1; flag=not flag",
        "t_rfc": 0*brian.ms,
    })
    result, state, p, expected = paired(model, s=inputs(model, targets=(), steps=37))
    np.testing.assert_array_equal(result.spikes.steps, np.zeros(6, np.int64))
    np.testing.assert_array_equal(expected.state["b"], 1)
    np.testing.assert_array_equal(expected.state["counter"], 1)


@pytest.mark.metal
def test_numpy_float32_keeps_integer32_intermediate_rounding(brian):
    from mlx_backend.model import DEFAULT_EQUATIONS
    from test_mlx_configured import inputs

    model = resolve_model({"eqs": DEFAULT_EQUATIONS + "a : 1\nb : 1\ncounter : integer\n",
        "eq_th": "t < dt", "eq_rst": "counter+=2147483647; counter+=1; a=counter/3; counter-=1; b=counter",
        "t_rfc": 0*brian.ms})
    _, _, _, expected = paired(model, s=inputs(model, targets=(), steps=2, trials=1))
    np.testing.assert_array_equal(expected.state["counter"], np.iinfo(np.int32).max)
    assert np.all(expected.state["a"] < 0)


@pytest.mark.metal
def test_numpy_float32_integer64_vector_assignment(brian, monkeypatch):
    from mlx_backend.model import DEFAULT_EQUATIONS
    from mlx_backend.models import Connectome
    from test_mlx_configured import inputs

    monkeypatch.setitem(brian.prefs, "core.default_integer_dtype", np.int64)
    model = resolve_model({"eqs": DEFAULT_EQUATIONS + "a : 1\nk : integer\n",
        "eq_th": "t < dt", "eq_rst": "k=big+i; a=k", "big": 2**62+2**38+1,
        "t_rfc": 0*brian.ms})
    empty = np.empty(0, np.int32)
    c = Connectome(np.arange(10), empty, empty, empty, np.empty(0))
    paired(model, c=c, s=inputs(model, targets=(), steps=2, trials=1))


@pytest.mark.metal
@pytest.mark.parametrize('delay,count',[(0,7),(1,8),(18,9),(33,10)])
def test_numpy_float32_collective_reset_boundaries_and_arrivals(brian,monkeypatch,delay,count):
    from mlx_backend.model import DEFAULT_EQUATIONS
    from mlx_backend.models import Connectome
    from test_mlx_configured import inputs
    from mlx_backend import configured

    monkeypatch.setitem(brian.prefs,'core.default_integer_dtype',np.int64)
    model=resolve_model({'eqs':DEFAULT_EQUATIONS+'a : 1\nk : integer\n',
        'eq_th':'(t < dt) and (i < count)','eq_rst':'v=v_rst;g=0*mV;k=big+i;a=k',
        'count':count,'big':2**62+2**38+1,'t_rfc':0*brian.ms,'t_dly':delay*.1*brian.ms})
    sources=np.arange(count,dtype=np.int32)
    weights=np.linspace(-200.125,400.25,count)
    c=Connectome(np.arange(17),sources,np.full(count,16,np.int32),weights,weights*.275)
    s=inputs(model,targets=(),steps=delay+3,trials=2)
    result,state,p,expected=paired(model,c=c,s=s)
    assert p.program.collective_reset and p.window==1
    silent,other=configured.run_prepared(p,s,record=False,return_state=True)
    assert not silent.spikes.steps.size
    for name,value in configured.host_state(other,p).items():assert_bits(value,expected.state[name],name)


@pytest.mark.metal
@pytest.mark.parametrize('delay',[0,1,18,33])
def test_cython_float32_delays_recording_and_continuation(brian,monkeypatch,delay):
    monkeypatch.setitem(brian.prefs,'codegen.target','cython')
    test_numpy_float32_recording_off_and_partial_continuation(brian,delay)


@pytest.mark.metal
def test_cython_float32_typed_reset_and_sensitive_threshold(brian,monkeypatch):
    from mlx_backend.model import DEFAULT_EQUATIONS
    from test_mlx_configured import inputs
    monkeypatch.setitem(brian.prefs,'codegen.target','cython')
    model=resolve_model({'eqs':DEFAULT_EQUATIONS+'a : 1\nb : 1\ncounter : integer\nflag : boolean\n',
        'eq_th':'(t < dt) or (a > 1-epsilon)','epsilon':2.**-25,
        'eq_rst':'v=v_rst;g=0*mV;a=1;b=16777216;b+=1;b-=16777216;counter+=1;flag=not flag',
        't_rfc':0*brian.ms})
    result,_,_,expected=paired(model,s=inputs(model,targets=(),steps=3,trials=2))
    np.testing.assert_array_equal(expected.state['b'],0)
    np.testing.assert_array_equal(expected.state['counter'],1)
    np.testing.assert_array_equal(result.spikes.steps,np.zeros(6,np.int64))


@pytest.mark.metal
@pytest.mark.parametrize('optimise',[True,False])
def test_cython_float32_custom_functions_and_optimisation(brian,monkeypatch,optimise):
    monkeypatch.setitem(brian.prefs,'codegen.target','cython')
    test_numpy_float32_custom_dynamics_and_supported_functions(brian,monkeypatch,optimise)


@pytest.mark.metal
@pytest.mark.parametrize("delay", [0, 1, 18, 33])
def test_numpy_float32_recording_off_and_partial_continuation(brian, delay):
    import mlx.core as mx
    from mlx_backend import configured
    from test_mlx_configured import inputs

    model = resolve_model({"t_dly": delay*.1*brian.ms})
    s = inputs(model, steps=83, trials=2)
    result, whole, p, expected = paired(model, s=s)
    unrecorded, quiet = configured.run_prepared(p, s, record=False, return_state=True, chunk_steps=33)
    assert unrecorded.spikes.steps.size == 0 and unrecorded.timings["collection_s"] == 0
    partial = configured.initial_state(p, 2)
    events = mx.array(s.events.reshape(-1))
    recorded = []
    while partial.step < 83:
        begin = partial.step
        length = min((7, 1, 11, 3)[begin % 4], p.window-begin % p.window, 83-begin)
        partial, bits = configured.advance(p, partial, events, 83, length=length)
        configured.evaluate_state(partial)
        for bit in range(length):
            for row in np.flatnonzero(np.asarray(bits) & (1 << bit)):
                recorded.append((s.trial_indices[row//p.neurons], row%p.neurons, begin+bit))
    assert sorted(recorded, key=lambda x:(x[0], x[2], x[1])) == list(zip(
        result.spikes.trials, result.spikes.neurons, result.spikes.steps))
    for other in (quiet, partial):
        for name in whole.values:
            np.testing.assert_array_equal(np.asarray(other.values[name]), np.asarray(whole.values[name]))
        for name in ("history", "due"):
            left, right = getattr(other, name), getattr(whole, name)
            for key in left.keys() | right.keys():
                zeros = np.zeros(2*p.neurons, np.uint32)
                np.testing.assert_array_equal(np.asarray(left.get(key, zeros)), np.asarray(right.get(key, zeros)))


@pytest.mark.metal
@pytest.mark.parametrize("delay,arrivals", [(0, 7), (18, 3), (18, 5), (33, 4), (33, 7)])
def test_numpy_float32_ordered_synaptic_rounding_cache_and_overflow(brian, delay, arrivals):
    from mlx_backend.models import Connectome
    from test_mlx_configured import inputs

    counts = np.array([2.**24, 1., -2.**24, 3., -7., 11., .125])[:arrivals]
    order = np.arange(arrivals)[::-1]
    c = Connectome(np.arange(10), order.astype(np.int32), np.full(arrivals, 9, np.int32),
                   counts[order], counts[order]*.275)
    model = resolve_model({"w_syn": brian.volt, "t_dly": delay*.1*brian.ms})
    s = inputs(model, targets=tuple(range(arrivals)), steps=delay+2, trials=1)
    _, _, _, expected = paired(model, c=c, s=s)
    total = np.float32(0)
    for weight in counts:
        total = np.float32(total+np.float32(weight))
    assert expected.state["g"][0, 9] == total


@pytest.mark.metal
@pytest.mark.parametrize("optimise", [True, False])
def test_numpy_float32_custom_dynamics_and_supported_functions(brian, monkeypatch, optimise):
    from mlx_backend.model import DEFAULT_EQUATIONS

    monkeypatch.setitem(brian.prefs, "codegen.loop_invariant_optimisations", optimise)
    model = resolve_model({
        "eqs": DEFAULT_EQUATIONS + "da/dt=-a/tau : 1\nb : 1\n",
        "eq_rst": "v=v_rst; g=0*mV; a=sqrt(abs(v/mV)); b=clip(a, 1.25, 3.75)+sign(g/mV)",
        "eq_th": "(v > v_th) or ((a > 0) and (a < 2))",
    }, dt_ms=.13)
    paired(model)


@pytest.mark.metal
def test_numpy_float32_zero_horizon_public_adapter_and_saved_layout_identity(brian, monkeypatch):
    import mlx_engine
    from dataclasses import replace
    from test_mlx_configured import graph, inputs
    from mlx_backend import configured

    model = resolve_model({})
    c = graph()
    # A pinned synthetic graph exercises layout reuse without shared data.
    c = replace(c, pin="float32-test-private-layout")
    paired(model, c=c, s=inputs(model, steps=0))
    s = inputs(model, steps=53)
    result, _, prepared, expected = paired(model, c=c, s=s)
    monkeypatch.setitem(brian.prefs, "core.default_float_dtype", np.float64)
    monkeypatch.setitem(brian.prefs, "codegen.target", "cython")
    assert configured.prepare(c, s, model) is prepared
    actual = mlx_engine.execute(c, s, (), model=model)
    assert_spikes_equal(actual.spikes, expected.spikes)


@pytest.mark.metal
def test_numpy_float32_every_step_matches_independent_cpu_trace(brian):
    import mlx.core as mx
    from configured_reference import reference
    from mlx_backend import configured
    from test_mlx_configured import graph, inputs

    model = resolve_model({"t_dly": 3.3*brian.ms, "t_rfc": .29995*brian.ms}, dt_ms=.3)
    c, s = graph(), inputs(model, steps=67, trials=2, amplitudes=(60.125, -2.75, 72.5))
    expected = reference(c, s, model, record_state=True)
    p = configured.prepare(c, s, model)
    state = configured.initial_state(p, 2)
    events = mx.array(s.events.reshape(-1))
    for step in range(67):
        state, bits = configured.advance(p, state, events, 67, length=1)
        configured.evaluate_state(state)
        for name, value in configured.host_state(state, p).items():
            assert_bits(value, expected.traces[name][:, step], f"step {step} {name}")


@pytest.mark.parametrize("target", ["numpy", "cython"])
@pytest.mark.parametrize('typed_weight',[False,True])
def test_float32_reference_replay_matches_actual_frozen_poisson_input(brian, monkeypatch, tmp_path, target,typed_weight):
    import importlib.util
    import pandas as pd
    from configured_reference import reference
    from mlx_backend.models import Connectome, Stimulus

    monkeypatch.setitem(brian.prefs, "codegen.target", target)
    path = Path(__file__).resolve().parents[1] / "code/paper-phil-drosophila/model.py"
    spec = importlib.util.spec_from_file_location("float32_frozen_model", path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    completeness, connectivity = tmp_path / "neurons.csv", tmp_path / "edges.parquet"
    pd.DataFrame(index=np.arange(3)).to_csv(completeness)
    sources, destinations = np.array([1, 0, 0], np.int32), np.array([2, 1, 2], np.int32)
    counts = np.array([180.5, 300.25, -23.75])
    pd.DataFrame({"Presynaptic_Index": sources, "Postsynaptic_Index": destinations,
                  "Excitatory x Connectivity": counts}).to_parquet(connectivity)
    parameters = dict(original.default_params, r_poi=1/(.1*brian.ms), r_poi2=0*brian.Hz)
    if typed_weight:
        parameters.update(w_syn=brian.Quantity(np.float32(.000275),dim=brian.volt.dim),f_poi=np.float32(250.))
    model = resolve_model(parameters, dt_ms=.1)
    neurons, synapses, spikes = original.create_model(completeness, connectivity, parameters)
    channels, neurons = original.poi(neurons, [0, 0], [2], parameters)
    monitor = brian.StateMonitor(neurons, ["v", "g", "rfc", "lastspike", "not_refractory"], record=True, when="end")
    brian.Network(neurons, synapses, spikes, monitor, *channels).run(6.1*brian.ms)
    events = np.ones((1, 61, 3), np.uint8)
    events[:, :, 2] = 0
    stimulus = Stimulus(events, (0, 0, 2), (0., 0., 0.), (7,), 0, 0, None)
    c = Connectome(np.arange(3), sources, destinations, counts, counts*.275)
    actual = reference(c, stimulus, model, record_state=True)
    np.testing.assert_array_equal(actual.spikes.steps, np.rint(np.asarray(spikes.t_)/model.dt_s))
    np.testing.assert_array_equal(actual.spikes.neurons, np.asarray(spikes.i))
    for name in actual.state:
        assert_bits(actual.state[name][0], np.asarray(getattr(neurons, name+"_")[:]), name)
        assert_bits(actual.traces[name][0], np.asarray(getattr(monitor, name+"_")).T, name)
    from mlx_backend import configured
    prepared=configured.prepare(c,stimulus,model)
    native,state=configured.run_prepared(prepared,stimulus,return_state=True)
    assert_spikes_equal(native.spikes, actual.spikes)
    for name,value in configured.host_state(state,prepared).items():
        assert_bits(value,actual.state[name],name)


@pytest.mark.metal
@pytest.mark.parametrize('overflow',[False,True])
def test_cython_float32_signed_boolean_extension_and_integer_wrap(brian,monkeypatch,overflow):
    from mlx_backend.model import DEFAULT_EQUATIONS
    from test_mlx_configured import inputs
    monkeypatch.setitem(brian.prefs,'codegen.target','cython')
    reset=('flag=not flag; counter=2147483647; counter+=int(flag); a=counter' if overflow
           else 'flag=not flag; counter=-int(flag); a=counter')
    model=resolve_model({'eqs':DEFAULT_EQUATIONS+'a : 1\ncounter : integer\nflag : boolean\n',
                         'eq_th':'t < dt','eq_rst':reset,'t_rfc':0*brian.ms})
    _,_,_,expected=paired(model,s=inputs(model,targets=(),steps=2,trials=1))
    np.testing.assert_array_equal(expected.state['counter'],-2**31 if overflow else -1)


@pytest.mark.metal
@pytest.mark.parametrize('target',['numpy','cython'])
@pytest.mark.parametrize('delay',[0,1,18,33])
def test_independent_float32_neurons_preserve_all_phases_and_continuation(brian,monkeypatch,target,delay):
    import mlx.core as mx
    from mlx_backend import configured
    from mlx_backend.models import Connectome
    from test_mlx_configured import inputs

    monkeypatch.setitem(brian.prefs,'codegen.target',target)
    model=resolve_model({'t_dly':delay*.1*brian.ms,'t_rfc':.29995*brian.ms})
    empty=np.empty(0,np.int32)
    c=Connectome(np.arange(3),empty,empty,empty,np.empty(0))
    s=inputs(model,steps=67,trials=3,amplitudes=(60.125,-2.75,72.5))
    result,whole,p,_=paired(model,c=c,s=s)
    assert p.independent_neurons and p.window==32
    quiet,quiet_state=configured.run_prepared(p,s,record=False,return_state=True)
    again,again_state=configured.run_prepared(p,s,return_state=True,chunk_steps=79,block_windows=3)
    assert again.timings['jit_and_warmup_s']==0
    assert_spikes_equal(again.spikes, result.spikes)
    assert quiet.spikes.steps.size==0
    partial=configured.initial_state(p,3)
    events=mx.array(s.events.reshape(-1))
    seen=[]
    while partial.step<67:
        begin=partial.step
        length=min(7,p.window-begin%p.window,67-begin)
        partial,bits=configured.advance(p,partial,events,67,length=length)
        configured.evaluate_state(partial)
        for bit in range(length):
            seen.extend((s.trial_indices[row//3],row%3,begin+bit)
                        for row in np.flatnonzero(np.asarray(bits)&(1<<bit)))
    assert sorted(seen,key=lambda x:(x[0],x[2],x[1]))==list(zip(
        result.spikes.trials,result.spikes.neurons,result.spikes.steps))
    for other in (quiet_state,again_state,partial):
        assert not other.history and not other.due
        for name in whole.values:
            np.testing.assert_array_equal(np.asarray(other.values[name]),np.asarray(whole.values[name]))


@pytest.mark.metal
@pytest.mark.parametrize('target',['numpy','cython'])
def test_independent_float32_long_window_keeps_spike_word_boundaries(brian,monkeypatch,target):
    from mlx_backend import configured
    from mlx_backend.models import Connectome
    from test_mlx_configured import inputs
    monkeypatch.setitem(brian.prefs,'codegen.target',target)
    model=resolve_model({'eq_th':'(t < dt) or ((t > 31*dt) and (t < 34*dt)) or (t > 1023*dt)',
                         't_rfc':0*brian.ms,'t_dly':0*brian.ms})
    empty=np.empty(0,np.int32);c=Connectome(np.arange(3),empty,empty,empty,np.empty(0))
    s=inputs(model,targets=(),steps=1057,trials=2)
    _,small,p,expected=paired(model,c=c,s=s)
    for record in (True,False):
        result,state=configured.run_prepared(p,s,record=record,return_state=True)
        if record:
            assert_spikes_equal(result.spikes, expected.spikes)
        for name,value in state.values.items():
            np.testing.assert_array_equal(np.asarray(value),np.asarray(small.values[name]))
