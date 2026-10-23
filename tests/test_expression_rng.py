"""Independent legacy NumPy stream checks, including actual native draws."""
import numpy as np
import pytest

from mlx_backend.expression_rng import from_numpy_states, seeded, uniforms


@pytest.mark.metal
@pytest.mark.parametrize('seed', [0, 81, 2**32-1, np.array([81, 19, 2], np.uint32)])
def test_native_mt19937_uniforms_and_full_state_match_numpy(seed):
    import mlx.core as mx
    reference = np.random.RandomState(seed)
    state = seeded([seed])
    for count in (0, 1, 310, 2, 313, 1703):
        actual, state = uniforms(state, count)
        expected = reference.rand(count)
        np.testing.assert_array_equal(np.asarray(actual).view(np.float64)[0], expected)
        expected_state = reference.get_state()
        np.testing.assert_array_equal(np.asarray(state.words), expected_state[1])
        assert np.asarray(state.position)[0] == expected_state[2]


@pytest.mark.metal
def test_native_device_counts_independent_trials_and_checkpoint_replay():
    import mlx.core as mx
    seeds = [7, 89, 17]
    state = seeded(seeds)
    before = np.asarray(state.words).copy()
    counts = mx.array([0, 17, 1003], dtype=mx.uint32)
    first, branch = uniforms(state, 1003, counts)
    replay, replay_state = uniforms(state, 1003, counts)
    np.testing.assert_array_equal(np.asarray(first), np.asarray(replay))
    np.testing.assert_array_equal(np.asarray(branch.words), np.asarray(replay_state.words))
    np.testing.assert_array_equal(np.asarray(state.words), before)
    second, continued = uniforms(branch, 71)
    for row, (seed, count) in enumerate(zip(seeds, [0, 17, 1003])):
        cpu = np.random.RandomState(seed)
        np.testing.assert_array_equal(np.asarray(first).view(np.float64)[row, :count], cpu.rand(count))
        np.testing.assert_array_equal(np.asarray(first)[row, count:], 0)
        np.testing.assert_array_equal(np.asarray(second).view(np.float64)[row], cpu.rand(71))
        np.testing.assert_array_equal(np.asarray(continued.words).reshape(3,624)[row],cpu.get_state()[1])
        assert np.asarray(continued.position)[row] == cpu.get_state()[2]


def test_snapshot_validation_does_not_advance_global_numpy_state():
    before = np.random.get_state()
    for state in [(), ('PCG64', np.zeros(624,np.uint32),624,0,0),
                  ('MT19937', np.zeros(624,np.uint32),625,0,0)]:
        with pytest.raises(ValueError):
            from_numpy_states([state])
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


@pytest.mark.metal
@pytest.mark.parametrize('position', [1, 621, 623, 624])
def test_arbitrary_ambient_position_and_cached_normal_survive_uniform_draws(position):
    cpu=np.random.RandomState(81)
    cpu.randn()
    original=list(cpu.get_state())
    original[2]=position
    cpu.set_state(tuple(original))
    state=from_numpy_states([cpu.get_state()])
    actual,state=uniforms(state,1003)
    np.testing.assert_array_equal(np.asarray(actual).view(np.float64)[0],cpu.rand(1003))
    np.testing.assert_array_equal(np.asarray(state.words),cpu.get_state()[1])
    assert np.asarray(state.position)[0]==cpu.get_state()[2]
    assert np.asarray(state.has_gauss)[0]==cpu.get_state()[3]
    assert np.asarray(state.gauss).view(np.float64)[0]==cpu.get_state()[4]


@pytest.mark.metal
@pytest.mark.parametrize('seed', [0,81,2389,2**32-1])
def test_native_normals_interleaved_uniforms_and_cached_pair_match_numpy(seed):
    from mlx_backend.expression_rng import normals
    cpu=np.random.RandomState(seed)
    state=seeded([seed])
    for kind,count in [('normal',1),('uniform',3),('normal',3),('normal',0),
                       ('normal',1003),('uniform',1),('normal',156),('normal',1007)]:
        if kind=='normal':actual,state=normals(state,count);expected=cpu.randn(count)
        else:actual,state=uniforms(state,count);expected=cpu.rand(count)
        np.testing.assert_array_equal(np.asarray(actual).view(np.float64)[0],expected)
        expected_state=cpu.get_state()
        np.testing.assert_array_equal(np.asarray(state.words),expected_state[1])
        assert np.asarray(state.position)[0]==expected_state[2]
        assert np.asarray(state.has_gauss)[0]==expected_state[3]
        assert np.asarray(state.gauss).view(np.float64)[0]==expected_state[4]


@pytest.mark.metal
def test_native_normal_regression_for_cpu_log_rounding():
    from mlx_backend.expression_rng import normals
    actual,_=normals(seeded([81]),15794)
    np.testing.assert_array_equal(np.asarray(actual).view(np.float64)[0],
                                  np.random.RandomState(81).randn(15794))


@pytest.mark.metal
@pytest.mark.parametrize('position', [1,621,623,624])
def test_native_normal_device_counts_ambient_positions_and_snapshots(position):
    import mlx.core as mx
    from mlx_backend.expression_rng import normals
    cpus=[np.random.RandomState(s) for s in (81,89,17)]
    for cpu in cpus:
        cpu.randn(1)
        snapshot=list(cpu.get_state());snapshot[2]=position;cpu.set_state(tuple(snapshot))
    state=from_numpy_states([cpu.get_state() for cpu in cpus])
    counts=mx.array([0,1003,1],mx.uint32)
    actual,continued=normals(state,1003,counts)
    replay,_=normals(state,1003,counts)
    np.testing.assert_array_equal(np.asarray(actual),np.asarray(replay))
    for row,(cpu,count) in enumerate(zip(cpus,(0,1003,1))):
        np.testing.assert_array_equal(np.asarray(actual).view(np.float64)[row,:count],cpu.randn(count))
        np.testing.assert_array_equal(np.asarray(actual)[row,count:],0)
        np.testing.assert_array_equal(np.asarray(continued.words).reshape(3,624)[row],cpu.get_state()[1])
        assert np.asarray(continued.position)[row]==cpu.get_state()[2]
        assert np.asarray(continued.has_gauss)[row]==cpu.get_state()[3]
        assert np.asarray(continued.gauss).view(np.float64)[row]==cpu.get_state()[4]


@pytest.mark.metal
def test_native_normal_fused_operations_match_system_fma():
    import ctypes
    import mlx.core as mx
    from mlx_backend.binary64 import MSL_HEADER
    from mlx_backend.expression_rng_math import NORMAL_MATH_HEADER
    cpu=ctypes.CDLL('/usr/lib/libSystem.B.dylib').fma
    cpu.argtypes=[ctypes.c_double]*3;cpu.restype=ctypes.c_double
    rng=np.random.RandomState(721)
    a=rng.uniform(-2,2,20000);b=rng.uniform(-2,2,20000)
    c=rng.uniform(-2,2,20000)
    c[:5000]=-(a*b)[:5000]
    c[5000:10000]=-np.nextafter(a*b,np.inf)[5000:10000]
    # Exercise alignment, exact cancellation and subnormal residuals well
    # beyond the small coefficients used by the logarithm polynomial.
    wide_a=np.ldexp(rng.uniform(-2,2,20000),rng.randint(-510,511,20000))
    wide_b=np.ldexp(rng.uniform(-2,2,20000),rng.randint(-510,511,20000))
    wide_c=np.ldexp(rng.uniform(-2,2,20000),rng.randint(-1000,1001,20000))
    wide_c[:10000]=-(wide_a*wide_b)[:10000]
    wide_c[10000:15000]=-np.nextafter(wide_a*wide_b,np.inf)[10000:15000]
    a=np.r_[a,wide_a];b=np.r_[b,wide_b];c=np.r_[c,wide_c]
    a=np.r_[a,1+2**-52,1+2**-52,1.0,1+2**-52];b=np.r_[b,1-2**-52,1+2**-52,1.0,1+2**-52]
    c=np.r_[c,-1.0,-1.0,2**-53,2**-53-3*2**-106]
    expected=np.array([cpu(x,y,z) for x,y,z in zip(a,b,c)])
    kernel=mx.fast.metal_kernel(name='expression_normal_fma_conformance',
        input_names=['a','b','c'],output_names=['result'],
        source='uint i=thread_position_in_grid.x;if(i<SIZE)result[i]=expr_fma(a[i],b[i],c[i]);',
        header=MSL_HEADER+NORMAL_MATH_HEADER,compile_options={'math_mode':'safe'})
    actual,=kernel(inputs=[mx.array(v.view(np.uint64)) for v in (a,b,c)],
        template=[('SIZE',len(a))],grid=(len(a),1,1),threadgroup=(256,1,1),
        output_shapes=[(len(a),)],output_dtypes=[mx.uint64])
    np.testing.assert_array_equal(np.asarray(actual).view(np.float64),expected)


@pytest.mark.metal
@pytest.mark.parametrize('seed',[0,81,2389])
def test_native_poisson_all_algorithms_interleave_with_cached_normals(seed):
    import mlx.core as mx
    from mlx_backend.expression_poisson import poisson
    from mlx_backend.expression_rng import normals
    cpu=np.random.RandomState(seed);state=seeded([seed])
    _,state=normals(state,1);cpu.randn(1)
    lambdas=np.tile(np.array([0.,1e-12,.01,.3,1.,3.,9.999999999999998,10.,10.000000000000002,
                             17.,100.,1000.,1e8,1e16]),13)
    expected=cpu.poisson(lambdas)
    actual,state,errors=poisson(state,mx.array(lambdas.view(np.uint64)[None,:]))
    mx.eval(actual,errors,state.words,state.position)
    np.testing.assert_array_equal(np.asarray(errors),0)
    np.testing.assert_array_equal(np.asarray(actual)[0],expected)
    np.testing.assert_array_equal(np.asarray(state.words),cpu.get_state()[1])
    assert np.asarray(state.position)[0]==cpu.get_state()[2]
    last,state=normals(state,3)
    np.testing.assert_array_equal(np.asarray(last).view(np.float64)[0],cpu.randn(3))


@pytest.mark.metal
def test_native_poisson_rejects_invalid_vector_before_consuming_stream():
    import mlx.core as mx
    from mlx_backend.expression_poisson import poisson
    lambdas=np.array([[1.,-1.],[1.,float('nan')],[1.,float('inf')],[1.,1e19]])
    state=seeded([81,82,83,84])
    actual,after,errors=poisson(state,mx.array(lambdas.view(np.uint64)))
    np.testing.assert_array_equal(np.asarray(errors),1)
    np.testing.assert_array_equal(np.asarray(state.words),np.asarray(after.words))
    np.testing.assert_array_equal(np.asarray(state.position),np.asarray(after.position))


@pytest.mark.metal
def test_checkpoint_serialization_round_trip_preserves_cached_normal():
    from mlx_backend.expression_rng import normals,to_numpy_states
    state=seeded([81,82]);_,state=normals(state,3)
    snapshots=to_numpy_states(state)
    restored=from_numpy_states(snapshots)
    actual,_=normals(restored,17)
    for row,snapshot in enumerate(snapshots):
        cpu=np.random.RandomState();cpu.set_state(snapshot)
        np.testing.assert_array_equal(np.asarray(actual).view(np.float64)[row],cpu.randn(17))


@pytest.mark.metal
def test_cpu_log_rounding_at_table_and_exponent_boundaries():
    import math
    import mlx.core as mx
    from mlx_backend.binary64 import MSL_HEADER
    from mlx_backend.expression_rng_math import NORMAL_MATH_HEADER
    centers=np.array([(1+index/128)*2.**exponent
                      for exponent in (-106,-1,0,1,63)for index in range(129)])
    values=np.r_[np.nextafter(centers,0),centers,np.nextafter(centers,np.inf)]
    expected=np.array([math.log(value)for value in values])
    kernel=mx.fast.metal_kernel(name='expression_log_boundaries',
        input_names=['values'],output_names=['result'],
        source='uint i=thread_position_in_grid.x;if(i<SIZE)result[i]=expr_apple_log(values[i]);',
        header=MSL_HEADER+NORMAL_MATH_HEADER,compile_options={'math_mode':'safe'})
    actual,=kernel(inputs=[mx.array(values.view(np.uint64))],template=[('SIZE',len(values))],
        grid=(len(values),1,1),threadgroup=(256,1,1),output_shapes=[(len(values),)],output_dtypes=[mx.uint64])
    np.testing.assert_array_equal(np.asarray(actual).view(np.float64),expected)
