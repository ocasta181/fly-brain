"""Every value and complete stream state against independent NumPy CPU."""
import numpy as np
import pytest
from mlx_backend.expression_rng import from_numpy_states,to_numpy_states

pytestmark = pytest.mark.metal


@pytest.mark.parametrize('candidate_count',[None,1])
@pytest.mark.parametrize('position',[0,1,623,624])
def test_parallel_normals_rejections_cache_masks_and_fallback(position,candidate_count):
    import mlx.core as mx
    from mlx_backend.expression_rng_parallel import parallel_normals
    counts=np.array([0,1,2,7,4099],np.uint32)
    cpus=[np.random.RandomState(81+row)for row in range(len(counts))]
    for row,cpu in enumerate(cpus):
        if row%2:cpu.randn(1)
        snapshot=list(cpu.get_state());snapshot[2]=position;cpu.set_state(tuple(snapshot))
    values,end=parallel_normals(from_numpy_states([cpu.get_state()for cpu in cpus]),
        int(counts.max()),mx.array(counts),candidate_count=candidate_count)
    mx.eval(values,end.words,end.position,end.has_gauss,end.gauss)
    actual=np.asarray(values).view(np.float64)
    for row,(cpu,count,got)in enumerate(zip(cpus,counts,to_numpy_states(end))):
        np.testing.assert_array_equal(actual[row,:count],cpu.randn(count))
        np.testing.assert_array_equal(actual[row,count:],0)
        expected=cpu.get_state()
        np.testing.assert_array_equal(got[1],expected[1]);assert got[2:]==expected[2:]


@pytest.mark.parametrize('lam',[0.,1e-12,.3,3.,9.999999999999998,10.,17.,100.,1e8,1e16])
@pytest.mark.parametrize('raw_capacity',[None,1])
def test_parallel_constant_poisson_products_ptrs_and_exhaustion(lam,raw_capacity):
    import mlx.core as mx
    from mlx_backend.expression_poisson_parallel import constant_poisson
    counts=np.array([0,1,2,7,1033],np.uint32)
    cpus=[np.random.RandomState(81+row)for row in range(len(counts))]
    for row,cpu in enumerate(cpus):
        cpu.randn(1);snapshot=list(cpu.get_state());snapshot[2]=(0,1,623,624,623)[row]
        cpu.set_state(tuple(snapshot))
    values,end,errors=constant_poisson(from_numpy_states([cpu.get_state()for cpu in cpus]),
        int(counts.max()),lam,mx.array(counts),raw_capacity=raw_capacity)
    np.testing.assert_array_equal(np.asarray(errors),0)
    actual=np.asarray(values)
    for row,(cpu,count,got)in enumerate(zip(cpus,counts,to_numpy_states(end))):
        np.testing.assert_array_equal(actual[row,:count],cpu.poisson(lam,count))
        np.testing.assert_array_equal(actual[row,count:],0)
        expected=cpu.get_state()
        np.testing.assert_array_equal(got[1],expected[1]);assert got[2:]==expected[2:]


def test_public_normal_path_parallel_jump_and_tail_match_cpu():
    import mlx.core as mx
    from mlx_backend.expression_rng import normals
    cpus=[np.random.RandomState(81),np.random.RandomState(89)]
    cpus[0].randn(1);cpus[1].rand(137)
    count=1024*1024+1
    values,end=normals(from_numpy_states([cpu.get_state()for cpu in cpus]),count,
        mx.array([count,0],mx.uint32))
    actual=np.asarray(values).view(np.float64)
    np.testing.assert_array_equal(actual[0],cpus[0].randn(count))
    np.testing.assert_array_equal(actual[1],0)
    for got,cpu in zip(to_numpy_states(end),cpus):
        expected=cpu.get_state()
        np.testing.assert_array_equal(got[1],expected[1]);assert got[2:]==expected[2:]


def test_large_normal_capacity_uses_each_device_count_and_preserves_cached_pairs():
    import mlx.core as mx
    from mlx_backend.expression_rng import normals
    counts=np.array([0,1,13,131072],np.uint32)
    cpus=[np.random.RandomState(81+row)for row in range(len(counts))]
    for row,cpu in enumerate(cpus):
        if row%2:cpu.randn(1)
        snapshot=list(cpu.get_state());snapshot[2]=(0,1,623,624)[row];cpu.set_state(tuple(snapshot))
    actual,end=normals(from_numpy_states([cpu.get_state()for cpu in cpus]),131072,mx.array(counts))
    values=np.asarray(actual).view(np.float64)
    for row,(cpu,count,got)in enumerate(zip(cpus,counts,to_numpy_states(end))):
        np.testing.assert_array_equal(values[row,:count],cpu.randn(count))
        np.testing.assert_array_equal(values[row,count:],0)
        expected=cpu.get_state()
        np.testing.assert_array_equal(got[1],expected[1]);assert got[2:]==expected[2:]


@pytest.mark.parametrize('width,frames,candidate_count',[(1,43,None),(3,43,1),(101,5,1),(1009,43,None),(65537,17,None)])
def test_normal_pool_values_and_every_full_checkpoint_match_cpu(width,frames,candidate_count):
    import mlx.core as mx
    from mlx_backend.expression_rng_parallel import normal_pool
    cpus=[np.random.RandomState(seed)for seed in (81,89,17)]
    for cpu,position in zip(cpus,(0,1,623)):
        cpu.randn(1);snapshot=list(cpu.get_state());snapshot[2]=position;cpu.set_state(tuple(snapshot))
    pool=normal_pool(from_numpy_states([cpu.get_state()for cpu in cpus]),width,frames,
                     candidate_count=candidate_count)
    actual=np.asarray(pool.values).view(np.float64)
    for frame in range(frames):
        for row,(cpu,got)in enumerate(zip(cpus,to_numpy_states(pool.state_at(frame)))):
            np.testing.assert_array_equal(actual[frame,row],cpu.randn(width))
            expected=cpu.get_state()
            np.testing.assert_array_equal(got[1],expected[1]);assert got[2:]==expected[2:]


@pytest.mark.parametrize('lam',[0.,.3,3.,9.99,10.,17.])
@pytest.mark.parametrize('raw_capacity',[None,1])
def test_poisson_pool_values_and_every_full_checkpoint_match_cpu(lam,raw_capacity):
    from mlx_backend.expression_poisson_parallel import poisson_pool
    cpus=[np.random.RandomState(seed)for seed in (81,89,17)]
    for cpu,position in zip(cpus,(0,1,623)):
        cpu.randn(1);snapshot=list(cpu.get_state());snapshot[2]=position;cpu.set_state(tuple(snapshot))
    width=101;frames=5
    pool=poisson_pool(from_numpy_states([cpu.get_state()for cpu in cpus]),width,frames,lam,
                     raw_capacity=raw_capacity)
    actual=np.asarray(pool.values)
    for frame in range(frames):
        for row,(cpu,got)in enumerate(zip(cpus,to_numpy_states(pool.state_at(frame)))):
            np.testing.assert_array_equal(actual[frame,row],cpu.poisson(lam,width))
            expected=cpu.get_state()
            np.testing.assert_array_equal(got[1],expected[1]);assert got[2:]==expected[2:]


@pytest.mark.parametrize('lam',[3.,17.])
def test_large_poisson_pool_coarse_checkpoint_restore_matches_cpu(lam):
    from mlx_backend.expression_poisson_parallel import poisson_pool
    cpu=np.random.RandomState(81);cpu.randn(1)
    snapshot=list(cpu.get_state());snapshot[2]=623;cpu.set_state(tuple(snapshot))
    width=65537;frames=17
    pool=poisson_pool(from_numpy_states([cpu.get_state()]),width,frames,lam)
    actual=np.asarray(pool.values)
    for frame in range(frames):
        np.testing.assert_array_equal(actual[frame,0],cpu.poisson(lam,width))
        got=to_numpy_states(pool.state_at(frame))[0];expected=cpu.get_state()
        np.testing.assert_array_equal(got[1],expected[1]);assert got[2:]==expected[2:]
