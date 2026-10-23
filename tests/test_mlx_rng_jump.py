"""Independent CPU checks of native MT jump-ahead and contiguous stream pieces."""
import numpy as np
import pytest
from mlx_backend.expression_rng import from_numpy_states,to_numpy_states

pytestmark = pytest.mark.metal


@pytest.mark.parametrize('position',[0,1,623,624])
def test_native_block_jumps_preserve_complete_numpy_snapshots(position):
    from mlx_backend.expression_rng_jump import jump_uniform_blocks
    cpu=np.random.RandomState(81);cpu.randn(1)
    snapshot=list(cpu.get_state());snapshot[2]=position;cpu.set_state(tuple(snapshot))
    actual=to_numpy_states(jump_uniform_blocks(from_numpy_states([cpu.get_state()]),3,20003))
    for part,got in enumerate(actual):
        if part:cpu.rand(312*20003)
        expected=cpu.get_state()
        np.testing.assert_array_equal(got[1],expected[1]);assert got[2:]==expected[2:]


def test_jump_convolution_preserves_arbitrary_accepted_mt_words():
    from mlx_backend.expression_rng_jump import jump_uniform_blocks
    cpus=[]
    for row,position in enumerate((0,1,623,624)):
        words=np.random.RandomState(103+row).randint(0,2**32,624,dtype=np.uint32)
        cpu=np.random.RandomState();cpu.set_state(('MT19937',words,position,1,.25+row));cpus.append(cpu)
    initial=from_numpy_states([cpu.get_state()for cpu in cpus])
    actual=to_numpy_states(jump_uniform_blocks(initial,3,37))
    for row,cpu in enumerate(cpus):
        for part in range(3):
            if part:cpu.rand(312*37)
            expected=cpu.get_state();got=actual[row*3+part]
            np.testing.assert_array_equal(got[1],expected[1]);assert got[2:]==expected[2:]


def test_parallel_uniforms_match_every_value_and_state_across_piece_boundaries():
    import mlx.core as mx
    from mlx_backend.expression_rng_jump import parallel_uniforms
    cpus=[np.random.RandomState(seed)for seed in (81,89)]
    for cpu,position in zip(cpus,(0,623)):
        cpu.randn(1);snapshot=list(cpu.get_state());snapshot[2]=position;cpu.set_state(tuple(snapshot))
    count=312*65000+17
    values,state=parallel_uniforms(from_numpy_states([cpu.get_state()for cpu in cpus]),count)
    mx.eval(values,state.words,state.position)
    actual=np.asarray(values).view(np.float64)
    snapshots=to_numpy_states(state)
    for row,cpu in enumerate(cpus):
        np.testing.assert_array_equal(actual[row],cpu.rand(count))
        expected=cpu.get_state();got=snapshots[row]
        np.testing.assert_array_equal(got[1],expected[1]);assert got[2:]==expected[2:]


@pytest.mark.parametrize('width,frames,parts',[(1,43,1),(3,43,8),(1009,43,8)])
def test_uniform_pool_values_and_each_checkpoint_match_cpu(width,frames,parts):
    import mlx.core as mx
    from mlx_backend.expression_rng_jump import uniform_pool
    cpus=[np.random.RandomState(seed)for seed in (81,89,17)]
    for cpu,position in zip(cpus,(0,1,623)):
        cpu.randn(1);snapshot=list(cpu.get_state());snapshot[2]=position;cpu.set_state(tuple(snapshot))
    pool=uniform_pool(from_numpy_states([cpu.get_state()for cpu in cpus]),width,frames,parts=parts)
    mx.eval(pool.values,pool.words,pool.positions)
    actual=np.asarray(pool.values).view(np.float64)
    words=np.asarray(pool.words);positions=np.asarray(pool.positions)
    for frame in range(frames):
        for row,cpu in enumerate(cpus):
            np.testing.assert_array_equal(actual[frame,row],cpu.rand(width))
            expected=cpu.get_state()
            np.testing.assert_array_equal(words[row,frame],expected[1])
            assert positions[row,frame]==expected[2]
    for got,cpu in zip(to_numpy_states(pool.state_at(frames-1)),cpus):
        assert got[2:]==cpu.get_state()[2:]


@pytest.mark.parametrize('parts',[1,8])
def test_device_limited_uniform_pool_keeps_only_requested_prefix(parts):
    import mlx.core as mx
    from mlx_backend.expression_rng_jump import uniform_pool
    counts=np.array([0,1,311,312,313,1009],np.uint32)
    cpus=[np.random.RandomState(81+row)for row in range(len(counts))]
    for row,cpu in enumerate(cpus):
        snapshot=list(cpu.get_state());snapshot[2]=(0,1,623,624,1,623)[row];cpu.set_state(tuple(snapshot))
    pool=uniform_pool(from_numpy_states([cpu.get_state()for cpu in cpus]),312,4,
                      parts=parts,counts=mx.array(counts))
    values=np.asarray(pool.values).transpose(1,0,2).reshape(len(counts),-1).view(np.float64)
    words=np.asarray(pool.words);positions=np.asarray(pool.positions)
    for row,(cpu,count)in enumerate(zip(cpus,counts)):
        np.testing.assert_array_equal(values[row,:count],cpu.rand(count))
        np.testing.assert_array_equal(values[row,count:],0)
        cpu=np.random.RandomState(81+row)
        snapshot=list(cpu.get_state());snapshot[2]=(0,1,623,624,1,623)[row];cpu.set_state(tuple(snapshot))
        for frame in range(int(count)//312):
            cpu.rand(312);expected=cpu.get_state()
            np.testing.assert_array_equal(words[row,frame],expected[1])
            assert positions[row,frame]==expected[2]
