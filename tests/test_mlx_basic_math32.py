"""Independent Brian execution of basic unary result widths on actual Metal."""
import numpy as np
import pytest

from trigonometry_checks import OriginalComparison


pytestmark = pytest.mark.metal


def compare_original(tmp_path, function, values, **options):
    import mlx.core as mx
    assert mx.metal.is_available()
    options = dict(dtype='float32', optimise=False, run_steps=(0, 1, 2), **options)
    OriginalComparison('numpy',function,values,tmp_path,**options).check_native()


@pytest.mark.parametrize('kind', ['float32', 'boolean', 'int8', 'int16', 'uint16', 'int32', 'int64'])
def test_exprel_array_dtype_and_branches(tmp_path, kind):
    if kind == 'float32':
        values = np.array([0., -0., .01, -.01, 1e-18, -1., 88., 718.], np.float32)
        options = {}
    elif kind == 'boolean':
        values = np.array([False, True, True, False])
        options = dict(phase_dtype='boolean')
    else:
        info = np.iinfo(kind)
        values = np.array([info.min, 0, 1, 2, min(89, info.max), info.max], kind)
        options = dict(phase_dtype='integer', integer_dtype=kind)
    compare_original(tmp_path, 'exprel', values, state_inputs=True, phase_symbol='basic_input',
        result_expression='exprel(basic_input)+.1', **options)


def test_exprel_half_composition(tmp_path):
    compare_original(tmp_path, 'exprel', np.array([False, True, True, False]),
        state_inputs=True, phase_dtype='boolean', phase_symbol='basic_flag',
        result_expression='exprel(log1p(basic_flag))+.1')


@pytest.mark.parametrize('value', [np.float32(0), np.float32(.1), np.float32(718)])
def test_exprel_namespace_scalar(tmp_path, value):
    compare_original(tmp_path, 'exprel', np.zeros(4), state_inputs=True,
        namespace_values={'basic_scalar': value}, result_expression='exprel(basic_scalar)+.1')


@pytest.mark.parametrize('word_bits', [1, 32])
def test_stable_recording_decodes_mixed_windows(word_bits):
    import mlx.core as mx
    from mlx_backend.configured import _collect_unit_windows
    rng = np.random.default_rng(321)
    n, trials, windows = 67, 3, 4
    ids = np.array([81, 3, 19], np.int64)
    starts = [0, 32, 64, 96] if word_bits == 32 else [0, 1, 2, 3]
    raw = rng.integers(0, 2**32 if word_bits == 32 else 2,
                       (windows, trials, n), dtype=np.uint32)
    raw[0] = 0
    if word_bits == 32:
        raw[-1] &= 0b111  # Last window is a partial continuation.
    parts = _collect_unit_windows([mx.array(word.reshape(-1)) for word in raw], starts, n, ids, word_bits)
    parts.sort(key=lambda part: int(part[0][0]))
    actual = np.stack([np.concatenate([p[i] for p in parts]) for i in range(3)], axis=1)
    expected = sorted((int(ids[t]), neuron, start+bit)
        for w, start in enumerate(starts) for t in range(trials) for bit in range(word_bits)
        for neuron in range(n) if raw[w,t,neuron] & (1 << bit))
    expected.sort(key=lambda x:(x[0],x[2],x[1]))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('neurons', [1, 31, 33, 67])
@pytest.mark.parametrize('word_bits', [1, 32])
def test_recording_words_preserve_trial_padding(neurons, word_bits):
    import mlx.core as mx
    from mlx_backend.configured import _unit_record_prefix
    trials, windows = 3, 5
    raw = np.random.default_rng(neurons).integers(0, 2**32,
        (trials, windows, neurons), dtype=np.uint32)
    if word_bits == 1:
        raw &= 1
    flags = ((raw[:, :, None, :] >> np.arange(word_bits, dtype=np.uint32)[None, None, :, None]) & 1)
    flags = flags.reshape(trials, -1)
    padding = (-flags.shape[1]) % 32
    flags = np.pad(flags, ((0, 0), (0, padding)))
    packed = np.packbits(flags.astype(np.uint8), axis=1, bitorder='little').view(np.uint32).reshape(-1)
    counts = flags.reshape(trials, -1, 32).sum(axis=2).reshape(-1)
    words, prefix = _unit_record_prefix(mx.array(raw.reshape(-1)), neurons, trials, windows, word_bits)
    np.testing.assert_array_equal(np.asarray(words), packed)
    np.testing.assert_array_equal(np.asarray(prefix), np.cumsum(counts))



@pytest.mark.parametrize('word_bits,pattern',[(1,'full'),(1,'missing'),(1,'sparse'),
    (32,'full'),(32,'missing'),(32,'partial'),(32,'sparse')])
def test_large_recording_hybrid_density_and_order(word_bits,pattern):
    import mlx.core as mx
    from mlx_backend.configured import _collect_unit_windows
    n,windows=(65537,8) if word_bits==1 else (8201,3)
    trials=2;ids=np.array([41,3],np.int64)
    raw=np.full((windows,trials,n),(1<<word_bits)-1,np.uint32)
    if pattern=='missing':raw[-1,-1,-1]^=np.uint32(1<<(word_bits-1))
    if pattern=='partial':raw[-1]&=np.uint32(7)
    if pattern=='sparse':
        raw[:]=0;raw[:,:,::101]=1
    starts=np.arange(windows)*(word_bits+7)
    parts=_collect_unit_windows([mx.array(word.reshape(-1)) for word in raw],
        starts,n,ids,word_bits)
    expected=[]
    for trial in np.argsort(ids):
        neurons=[];steps=[]
        for w,start in enumerate(starts):
            for bit in range(word_bits):
                selected=np.flatnonzero(raw[w,trial] & np.uint32(1<<bit))
                neurons.append(selected);steps.append(np.full(len(selected),start+bit,np.int64))
        expected.append((ids[trial],np.concatenate(neurons),np.concatenate(steps)))
    assert len(parts)==len(expected)
    for actual,(trial,neurons,steps) in zip(parts,expected):
        np.testing.assert_array_equal(actual[0],np.full(len(neurons),trial,np.int64))
        np.testing.assert_array_equal(actual[1],neurons)
        np.testing.assert_array_equal(actual[2],steps)


@pytest.mark.parametrize('function', ['exp', 'log', 'expm1', 'log1p', 'exprel'])
@pytest.mark.parametrize('integer_dtype', ['int16', 'uint16'])
def test_small_integer_float32_loops(tmp_path, function, integer_dtype):
    values = np.array([0, 1, 2, 3, 7, 11, 17, 33], integer_dtype)
    compare_original(tmp_path, function, values, state_inputs=True, phase_dtype='integer',
        phase_symbol='basic_input', integer_dtype=integer_dtype,
        result_expression=f'{function}(basic_input)+.1')


@pytest.mark.parametrize('function', ['exp', 'log', 'expm1', 'log1p', 'exprel'])
@pytest.mark.parametrize('scalar_dtype', [np.float16, np.float32, np.float64])
def test_namespace_scalar_result_composition(tmp_path, function, scalar_dtype):
    compare_original(tmp_path, function, np.zeros(4), state_inputs=True,
        namespace_values={'basic_scalar': scalar_dtype(.1)},
        result_expression=f'{function}(basic_scalar)+1+.1')


@pytest.mark.parametrize('function', ['exp', 'expm1', 'log1p', 'exprel'])
def test_namespace_scalar_result_alias(tmp_path, function):
    compare_original(tmp_path, function, np.zeros(4), state_inputs=True, phase_symbol='basic_input',
        namespace_values={'basic_scalar': np.float32(.1)},
        result_expression=f'{function}(basic_scalar);basic_input=sample;sample+=.1;sample=basic_input+.1')


@pytest.mark.parametrize('function', ['exp', 'log', 'expm1', 'log1p', 'exprel'])
def test_actual_float32_boundaries(tmp_path, function):
    raw = np.array([0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000,
        0x3f7fffff, 0x3f800000, 0x3f800001, 0xbf800000, 0xbf800001,
        0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
        0x7fc00123, 0xffc00123, 0x7f800001], np.uint32)
    values = np.r_[raw.view(np.float32), np.array([.01, .1, .2, .3, .7, 1.1, 2.1, 3.3], np.float32)]
    compare_original(tmp_path, function, values, state_inputs=True, phase_symbol='basic_input',
        result_expression=f'{function}(basic_input)+.1')


@pytest.mark.parametrize('function', ['exp', 'log', 'expm1', 'log1p'])
def test_scalar32_provider_matches_independent_numpy(function):
    import mlx.core as mx
    from mlx_backend import binary64, binary32, basic_math32
    rng = np.random.default_rng(20261018)
    raw = np.r_[np.array([0, 1, 2, 0x007fffff, 0x00800000, 0x3f7fffff,
        0x3f800000, 0x3f800001, 0x7f7fffff, 0x7f800000, 0x7fc00001, 0x7f800001], np.uint32),
        rng.integers(0, 0x7f800000, 4096, dtype=np.uint32)]
    raw = np.r_[raw, raw | np.uint32(0x80000000),
        rng.uniform(-120, 100, 2048).astype(np.float32).view(np.uint32),
        rng.uniform(-1.1, 2, 2048).astype(np.float32).view(np.uint32)]
    with np.errstate(all='ignore'):
        expected = getattr(np, function)(raw.view(np.float32)).view(np.uint32)
    kernel = mx.fast.metal_kernel(name='basic32_provider_'+function,
        input_names=['raw'], output_names=['out'],
        header=binary64.MSL_HEADER+binary32.MSL_HEADER+basic_math32.MSL_HEADER,
        source=f'uint i=thread_position_in_grid.x;if(i<N)out[i]=_eon_basic32_{function}(raw[i]);',
        compile_options={'math_mode': 'safe'})
    actual, = kernel(inputs=[mx.array(raw)], template=[('N', len(raw))],
        grid=(len(raw), 1, 1), threadgroup=(128, 1, 1), output_shapes=[raw.shape], output_dtypes=[mx.uint32])
    np.testing.assert_array_equal(np.asarray(actual), expected)
