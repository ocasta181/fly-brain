"""Independent CPU and GPU checks for simultaneous SSA output stores."""
import numpy as np
import pytest


@pytest.mark.metal
@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int64', 'bool'])
def test_output_swaps_preserve_incoming_words(dtype):
    import mlx.core as mx
    from mlx_backend.clang32 import Node, render
    types = {'float32': mx.uint32, 'float64': mx.uint64,
             'int64': mx.int64, 'bool': mx.bool_}
    storage = np.uint32 if dtype == 'float32' else np.uint64 if dtype == 'float64' else np.dtype(dtype)
    a = np.array([0, 1, 2, 3], dtype=storage)
    b = np.array([3, 2, 1, 0], dtype=storage)
    if dtype == 'float32':
        a = np.array([0x80000000, 0x7fc12345, 0x00000001, 0x3f800000], np.uint32)
    elif dtype == 'float64':
        a = np.array([0x8000000000000000, 0x7ff8123456789abc, 1, 0x3ff0000000000000], np.uint64)
    body = render({'x': Node('argument', dtype, value='y'),
                   'y': Node('argument', dtype, value='x')}, set(),
                  code_names={'x': 'left', 'y': 'right'})
    scalar = 'uint' if dtype == 'float32' else 'ulong' if dtype == 'float64' else 'long' if dtype == 'int64' else 'bool'
    kernel = mx.fast.metal_kernel(name='reset_alias_swap_'+dtype,
        input_names=['a', 'b'], output_names=['out'],
        source=f'uint i=thread_position_in_grid.x; {scalar} left=a[i],right=b[i];'
               + body + 'out[2*i]=left;out[2*i+1]=right;',
        compile_options={'math_mode': 'safe'})
    actual, = kernel(inputs=[mx.array(a), mx.array(b)], grid=(4, 1, 1),
        threadgroup=(4, 1, 1), output_shapes=[(4, 2)], output_dtypes=[types[dtype]])
    mx.eval(actual)
    np.testing.assert_array_equal(np.asarray(actual), np.column_stack((b, a)))


@pytest.mark.metal
@pytest.mark.parametrize('target,dtype,mutation', [
    ('cython', 'float32', 'alias_phase+=1/32'),
    ('cython', 'float64', 'alias_phase+=1/32'),
    ('numpy', 'float32', 'alias_phase+=1/32'),
    pytest.param('cython', 'float32', 'alias_phase=sin(alias_phase)',
                 id='scalar-sine-interaction')])
def test_reset_copy_precedes_mutation_original_api(tmp_path, monkeypatch, target, dtype, mutation):
    from trigonometry_checks import OriginalComparison
    values = np.array([.25, -.75, 2., -3.], dtype=dtype)
    options = dict(state_inputs=True, dtype=dtype, phase_symbol='alias_phase',
        result_expression='alias_phase; '+mutation,
        run_steps=(1, 2, 0, 3), cache_suffix='-reset-assignment-copy')
    comparison = OriginalComparison(target, 'sin', values, tmp_path, **options)
    observed = []
    if 'sin(' in mutation:
        from mlx_backend import configured
        factory = configured._neuron_kernel
        def traced_factory(*args, **kwargs):
            kernel = factory(*args, **kwargs)
            def launch(**launch_kwargs):
                observed.append(launch_kwargs['threadgroup'][0])
                return kernel(**launch_kwargs)
            return launch
        monkeypatch.setattr(configured, '_neuron_kernel', traced_factory)
    with comparison.native() as (_, _, device):
        assert device.native_steps == 6 and device.native_runs == 4
    if 'sin(' in mutation:
        assert observed and set(observed) == {len(values)}
