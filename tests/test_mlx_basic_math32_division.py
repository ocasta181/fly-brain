"""Existing safe-mode division reused without changing raw NaN rules."""
import numpy as np
import pytest

from mlx_backend import basic_math32, binary64, clang32, math64, operator64, stable_math64
from test_mlx_stable_math64 import division_cases, host


def test_only_float32_graphs_opt_into_safe_division():
    a=clang32.Node('argument','float64',value='a')
    b=clang32.Node('argument','float64',value='b')
    graph={'out':clang32.Node('fdiv','float64',(a,b))}
    used=set()
    assert 'd_div(a, b)' in clang32.render(graph,used)
    assert not used
    assert '_eon_float32_safe_div(a, b)' in clang32.render(graph,used,safe_float32_div=True)
    assert used=={'_float32_safe_div'}


@pytest.mark.metal
def test_safe_division_exact_hardware_and_canonical_nan_bits(host):
    import mlx.core as mx
    assert mx.metal.is_available()
    special=np.array([0,1,0x8000000000000000,0x7ff0000000000000,
        0xfff0000000000000,0x7ff0000000000001,0xfff0000000000023,
        0x7ff8123456789abc,0xfff8fedcba987654],np.uint64)
    raw=np.ascontiguousarray(np.concatenate((division_cases(),
        np.array([(a,b) for a in special for b in special],np.uint64))))
    columns=[np.ascontiguousarray(raw[:,i]) for i in range(2)]
    expected=np.empty(len(raw),np.uint64)
    host.reference_div(*columns,expected,len(raw))
    kernel=mx.fast.metal_kernel(name='float32_safe_division_proof',input_names=['a','b'],
        output_names=['actual','canonical'],source='''
        uint i=thread_position_in_grid.x;
        if(i<COUNT) {actual[i]=_eon_float32_safe_div(a[i],b[i]);canonical[i]=d_div(a[i],b[i]);}
        ''',header=binary64.MSL_HEADER+stable_math64._STABLE_DIV_HEADER
            +basic_math32.COMPILED_DIV_HEADER,compile_options={'math_mode':'safe'})
    actual,canonical=kernel(inputs=[mx.array(v) for v in columns],template=[('COUNT',len(raw))],
        grid=(len(raw),1,1),threadgroup=(128,1,1),
        output_shapes=[expected.shape,expected.shape],output_dtypes=[mx.uint64,mx.uint64])
    mx.eval(actual,canonical)
    actual,canonical=np.asarray(actual),np.asarray(canonical)
    # Every bit, including both-NaN payload and signaling priority, is unchanged.
    np.testing.assert_array_equal(actual,canonical)
    finite_result=~np.isnan(expected.view(np.float64))
    np.testing.assert_array_equal(actual[finite_result],expected[finite_result])
    # Invalid non-NaN operands retain the predefined canonical positive qNaN.
    invalid=~finite_result & ~np.isnan(raw.view(np.float64)).any(axis=1)
    np.testing.assert_array_equal(actual[invalid],np.full(invalid.sum(),0x7ff8000000000000,np.uint64))


@pytest.mark.metal
def test_expm1_exact_fma_reuse_keeps_independent_brian_bits():
    import mlx.core as mx
    from test_mlx_stable_math64 import stable_cases, reference_bits
    assert mx.metal.is_available()
    rng=np.random.default_rng(20261018)
    raw=np.unique(np.concatenate((stable_cases(),rng.integers(
        0,np.iinfo(np.uint64).max,32768,dtype=np.uint64))))
    expected=reference_bits('expm1',raw)
    outputs=[]
    for fma in ('stable_fma','basic32_fma_normal','_eon_operator_polynomial_fma'):
        header=binary64.MSL_HEADER+math64.MSL_MATH_HEADER
        if fma=='basic32_fma_normal':header+=basic_math32.NORMAL_FMA_HEADER
        if fma=='_eon_operator_polynomial_fma':header+=operator64._OPERATOR_FMA_HEADER
        header+=stable_math64.metal_stable_math_header({'expm1'},expm1_fma=fma)
        kernel=mx.fast.metal_kernel(name='float32_expm1_'+fma,
            input_names=['x'],output_names=['out'],
            source='uint i=thread_position_in_grid.x;if(i<COUNT)out[i]=d_expm1(x[i]);',
            header=header,compile_options={'math_mode':'safe'})
        out,=kernel(inputs=[mx.array(raw)],template=[('COUNT',len(raw))],
            grid=(len(raw),1,1),threadgroup=(128,1,1),
            output_shapes=[raw.shape],output_dtypes=[mx.uint64])
        mx.eval(out);outputs.append(np.asarray(out))
        # Every raw bit is required, including infinities, subnormals and NaNs.
        np.testing.assert_array_equal(outputs[-1],expected)
    for output in outputs[1:]:np.testing.assert_array_equal(output,outputs[0])
