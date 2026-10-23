"""Shared execution checks for independent original-API math comparisons."""
from contextlib import contextmanager

import numpy as np
import pytest

from state_assertions import assert_array_bits_equal
from trigonometry_reference import original_reference


class OriginalComparison:
    """Keep original CPU execution separate from its matching native context."""
    def __init__(self, target, function, inputs, directory, **options):
        self.target, self.function = target, function
        self.inputs, self.directory, self.options = inputs, directory, options
        self.expected, self.cpu = original_reference(
            target, function, inputs, directory/'cpu', **options)

    @contextmanager
    def native(self, *, target=None, compare_report=True):
        from mlx_backend.brian_device import original_api

        with original_api(workers=False) as device:
            actual, native = original_reference(
                self.target if target is None else target, self.function,
                self.inputs, self.directory/'native', runtime_device=device,
                forbid_cpu_steps=True, **self.options)
            yield actual, native, device
        assert_array_bits_equal(actual, self.expected)
        if compare_report:
            for name in ('state_bits', 'steps', 'neurons'):
                assert native[name] == self.cpu[name], name

    def check_native(self, *, target=None, compare_report=True):
        with self.native(target=target, compare_report=compare_report):
            pass


def check_original_small_integer_loop(tmp_path,integer_dtype,function):
    mx=pytest.importorskip("mlx.core")
    if not mx.metal.is_available():pytest.skip("Metal required")
    from mlx_backend.numpy_half import Renderer,half_precision
    from mlx_backend import binary64,binary32,math64,binary16,operator32,operator64
    from mlx_backend.trigonometry32 import metal_trigonometry32_header
    symbol="phase_"+integer_dtype
    options=dict(state_inputs=True,phase_dtype="integer",phase_symbol=symbol,optimise=False,
        integer_dtype=integer_dtype,result_expression=function+"("+symbol+")+.1")
    info=np.iinfo(integer_dtype)
    inputs=np.arange(info.min,info.max+1,dtype=np.int64).astype(integer_dtype)
    expected,cpu=original_reference("numpy",function,inputs,tmp_path/"cpu",**options)
    np.save(tmp_path/"actual-eon-output.npy",expected)
    renderer=Renderer({symbol:np.dtype(integer_dtype)},{},set(),set(),"reset",
        half_ufuncs={function:(function+"_f32","_eon_trig32_"+{"sin":"sine","cos":"cosine","tan":"tangent"}[function]+"_scalar")})
    call=renderer.expression(function+"("+symbol+")")
    assert call.narrow is True and half_precision(call)==(np.dtype(integer_dtype).itemsize==1)
    result=renderer.expression(options["result_expression"])
    header=binary64.MSL_HEADER+binary32.MSL_HEADER+math64.MSL_MATH_HEADER
    header+=binary16.MSL_HEADER+binary16.MSL_F32_HEADER+metal_trigonometry32_header({function})
    # Half arithmetic uses the same exact scalar32 operators selected by
    # the production compiler's dependency collector.
    if 'operator_remainder_float32' in renderer.used_math:
        header+=operator64.MSL_REMAINDER_HEADER+operator32.MSL_REMAINDER_HEADER
    kernel=mx.fast.metal_kernel(name="small_integer_trig",input_names=["values"],output_names=["out"],
        header=header,source="uint i=thread_position_in_grid.x;if(i<N){long "+symbol+"=values[i];out[i]="+result.code+";}",
        compile_options={"math_mode":"safe"})
    (actual,)=kernel(inputs=[mx.array(inputs)],template=[("N",len(inputs))],grid=(len(inputs),1,1),
        threadgroup=(128,1,1),output_shapes=[inputs.shape],output_dtypes=[mx.uint64])
    mx.eval(actual);np.testing.assert_array_equal(np.asarray(actual),expected.view(np.uint64))



def check_original_half_operators(tmp_path,expression):
    mx=pytest.importorskip("mlx.core")
    if not mx.metal.is_available():pytest.skip("Metal required")
    values=np.array([False,True,True,False])
    options=dict(state_inputs=True,phase_dtype="boolean",phase_symbol="phase_ops",
        optimise=False,result_expression=expression,run_steps=(1,2,0,3),
        namespace_values={"exponent_offset":2.0001,"promote32":np.float32(1e5)})
    comparison=OriginalComparison('numpy', 'sin', values, tmp_path, **options)
    comparison.check_native()

