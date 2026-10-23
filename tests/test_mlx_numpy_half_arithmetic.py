"""Half arithmetic NaN ordering reuses the existing scalar32 operator family."""
from trigonometry_checks import OriginalComparison
import ast
import numpy as np
import pytest
from native_execution import require_metal


@pytest.mark.metal
@pytest.mark.parametrize("policy",["renderer","operator_primitives"])
def test_half_arithmetic_nan_and_zero_boundaries(policy):
    mx = require_metal('Metal required')
    from mlx_backend import binary64,binary32,binary16,math64,operator32,operator64
    from mlx_backend.numpy_half import Renderer,Expression
    corners=np.array([0,0x8000,0x3c00,0xbc00,0x7c00,0xfc00,
        0x7e01,0xfe02,0x7c01,0xfc02],np.uint16)
    values=np.stack(np.meshgrid(corners,corners,indexing="ij"),axis=-1).reshape(-1,2)
    operations=[(ast.Add(),np.add,"add"),(ast.Sub(),np.subtract,"sub"),
                (ast.Mult(),np.multiply,"mul"),(ast.Div(),np.divide,"div")]
    with np.errstate(all="ignore"):
        expected=np.stack([fn(*values.view(np.float16).T) for _,fn,_ in operations],axis=1).view(np.uint16)
    renderer=Renderer({}, {},set(),set(),"reset")
    args=[Expression(f"h_to_d(values[2*i+{column}])","float",False,True,half=True) for column in (0,1)]
    expressions=[]
    for operation,_,name in operations:
        if policy=="renderer":expressions.append(renderer._binary(*args,operation).code)
        else:expressions.append(f"h_f32_to_d(f_operator_{name}(h_d_to_f({args[0].code}),h_d_to_f({args[1].code})))")
    source="uint i=thread_position_in_grid.x;if(i<N){"+"".join(
        f"out[4*i+{column}]=h_from_d({code});" for column,code in enumerate(expressions))+"}"
    kernel=mx.fast.metal_kernel(name="half_arithmetic_"+policy,input_names=["values"],output_names=["out"],
        header=binary64.MSL_HEADER+binary32.MSL_HEADER+math64.MSL_MATH_HEADER+
            binary16.MSL_HEADER+binary16.MSL_F32_HEADER+operator64.MSL_REMAINDER_HEADER+operator32.MSL_REMAINDER_HEADER,
        source=source,compile_options={"math_mode":"safe"})
    (actual,)=kernel(inputs=[mx.array(values.astype(np.uint32))],template=[("N",len(values))],
        grid=(len(values),1,1),threadgroup=(128,1,1),output_shapes=[expected.shape],output_dtypes=[mx.uint32])
    mx.eval(actual)
    np.testing.assert_array_equal(np.asarray(actual),expected.astype(np.uint32))


@pytest.mark.metal
def test_original_numpy_half_subtract_keeps_right_nan_sign(tmp_path):
    mx = require_metal('Metal required')
    values=np.array([False,True,True,False])
    options=dict(state_inputs=True,phase_dtype="boolean",phase_symbol="phase_sub_nan",
        optimise=False,result_expression="arctan(phase_sub_nan)-sinh(arctan(phase_sub_nan)*negative_phase)",
        namespace_values={"negative_phase":np.uint64(0xfff8000000000123).view(np.float64)},
        run_steps=(1,2,0,3))
    comparison=OriginalComparison('numpy', 'arctan', values, tmp_path, **options)
    assert np.all(np.isnan(comparison.expected)) and np.all(np.signbit(comparison.expected))
    comparison.check_native()
