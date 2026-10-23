"""Independent NumPy clip boundary contract for transient half arrays."""
from trigonometry_checks import OriginalComparison
import numpy as np
import pytest
from native_execution import require_metal


@pytest.mark.metal
def test_numpy_half_clip_array_boundaries():
    mx = require_metal('Metal required')
    from brian2.core.functions import DEFAULT_FUNCTIONS
    from mlx_backend import binary64,binary16
    from mlx_backend.numpy_half import Renderer,Expression
    corners=np.array([0,0x8000,0x3c00,0xbc00,0x7c00,0xfc00,
        0x7e01,0xfe02,0x7c01,0xfc02],np.uint16)
    values=np.stack(np.meshgrid(corners,corners,corners,indexing="ij"),axis=-1).reshape(-1,3)
    actual_function=DEFAULT_FUNCTIONS["clip"].implementations["numpy"].get_code(None)
    with np.errstate(all="ignore"):
        expected=actual_function(*values.view(np.float16).T).view(np.uint16)
    renderer=Renderer({}, {},set(),set(),"reset")
    for column,name in enumerate(("value","low","high")):
        renderer.temporaries[name]=Expression(f"h_to_d(values[3*i+{column}])","float",False,True,half=True)
    result=renderer.expression("clip(value,low,high)")
    kernel=mx.fast.metal_kernel(name="half_clip_array_boundaries",input_names=["values"],output_names=["out"],
        header=binary64.MSL_HEADER+binary16.MSL_HEADER,
        source="uint i=thread_position_in_grid.x;if(i<N)out[i]=h_from_d("+result.code+");",
        compile_options={"math_mode":"safe"})
    (actual,)=kernel(inputs=[mx.array(values.astype(np.uint32))],template=[("N",len(values))],
        grid=(len(values),1,1),threadgroup=(128,1,1),output_shapes=[expected.shape],output_dtypes=[mx.uint32])
    mx.eval(actual)
    np.testing.assert_array_equal(np.asarray(actual),expected.astype(np.uint32))


@pytest.mark.metal
def test_original_numpy_half_clip_preserves_equal_signed_zero(tmp_path):
    mx = require_metal('Metal required')
    values=np.array([False,True,True,False])
    options=dict(state_inputs=True,phase_dtype="boolean",phase_symbol="phase_clip",
        optimise=False,result_expression="clip(-sinh(phase_clip),clip_lower,clip_upper)",
        namespace_values={"clip_lower":np.float64(-0.),"clip_upper":np.float64(0.)},
        run_steps=(1,2,0,3))
    comparison=OriginalComparison('numpy', 'sinh', values, tmp_path, **options)
    assert np.all(comparison.expected.view(np.uint64)==0x8000000000000000)
    comparison.check_native()
