"""Actual Eon half-cosine NaN policy is separate from scalar32 cosine."""
from trigonometry_checks import OriginalComparison
import numpy as np
import pytest
from native_execution import require_metal


@pytest.mark.metal
@pytest.mark.parametrize("dtype",["float32","float64"])
def test_original_numpy_half_cosine_preserves_negative_nan(tmp_path,dtype):
    mx = require_metal('Metal required')
    negative_nan=np.uint64(0xfff8000000000123).view(np.float64)
    values=np.array([False,True,True,False])
    options=dict(state_inputs=True,phase_dtype="boolean",phase_symbol="phase_nan",dtype=dtype,
        optimise=False,result_expression="cos(sin(phase_nan)*negative_phase)",
        namespace_values={"negative_phase":negative_nan},run_steps=(1,2,0,3))
    comparison=OriginalComparison('numpy', 'cos', values, tmp_path, **options)
    comparison.check_native()
