"""Actual Brian namespace Constant values differ from their declared dtype."""
from trigonometry_checks import OriginalComparison
import numpy as np
import pytest
from native_execution import require_metal


@pytest.mark.metal
@pytest.mark.parametrize("target,value",[
    ("numpy",np.float32(.1)),("cython",np.float32(.1)),
    ("numpy",np.float16(.1)),("cython",np.float16(.1)),
    ("numpy",np.uint16(0xfc01).view(np.float16)),
    ("cython",np.uint16(0xfc01).view(np.float16)),
])
def test_original_forward_scalar_namespace_value_type(tmp_path,target,value):
    mx = require_metal('Metal required')
    options=dict(state_inputs=True,result_expression="sin(scalar_phase)",
        namespace_values={"scalar_phase":value},cache_suffix="-namespace-value")
    comparison=OriginalComparison(target, 'sin', np.zeros(4), tmp_path, **options)
    comparison.check_native()
