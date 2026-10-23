"""Actual Brian dtype flow into the shared half component."""
from trigonometry_checks import OriginalComparison
import numpy as np
import pytest
from native_execution import require_metal



@pytest.mark.metal
def test_original_numpy_boolean_sqrt_then_arctan_retains_half_loop(tmp_path):
    mx = require_metal('Metal required')
    values=np.array([False,True,True,False])
    options=dict(state_inputs=True,phase_dtype="boolean",phase_symbol="phase_root",
        optimise=False,result_expression="arctan(sqrt(phase_root))+.1",run_steps=(1,2,0,3))
    comparison=OriginalComparison('numpy', 'arctan', values, tmp_path, **options)
    assert np.sqrt(values).dtype==np.float16
    assert np.arctan(np.sqrt(values)).dtype==np.float16
    with comparison.native() as (actual,native,device):
        assert device.native_steps==6


@pytest.mark.metal
def test_original_numpy_half_promoted_floor_uses_selected_array_cast_loop(tmp_path):
    mx = require_metal('Metal required')
    # Nine selected rows have an eight-element FCVTZS64+narrow prefix and
    # a scalar FCVTZS32 tail; physical array position cannot substitute rank.
    values=np.arange(17)%2==0
    options=dict(state_inputs=True,phase_dtype="boolean",phase_symbol="phase_cast",
        optimise=False,result_expression="floor(arctan(phase_cast)+cast_offset)+.1",
        threshold_expression="phase_cast",namespace_values={"cast_offset":1e20},
        run_steps=(1,2,0,3))
    comparison=OriginalComparison('numpy', 'arctan', values, tmp_path, **options)
    comparison.check_native()
