"""Actual-model composition must retain both division helper namespaces."""
from trigonometry_checks import OriginalComparison
import pytest
from trigonometry_checks import check_original_half_operators as check_original


@pytest.mark.metal
def test_numpy_tangent_and_remainder_headers_compose(tmp_path):
    check_original(tmp_path,"tan(sin(phase_ops)%.3)")


@pytest.mark.metal
@pytest.mark.parametrize('target',['numpy','cython'])
def test_original_forward_pair_power_remainder_graph(tmp_path,target):
    """Actual optimized operators retain the selected sine/cosine context."""
    import numpy as np

    values=np.array([.1,.3,-.3,.75,-.75,3.,-3.,2**26,-2**26],np.float32)
    options=dict(state_inputs=True,dtype='float32',phase_symbol='phase_ops',
        cache_suffix='-forward-operator-composition',run_steps=(1,2,0,1),
        result_expression='sin(phase_ops)+cos(phase_ops)+(phase_ops%0.375)**3')
    comparison=OriginalComparison(target, 'sin', values, tmp_path, **options)
    comparison.check_native()
