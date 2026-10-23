"""NumPy masked integer updates through the actual Brian object API."""
import brian2 as b
import numpy as np
import pytest

from original_api_protocol import check_original_api_runs, observe_run
from test_mlx_compiled_integer_adapter import _typed_fixture, _forbid_cpu_steps


def masked_fixture(n, *, record=False):
    """Keep the active-vector SIMD prefix and tail observable at each tick."""
    f = np.resize(np.array([4294967297., -3.75, 2.**31, -2.**31-1,
                           2.**63, -2.**63, np.nan, np.inf, -np.inf]), n)
    return _typed_fixture(
        dict(count=np.dtype('int64'), f=np.dtype('float64'), flag=np.dtype('bool')),
        'count += int(f) + 4294967297; flag = count > 0',
        dict(count=np.resize(np.array([0, 2**63-1, -2**63, 17], np.int64), n),
             f=f, flag=np.ones(n, bool)),
        threshold='((i+timestep(t,dt))%5!=0) and ((i+2*timestep(t,dt))%7!=0)',
        record=record, default_reset=False)


@pytest.mark.metal
@pytest.mark.parametrize('record', [False, True])
@pytest.mark.parametrize('n', [17, 35])
def test_numpy_masked_integer_api_preserves_state_monitor_and_continuation(n, record, monkeypatch):
    def exercise(fixture, mode):
        yield observe_run(fixture,.7)
        fixture.neurons.f[0] = 2**32 + 3.
        fixture.neurons.count[1] = -27
        yield observe_run(fixture,.4)
        if mode == 'cpu':
            assert fixture.neurons.resetter['spike'].codeobj.class_name == 'numpy'
        else:
            device = b.get_device()
            assert device.native_steps == 11
            assert device.native_runs == 2
    check_original_api_runs(lambda mode:masked_fixture(n,record=record),exercise,
        monkeypatch,guard=_forbid_cpu_steps,integer_dtype=np.int64)


@pytest.mark.metal
@pytest.mark.parametrize('spikes', [False, True])
@pytest.mark.parametrize('record', [False, True])
def test_numpy_masked_scalar_store_error_prefix_and_repaired_continuation(spikes, record, monkeypatch):
    """Empty selected vectors still execute NumPy's scalar assignment check."""
    def build():
        return _typed_fixture(dict(x=np.dtype('int8'), y=np.dtype('int8')),
            'x=first_good; y=bad; g+=1*mV',
            dict(x=np.full(5, 7, np.int8), y=np.full(5, 8, np.int8)),
            threshold=str(spikes), record=record, default_reset=False,
            namespace=dict(first_good=17., bad=float('nan')))
    def exercise(fixture, mode):
        yield observe_run(fixture,0.)
        failed=observe_run(fixture,.1,raises=b.BrianObjectException)
        if mode == 'cpu':
            assert isinstance(failed.error.__cause__, ValueError)
        yield failed
        fixture.neurons.namespace['bad']=3.5
        yield observe_run(fixture,.3)
    check_original_api_runs(lambda mode:build(),exercise,monkeypatch,guard=_forbid_cpu_steps)
