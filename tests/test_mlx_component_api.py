"""The state-dependent Brian prepass retains its canonical due-mask producer."""

import numpy as np
import pytest

from mlx_backend.brian_device import original_api
from original_api_reference import (original_runtime,build_original_api_fixture,
    assert_snapshot_equal,capture_rng_state,assert_rng_state_equal)


@pytest.mark.metal
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_all_input_api_prepass_keeps_due_masks_rng_weights_and_partial_clocks(tmp_path,dtype):
    import brian2 as b
    expected=[]
    for native in (False,True):
        with original_runtime(
                original_api, seed=813, target='numpy', float_dtype=dtype, native=native) as device:
            fixture=build_original_api_fixture(tmp_path/str(native),
                parameters=dict(t_dly=.3*b.ms,r_poi=3700*b.Hz,r_poi2=7200*b.Hz),
                exc=(0,1,2,3),exc2=(0,2),silenced=())
            # Brian sorts equal-order Poisson sources by actual object
            # name. Keep the same scheduler and draw-to-channel mapping
            # even while the CPU fixture's objects are still alive.
            for channel,source in enumerate(fixture.inputs):
                source._name=f'component_api_input_{channel:04d}'
            for index,duration in enumerate((.15,.35,0.,.1)):
                if index==1:
                    fixture.synapses.w_[:]=np.asarray(fixture.synapses.w_[:])*(-.5)
                    fixture.neurons.rfc_[0]=.0003
                snapshot=fixture.run_ms(duration)
                if native:
                    assert_snapshot_equal(snapshot,expected[index])
                    assert not device._sessions[fixture.network].prepared.scan_arrivals
                else:expected.append(snapshot)
            if native:
                assert_rng_state_equal(capture_rng_state(),rng)
                np.testing.assert_array_equal(np.random.random(10),draw)
            else:
                rng=capture_rng_state();draw=np.random.random(10)
