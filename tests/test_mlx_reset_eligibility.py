"""Original Eon reset statements preserve the Poisson phase's actual mask."""
from native_execution import forbid_cpu_steps, require_metal
from contextlib import nullcontext

import numpy as np
import pytest

mx = require_metal('Metal is required', allow_module_level=True)

from original_api_reference import (
    assert_rng_state_equal, assert_snapshot_equal, capture_rng_state,
    cpu_runtime, run_original_api,
)

pytestmark = pytest.mark.metal


@pytest.mark.parametrize("target,delay,order,inputs,random_reset", [
    ("numpy", .2, "default", False, False),
    ("numpy", .2, "default", True, False),
    ("numpy", 0., "default", True, False),
    ("numpy", .2, "reset_before_input", True, False),
    ("cython", .2, "default", True, False),
    ("cython", 0., "default", True, False),
    ("numpy", 0., "default", True, True),
    ("cython", .2, "reset_before_input", True, True),
])
def test_reset_not_refractory_preserves_original_state_and_draws(
    tmp_path, monkeypatch, target, delay, order, inputs, random_reset,
):
    import brian2 as b
    from mlx_backend.brian_device import original_api

    references = None
    for native in (False, True):
        with cpu_runtime(seed=741, target=target):
            if order == "reset_before_input":
                b.prefs.core.network.default_schedule = [
                    "start", "groups", "thresholds", "resets", "synapses", "end",
                ]
            parameters = dict(
                eq_th="v > v_th or t < dt",
                eq_rst="v=v_rst;g=0*mV;not_refractory=True"
                    + (";v+=rand()*mV" if random_reset else ""),
                t_dly=delay*b.ms, t_rfc=.3*b.ms,
                r_poi=3700*b.Hz, r_poi2=7200*b.Hz,
            )
            with original_api(workers=False) if native else nullcontext() as device:
                def no_cpu_steps(fixture):
                    forbid_cpu_steps(fixture.network, monkeypatch, 'CPU timestep executed on MLX path')

                actual = run_original_api(
                    tmp_path/str(native), (0., .15, .15, .4),
                    parameters=parameters, exc=(0, 0) if inputs else (),
                    exc2=(2,) if inputs else (), silenced=(),
                    initial_state={"v": np.array([-44., -48., -43., -49.])*b.mV},
                    mutations={0: no_cpu_steps} if native else None,
                )
                rng = capture_rng_state()
                if native:
                    assert device.native_runs == 4
                    for observed, expected in zip(actual.snapshots, references):
                        assert_snapshot_equal(observed, expected)
                    assert_rng_state_equal(rng, expected_rng)
                else:
                    references, expected_rng = actual.snapshots, rng
