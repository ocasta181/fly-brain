"""Original schedules compose random ODE inputs, Poisson channels and delay."""

import numpy as np
import pytest


@pytest.mark.metal
@pytest.mark.parametrize("target,dtype,phases,delay,record", [
    ("numpy", np.float32, ("resets", "groups", "thresholds", "synapses"), 0., True),
    ("cython", np.float64, ("synapses", "groups", "thresholds", "resets"), .2, False),
])
def test_original_schedule_random_state_and_input_continuation(
        tmp_path, monkeypatch, target, dtype, phases, delay, record):
    import brian2 as b
    from original_api_reference import build_original_api_fixture
    from original_api_protocol import check_original_api_runs, observe_run

    from mlx_backend.expression_poisson_input import ExpressionPoisson

    def build(mode):
        b.prefs.core.network.default_schedule = ["start", *phases, "end"]
        fixture = build_original_api_fixture(
            tmp_path/str(mode=='native'), parameters={
                "eqs": "dv/dt=(-v+u*mV)/ms:volt\n"
                       "dg/dt=-g/(2*ms):volt\nrfc:second\n"
                       "u=rand():1(shared,constant over dt)",
                "eq_th": "rand()<.4", "eq_rst": "v=v_rst+randn()*mV;g=rand()*mV",
                "t_rfc": np.array([.1, .2, .3, .4])*b.ms,
                "t_dly": delay*b.ms, "r_poi": 3700*b.Hz, "r_poi2": 7200*b.Hz,
            }, exc=(1, 1), exc2=(3,), silenced=(),
            initial_state={"g": np.array([1., -3., 4., 2.])*b.mV})
        if not record:
            fixture.network.remove(fixture.monitor)
            fixture.monitor = b.SpikeMonitor(fixture.neurons, record=False)
            fixture.network.add(fixture.monitor)
        return fixture

    def exercise(fixture, mode):
        for index, duration in enumerate((0., .1, .15, .05, .2, 3.1, .3)):
            if index == 4:
                fixture.inputs[1].active = False
                fixture.neurons.rfc = np.array([.3, .1, .4, .2])*b.ms
            observed=observe_run(fixture,duration,rng=True)
            if mode=='native':
                from brian2.devices.device import get_device
                assert isinstance(get_device()._sessions[fixture.network].sampler, ExpressionPoisson)
            yield observed
        assert int(fixture.neurons.clock.variables["timestep"].get_value()[0]) > 32
    check_original_api_runs(build,exercise,monkeypatch,seed=773,target=target,float_dtype=dtype)
