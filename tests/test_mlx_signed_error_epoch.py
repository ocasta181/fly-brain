"""Negative absolute Brian clocks retain ordered native failure epochs."""
from native_execution import forbid_cpu_steps as guard_cpu_steps
from dataclasses import replace

import numpy as np
import pytest


@pytest.mark.metal
def test_negative_clock_error_prefix_and_sticky_epoch(tmp_path, monkeypatch, record_property):
    import brian2 as b
    import mlx.core as mx
    from mlx_backend import configured
    from mlx_backend.brian_device import original_api
    from mlx_backend.runtime_errors import first_error, model_exception
    from original_api_reference import (
        assert_rng_state_equal, assert_snapshot_equal, build_original_api_fixture,
        capture_rng_state, original_runtime,
    )

    snapshots, streams, messages = [], [], []
    values = dict(x=[2, 3, 4, 5], y=[2, 2, -1, 2], column=[0]*4)
    for native in (False, True):
        with original_runtime(
                original_api, dt_ms=.1, seed=745, integer_dtype=np.int64, native=native) as device:
            from original_api_reference import load_original_model
            parameters = dict(
                eqs=load_original_model().default_params["eqs"]+"x:integer\ny:integer\ncolumn:integer\n",
                eq_th="int(drive(t,column))+x**y>0",
                eq_rst="v=v_rst;g=0*mV;column+=3*int(i==0)", t_rfc=0*b.ms,
                t_dly=.2*b.ms, drive=b.TimedArray([[1., 2.], [3., 4.]], dt=.1*b.ms))
            fixture = build_original_api_fixture(tmp_path/str(native), parameters=parameters,
                exc=(), exc2=(), silenced=(), initial_state=values)
            if native:
                guard_cpu_steps(fixture.network, monkeypatch, 'Negative-clock native error executed a CPU timestep', patch_preparation=True)
            fixture.run_ms(-.1)
            empty = fixture.capture()
            with pytest.raises(b.BrianObjectException) as failure:
                fixture.run_ms(.1)
            cause = failure.value.__cause__
            assert type(cause) is ValueError
            assert str(cause) == "Integers to negative integer powers are not allowed."
            snapshots.append((empty, fixture.capture()))
            streams.append(capture_rng_state())
            messages.append((type(cause), str(cause)))
            if native:
                assert device.native_runs == 1 and device.native_steps == 1
                prepared = device._sessions[fixture.network].prepared
                assert prepared.program.runtime_error_order == "operation"
                record_property("native_runs", device.native_runs)
                record_property("native_steps", device.native_steps)
                record_property("failure_clock_tick", snapshots[-1][-1].clock_timestep)

    assert messages[0] == messages[1]
    for actual, expected in zip(snapshots[1], snapshots[0]):
        assert_snapshot_equal(actual, expected)
    assert_rng_state_equal(streams[1], streams[0])
    assert snapshots[0][-1].clock_timestep == -1

    # Keep failures across two windows spanning -1 -> 0. The later low-row
    # TimedArray error must not replace the earlier high-row power failure.
    initial = configured.initial_state(prepared)
    state = replace(initial, step=-1, values={**initial.values,
        **{name: mx.array(np.asarray(value, np.int64)) for name, value in values.items()}})
    events = mx.zeros((1,), mx.uint8)
    for _ in range(2):
        state, _ = configured.advance(prepared, state, events, 1, 1,
            event_begin=-1, event_steps=2)
        assert state.error_epoch_origin == -1
    mx.eval(state.release)
    releases = np.asarray(state.release)
    assert isinstance(model_exception(prepared.program.runtime_errors, int(releases[0]) & 65535), IndexError)
    assert first_error(releases, "operation")[0] == 2
    with pytest.raises(ValueError, match="Integers to negative integer powers"):
        configured.evaluate_state(state)
    record_property("sticky_error_epoch_origin", state.error_epoch_origin)
