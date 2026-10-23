"""Actual original Eon CPU targets versus native target-aware TimedArrays."""
from state_assertions import assert_raw_state, assert_spikes_equal
from native_execution import forbid_call, require_metal

from hashlib import sha256
import importlib.util
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest

mx = require_metal('Metal is required', allow_module_level=True)
b = pytest.importorskip("brian2")

from mlx_backend import configured
from mlx_backend.compiler import compile_model
from mlx_backend.model import resolve_model
from mlx_backend.models import Connectome, Stimulus

pytestmark = pytest.mark.metal


@pytest.mark.parametrize("dimensions", [1, 2])
def test_compiled_timedarray_lif_arrivals_states_and_continuation(dimensions, monkeypatch):
    """Exercise the real compiled linear update, beyond a quiescent TA probe."""
    from configured_reference import reference
    monkeypatch.setitem(b.prefs, "codegen.target", "cython")
    monkeypatch.setitem(b.prefs, "codegen.runtime.cython.cache_dir",
                        str(Path(__file__).resolve().parents[1] / "executions/cython-timedarray-arrivals"))
    values = np.array([0., .2, -.1, .3])
    if dimensions == 2:
        values = values[:, None]*np.array([[1., .5, 1.5]])
    call = "drive(t)" if dimensions == 1 else "drive(t,i)"
    model = resolve_model({"drive": b.TimedArray(values, dt=.15*b.ms),
                           "eq_th": f"v > v_th + {call}*mV", "t_rfc": .35*b.ms})
    counts = np.array([400.125, -99.875, 200.25, 10.125, -3.125, 4.25])
    connectome = Connectome(np.array(["left", "middle", "right"]),
        np.array([2, 0, 0, 1, 2, 0], np.int32),
        np.array([1, 1, 2, 2, 1, 1], np.int32), counts, counts*.275)
    events = np.zeros((1, 83, 2), np.uint8)
    events[:, [0, 1, 6, 17, 18, 31, 32, 33, 63, 64, 65], :] = 1
    stimulus = Stimulus(events, (0, 1), (100., 100.), (0,), 0, 0,
                        sha256(events.tobytes()).hexdigest(), model.dt_ms)
    expected = reference(connectome, stimulus, model)
    prepared = configured.prepare(connectome, stimulus, model)
    assert prepared.program.refractory_policy == "cython_fast"
    forbid_call(b.Network, 'run', 'Native execution ran a CPU timestep', monkeypatch)
    result, state = configured.run_prepared(prepared, stimulus, return_state=True,
                                            chunk_steps=19, block_windows=2)
    assert_spikes_equal(result.spikes, expected.spikes)
    assert_raw_state(configured.host_state(state, prepared), expected.state)
    continued = configured.initial_state(prepared)
    staged = mx.array(events.reshape(-1))
    while continued.step < events.shape[1]:
        length = min(7, prepared.window-continued.step % prepared.window,
                     events.shape[1]-continued.step)
        continued, _ = configured.advance(prepared, continued, staged, events.shape[1], length)
    configured.evaluate_state(continued)
    for name in state.values:
        np.testing.assert_array_equal(np.asarray(continued.values[name]), np.asarray(state.values[name]))
    np.testing.assert_array_equal(np.asarray(continued.release), np.asarray(state.release))


@pytest.mark.parametrize("target,case", [
    ("numpy", "threshold"), ("cython", "threshold"), ("auto", "threshold"),
    ("cpp", "threshold"), ("numpy", "reset"), ("cython", "reset"),
    ("cython", "negative"), ("cpp", "negative"), ("cython", "invalid"),
    ("cpp", "float"), ("cpp", "float32_column"), ("cython", "integer_wrap"),
    ("cython_strict", "threshold"), ("cython_nocontract", "threshold"),
    ("cython", "excited"), ("cpp", "excited"),
    ("cython", "equation"), ("cpp", "equation"),
    ("cython", "equation2d"), ("cpp", "equation2d"),
    ("cython", "equation2d_arrivals"), ("cpp", "equation2d_arrivals"),
])
def test_original_cpu_target_matches_native(tmp_path, monkeypatch, target, case):
    from brian2.devices.device import get_device
    previous_device, previous_target = get_device(), b.prefs.codegen.target
    previous_dt = b.defaultclock.dt
    monkeypatch.setitem(b.prefs, "codegen.runtime.cython.cache_dir",
                        str(Path(__file__).resolve().parents[1] / f"executions/cython-timedarray-targets-{target}"))
    # Brian's in-memory extension cache omits compiler flags and cache_dir.
    # Distinguish each policy in the independent CPU source as well as its
    # on-disk cache, so a default-fast-math module cannot satisfy a strict run.
    from brian2.codegen.runtime.cython_rt.extension_manager import cython_extension_manager
    create_extension = cython_extension_manager.create_extension
    def policy_extension(code, *args, **kwargs):
        return create_extension(f"# TimedArray CPU policy: {target}\n"+dedent(code), *args, **kwargs)
    monkeypatch.setattr(cython_extension_manager, "create_extension", policy_extension)
    if target == "cython_strict":
        monkeypatch.setitem(b.prefs, "codegen.cpp.extra_compile_args", ["-O3", "-fno-fast-math", "-ffp-contract=off"])
    elif target == "cython_nocontract":
        monkeypatch.setitem(b.prefs, "codegen.cpp.extra_compile_args", ["-O3", "-ffast-math", "-fno-finite-math-only", "-ffp-contract=off"])
    path = Path(__file__).resolve().parents[1] / "code/paper-phil-drosophila/model.py"
    spec = importlib.util.spec_from_file_location("original_timed_target", path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    neuron_ids = ["one", "two"] if case in ("excited", "float32_column") or case.startswith("equation2d") else ["one"]
    destination, count = (1, 1.25) if case in ("excited", "equation2d_arrivals") else (0, 0.)
    pd.DataFrame(index=neuron_ids).to_csv(tmp_path / "neurons.csv")
    pd.DataFrame({"Presynaptic_Index": [0], "Postsynaptic_Index": [destination],
                  "Excitatory x Connectivity": [count]}).to_parquet(tmp_path / "edges.parquet")
    try:
        if target == "cpp":
            b.set_device("cpp_standalone", directory=str(tmp_path / "cpp"), build_on_run=True)
            monkeypatch.setitem(b.prefs, "devices.cpp_standalone.extra_make_args_unix", ["-j2"])
        else:
            b.set_device("runtime")
            b.prefs.codegen.target = "cython" if target.startswith("cython") else target
        b.defaultclock.dt = .1*b.ms
        params = dict(original.default_params)
        params.update(eqs=params["eqs"]+"sample : 1\ncounter : integer\n",
                      t_rfc=0*b.ms, eq_th="t>=0*ms")
        if case.startswith("equation"):
            values = [[1., .5], [2., 3.], [3., -1.], [4., .25]] if case.startswith("equation2d") else [1., 2., 3., 4.]
            call = "drive(t,i)" if case.startswith("equation2d") else "drive(t)"
            params["drive"] = b.TimedArray(values, dt=1*b.ms)
            params["eqs"] = params["eqs"].replace("v_0 - v + g", f"v_0 - v + g + {call}*mV")
            params["eq_th"] = "v > v_th"
            params["eq_rst"] = f"v=v_rst; sample={call}; counter+=1"
            if case == "equation2d_arrivals":
                params.update(eq_th="(i==0) or (v > v_th)", t_dly=0*b.ms, t_rfc=.35*b.ms)
        elif case == "excited":
            params["drive"] = b.TimedArray([.1, .2, -.1, 0.], dt=b.ms)
            params["eq_th"] = "(i==0) or (v > v_th+drive(t)*mV)"
            params["eq_rst"] = "v=v_rst; sample=drive(t); counter+=1"
            params["t_dly"] = 0*b.ms
        elif case in ("threshold", "reset"):
            params["drive"] = b.TimedArray(np.tile([0., 1.], 512), dt=b.defaultclock.dt/8)
            params["eq_rst"] = "v=v_rst; sample=drive(t+dt/16); counter+=1"
            if case == "threshold":
                params["eq_th"] = "drive(t+dt/16)>.5"
        else:
            params["drive"] = b.TimedArray([[0., 1.], [2., 3.]], dt=b.defaultclock.dt)
            column = {"negative": "-1", "invalid": "2", "float": "1.9",
                      "float32_column": "column_scale*(i+1)",
                      "integer_wrap": "4294967297"}[case]
            if case == "float32_column":
                params["column_scale"] = np.float32(.75)
            params["eq_rst"] = f"v=v_rst; sample=drive(t, {column}); counter+=1"
        model = resolve_model(params)
        if target == "cpp":
            # Preparation on an active standalone device must not register
            # temporary groups/weak variables in the original application build.
            compile_model(model)
        neurons, synapses, monitor = original.create_model(tmp_path / "neurons.csv", tmp_path / "edges.parquet", params)
        steps = 67 if case in ("threshold", "reset", "excited") or case.startswith("equation") else 3
        objects = [neurons, synapses, monitor]
        if case == "excited" or case.startswith("equation"):
            trace = b.StateMonitor(neurons, ["v", "g"], record=True, when="end")
            objects.append(trace)
        b.Network(*objects).run(steps*b.defaultclock.dt, namespace={})
        expected_steps = np.rint(np.asarray(monitor.t_)/model.dt_s).astype(np.int64)
        expected_neurons = np.asarray(monitor.i[:], dtype=np.int64)
        expected_state = {name: np.asarray(getattr(neurons, name+"_")[:]).copy()
                          for name in ("v", "g", "rfc", "lastspike", "not_refractory", "sample", "counter")}
        if case == "excited" or case.startswith("equation"):
            expected_traces = {name: np.asarray(getattr(trace, name+"_")).T.copy() for name in ("v", "g")}
        resolved_target = type(neurons.thresholder["spike"].codeobj).__name__
        if target != "cpp":
            state_object = neurons.state_updater.codeobj
            (tmp_path / "state-source.txt").write_text(str(state_object.code.run))
            if state_object.class_name == "cython":
                (tmp_path / "state-module.txt").write_text(state_object.compiled_code["run"].__file__)
    finally:
        if target == "cpp":
            get_device().reinit()
        b.set_device(previous_device)
        b.prefs.codegen.target = previous_target
        b.defaultclock.dt = previous_dt
    assert model.reference_codegen_target == {"NumpyCodeObject": "numpy", "CythonCodeObject": "cython",
                                              "CPPStandaloneCodeObject": "cpp"}[resolved_target]
    forbid_call(b.Network, 'run', 'Native execution ran a CPU timestep', monkeypatch)
    # Deliberately change ambient preference after resolving the CPU contract.
    monkeypatch.setitem(b.prefs, "codegen.target", "numpy" if target != "numpy" else "cython")
    connectome = Connectome(np.array(neuron_ids), np.array([0], np.int32), np.array([destination], np.int32),
                           np.array([count]), np.array([count*.275]))
    events = np.zeros((1, steps, 0), np.uint8)
    stimulus = Stimulus(events, (), (), (0,), 0, 0, sha256(events.tobytes()).hexdigest(), model.dt_ms)
    prepared = configured.prepare(connectome, stimulus, model)
    result, state = configured.run_prepared(prepared, stimulus, return_state=True)
    np.testing.assert_array_equal(result.spikes.steps, expected_steps)
    assert np.all(result.spikes.trials == 0)
    np.testing.assert_array_equal(result.spikes.neurons, expected_neurons)
    if case == "excited" or case.startswith("equation"):
        traced = configured.initial_state(prepared)
        staged = mx.array(events.reshape(-1))
        for step in range(steps):
            traced, _ = configured.advance(prepared, traced, staged, steps, 1)
            actual_trace = configured.host_state(traced, prepared)
            for name in ("v", "g"):
                np.testing.assert_array_equal(actual_trace[name][0].view(np.uint64),
                    expected_traces[name][step].view(np.uint64), err_msg=f"{name} at step {step}")
    actual_state = configured.host_state(state, prepared)
    assert_raw_state({name: actual_state[name][0] for name in expected_state}, expected_state)
    if case == "threshold":
        assert expected_steps[expected_steps < 3].tolist() == ([] if target == "numpy" else [0, 1, 2])
    elif case == "reset":
        assert expected_steps.tolist() == list(range(steps))
    elif case in ("negative", "invalid"):
        assert np.isnan(expected_state["sample"]).all()
