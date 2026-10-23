"""Self-local zero-delay steps retain the real original API's exact contract."""
from native_execution import forbid_cpu_steps, forbid_network_cpu_steps
from contextlib import nullcontext

import numpy as np
import pytest


@pytest.mark.metal
@pytest.mark.parametrize("target", ["numpy", "cython"])
@pytest.mark.parametrize("case,record", [("dense", True), ("sparse", False), ("sparse", True),
                                         ("cross", True), ("input", True),
                                         ("shared", True), ("schedule", True), ("delay", True)])
def test_original_api_local_steps(tmp_path, monkeypatch, target, case, record):
    import brian2 as b
    import original_api_reference as reference
    from mlx_backend import configured
    from mlx_backend.brian_device import original_api

    monkeypatch.setattr(reference, "write_tiny_dataset", reference.synthetic_dataset(4, labelled=False,
        edges=((2, 2, 1.25), (0, 0, 2**54), (0, 0, -2**54),
               (0, 1 if case == "cross" else 0, .5), (1, 1, 0.), (2, 2, -.25), (3, 3, -0.))))
    calls = []
    advance = configured.advance

    def observed(p, s, *args, **kwargs):
        calls.append((p.independent_neurons, args[2]))
        return advance(p, s, *args, **kwargs)

    monkeypatch.setattr(configured, "advance", observed)
    expected = None
    for native in (False, True):
        with reference.cpu_runtime(seed=953, target=target):
            with original_api(workers=False) if native else nullcontext():
                fixture = reference.build_original_api_fixture(
                    tmp_path/str(native), parameters=dict(
                        eqs=reference.load_original_model().default_params["eqs"]+"\nx:1"
                            + ("\nz:1 (shared)" if case == "shared" else ""),
                        eq_th="x>.5", eq_rst="v=v_rst"+("+z*mV" if case == "shared" else ""),
                        w_syn=1*b.mV,
                        t_rfc=.2*b.ms, t_dly=(.2 if case == "delay" else 0)*b.ms, r_poi=0*b.Hz),
                    exc=(0,) if case == "input" else (), exc2=(), silenced=(),
                    initial_state={"x": [1., -1., -1., -1.] if case == "sparse" else 1.,
                                   "g": np.array([-0., 0., -.25, .5])*b.mV})
                if case == "shared":
                    fixture.neurons.z = .25
                if case == "schedule":
                    fixture.network.schedule = ["start", "groups", "thresholds", "resets", "synapses", "end"]
                if not record:
                    fixture.network.remove(fixture.monitor)
                    fixture.monitor = b.SpikeMonitor(fixture.neurons, record=False)
                    fixture.network.add(fixture.monitor)
                if native:
                    forbid_cpu_steps(fixture.network, monkeypatch)
                snapshots = []
                for duration in (0., .1, .2, 3.3, .2):
                    fixture.run_ms(duration)
                    snapshots.append((fixture.capture(), reference.capture_rng_state()))
                if native:
                    for (actual, rng), (wanted, wanted_rng) in zip(snapshots, expected):
                        reference.assert_snapshot_equal(actual, wanted)
                        reference.assert_rng_state_equal(rng, wanted_rng)
                    eligible = case in ("dense", "sparse")
                    assert all(independent == eligible for independent, _ in calls)
                    assert any(length > 1 for _, length in calls) == (eligible or case == "delay")
                else:
                    expected = snapshots


@pytest.mark.metal
@pytest.mark.parametrize("error", [False, True])
def test_local_original_run_trial_and_error_prefix(tmp_path, monkeypatch, error):
    import brian2 as b
    import original_api_reference as reference
    from mlx_backend.brian_device import original_api

    nodes, edges = reference.synthetic_dataset(4, labelled=False,
        edges=((0, 0, .5), (0, 0, -.25), (2, 2, 1.)))(tmp_path)
    original = reference.load_original_model()
    params = {**original.default_params, "eqs": original.default_params["eqs"]+"x:1\na:1\n",
              "eq_th": "True", "eq_rst": "v=v_rst;x+=1;a=1/(3-x)" if error else "v=v_rst;x+=1",
              "t_rfc": 0*b.ms, "t_dly": 0*b.ms, "t_run": .8*b.ms}
    create = original.create_model
    captured = []

    def capture_model(*args):
        objects = create(*args)
        captured[:] = objects
        return objects

    monkeypatch.setattr(original, "create_model", capture_model)
    before = b.Network.before_run
    snapshots = []
    for native in (False, True):
        with reference.cpu_runtime(seed=953, target="cython"):
            with original_api(workers=False) if native else nullcontext():
                with monkeypatch.context() as patch:
                    if native:
                        forbid_network_cpu_steps(b.Network, patch, 'CPU timestep executed', before=before)
                    if error:
                        with pytest.raises(b.core.base.BrianObjectException) as raised:
                            original.run_trial([], [], [], nodes, edges, params)
                        assert isinstance(raised.value.__cause__, ZeroDivisionError)
                    else:
                        original.run_trial([], [], [], nodes, edges, params)
                group, syn, monitor = captured
                values = {name: np.asarray(group.variables[name].get_value()).copy()
                          for name in ("v", "g", "x", "a", "rfc", "lastspike", "not_refractory", "_spikespace")}
                values.update({"monitor_"+name: np.asarray(monitor.variables[name].get_value()).copy()
                               for name in ("i", "t", "count", "N")})
                values.update({"clock_"+name: np.asarray(group.clock.variables[name].get_value()).copy()
                               for name in ("t", "timestep")})
                snapshots.append((values, reference.capture_rng_state()))
    for name, wanted in snapshots[0][0].items():
        actual = snapshots[1][0][name]
        assert actual.dtype == wanted.dtype and actual.shape == wanted.shape
        assert actual.tobytes() == wanted.tobytes(), name
    reference.assert_rng_state_equal(snapshots[1][1], snapshots[0][1])
