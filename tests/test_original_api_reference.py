"""Independent CPU baseline for the frozen application's observable API."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from original_api_reference import (
    COUNTS, DESTINATIONS, NEURON_IDS, SOURCES, assert_snapshot_equal,
    build_original_api_fixture, cpu_runtime, run_original_api,
)


@pytest.fixture
def cpu():
    pytest.importorskip("brian2")
    with cpu_runtime() as b:
        yield b


def endpoints(b, **overrides):
    return {"r_poi": 1 / b.defaultclock.dt, "r_poi2": 0*b.Hz, **overrides}


def test_original_objects_keep_raw_graph_silencing_and_string_ids(cpu, tmp_path):
    b = cpu
    fixture = build_original_api_fixture(tmp_path, parameters=endpoints(b))
    assert type(fixture.neurons) is b.NeuronGroup
    assert type(fixture.synapses) is b.Synapses
    assert type(fixture.monitor) is b.SpikeMonitor
    assert all(type(source) is b.PoissonInput for source in fixture.inputs)
    assert tuple(pd.read_csv(fixture.paths[0], index_col=0).index) == NEURON_IDS
    initial = fixture.capture()
    np.testing.assert_array_equal(initial.synaptic_pre, SOURCES)
    np.testing.assert_array_equal(initial.synaptic_post, DESTINATIONS)
    weights = np.asarray(COUNTS) * float(fixture.parameters["w_syn"])
    weights[np.asarray(SOURCES) == 3] = 0
    np.testing.assert_array_equal(initial.weights_v, weights)
    np.testing.assert_array_equal(initial.state["rfc"], [0, .0022, 0, .0022])
    assert len(fixture.inputs) == 3


@pytest.mark.parametrize("durations", [(.15, .15), (.1, .2), (0., .15, 0., .15, 0.)])
def test_partitioned_requested_horizon_uses_cpu_clock_semantics(cpu, tmp_path, durations):
    b = cpu
    parameters = endpoints(b, t_dly=.2*b.ms)
    whole = run_original_api(tmp_path / "whole", [.3], parameters=parameters)
    split = run_original_api(tmp_path / "split", durations, parameters=parameters)
    # Requested Network time accumulates binary64 durations; .1+.2 ms and
    # .3 ms need not have identical bits even when the executed steps agree.
    assert_snapshot_equal(replace(split.snapshots[-1], network_t_s=whole.snapshots[-1].network_t_s),
                          whole.snapshots[-1])
    assert split.snapshots[-1].network_t_s == sum(value*.001 for value in durations)
    if .15 in durations:
        first = split.snapshots[durations.index(.15)]
        assert first.network_t_s == .00015
        assert first.clock_t_s == .0002
        assert first.clock_timestep == 2
    assert split.snapshots[-1].clock_timestep == 3


@pytest.mark.parametrize("delay_steps", [0, 1, 18, 33, 65])
def test_non_window_aligned_repeated_runs_keep_pending_arrivals(cpu, tmp_path, delay_steps):
    b = cpu
    parameters = endpoints(b, t_dly=delay_steps*b.defaultclock.dt)
    whole = run_original_api(tmp_path / "whole", [12.9], parameters=parameters)
    split = run_original_api(tmp_path / "split", [.15, .05, 1.7, 0., 2.35, 8.65],
                             parameters=parameters)
    assert_snapshot_equal(split.snapshots[-1], whole.snapshots[-1])
    assert whole.snapshots[-1].spike_i.size > 0


def test_zero_duration_preserves_initial_and_accumulated_observables(cpu, tmp_path):
    fixture = build_original_api_fixture(tmp_path, parameters=endpoints(cpu))
    initial = fixture.capture()
    assert_snapshot_equal(fixture.run_ms(0), initial)
    completed = fixture.run_ms(2.1)
    assert completed.spike_i.size > 0
    assert_snapshot_equal(fixture.run_ms(0), completed)


def test_seeded_n1_poisson_runs_reproduce_states_monitor_and_next_rng(cpu, tmp_path):
    b = cpu
    parameters = {"r_poi": 3700*b.Hz, "t_dly": .2*b.ms}
    b.seed(589)
    first = run_original_api(tmp_path / "first", [1.15, .15, 7.8],
                             parameters=parameters, exc=(0,), exc2=())
    first_next = np.random.random(12)
    b.seed(589)
    second = run_original_api(tmp_path / "second", [1.15, .15, 7.8],
                              parameters=parameters, exc=(0,), exc2=())
    for actual, expected in zip(second.snapshots, first.snapshots):
        assert_snapshot_equal(actual, expected)
    np.testing.assert_array_equal(np.random.random(12), first_next)
    assert first.snapshots[-1].spike_i.size > 0


def test_state_refractory_weights_and_silencing_mutate_actual_objects(cpu, tmp_path):
    b = cpu
    fixture = build_original_api_fixture(tmp_path, parameters=endpoints(b, t_dly=.3*b.ms))
    initial = fixture.run_ms(.2)
    fixture.neurons.v = [-44, -48.5, -53.25, -51.125] * b.mV
    fixture.neurons.g = [1.25, -2.5, 3.75, -.125] * b.mV
    fixture.neurons.rfc = [.125, .275, .425, .575] * b.ms
    fixture.synapses.w[0] = -1.125*b.mV
    fixture.synapses.w[1] = .3125*b.mV
    fixture.original.silence([2], fixture.synapses)
    edited = fixture.capture()
    assert_snapshot_equal(fixture.run_ms(0), edited)
    later = fixture.run_ms(.7)
    np.testing.assert_array_equal(later.weights_v, edited.weights_v)
    np.testing.assert_array_equal(later.state["rfc"], edited.state["rfc"])
    assert later.spike_N[0] > initial.spike_N[0]
    np.testing.assert_array_equal(initial.weights_v[0], np.asarray(COUNTS)[0]*.000275)
    np.testing.assert_array_equal(edited.weights_v[np.asarray(SOURCES) == 2], 0)


def test_original_monitor_trains_and_dataframe_use_live_si_state(cpu, tmp_path):
    fixture = run_original_api(tmp_path, [2.3], parameters=endpoints(cpu))
    snapshot = fixture.snapshots[-1]
    np.testing.assert_array_equal(snapshot.spike_count,
                                  np.bincount(snapshot.spike_i, minlength=len(NEURON_IDS)))
    assert snapshot.spike_N.tolist() == [len(snapshot.spike_i)]
    for neuron, times in snapshot.spike_trains_s.items():
        np.testing.assert_array_equal(times, snapshot.spike_t_s[snapshot.spike_i == neuron])
    frame = fixture.original.construct_dataframe(
        [fixture.original.get_spk_trn(fixture.monitor)], "api-test", dict(enumerate(NEURON_IDS)),
    )
    assert set(frame["flywire_id"]) <= set(NEURON_IDS)
    np.testing.assert_array_equal(np.sort(frame["t"]), np.sort(snapshot.spike_t_s))
    assert set(frame["trial"]) == {0}
