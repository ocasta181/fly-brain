"""Resolve ambient schedules against Brian's actual object ordering (CPU)."""

from itertools import permutations

import numpy as np
import pytest

from configured_reference import reference
from mlx_backend.model import resolve_model
from mlx_backend.models import Connectome, Stimulus

PHASES = ("groups", "thresholds", "synapses", "resets")
EXTRA_SCHEDULES = [
    ("start", *PHASES),  # end is unused by the application
    ("start", *PHASES, "groups"),  # last duplicate wins, never runs twice
    ("start", "before_groups", "after_thresholds", *PHASES, "end"),
    ("start", "custom", "resets", "synapses", "thresholds", "groups"),
    ("end", "resets", "groups", "start", "synapses", "thresholds"),
]


def brian_network(b):
    neurons = b.NeuronGroup(1, "v : volt\ng : volt\nrfc : second",
        threshold="False", reset="v=0*volt", refractory="rfc")
    synapses = b.Synapses(neurons, neurons, on_pre="g+=1*mV")
    synapses.connect(condition=False)
    spikes = b.SpikeMonitor(neurons)
    poisson = b.PoissonInput(neurons, "v", N=1, rate=0*b.Hz, weight=1*b.mV)
    return b.Network(neurons, synapses, spikes, poisson)


@pytest.mark.parametrize("schedule", [("start", *p, "end") for p in permutations(PHASES)]
                         + EXTRA_SCHEDULES)
def test_resolved_order_matches_independent_brian_objects(monkeypatch, schedule):
    b = pytest.importorskip("brian2")
    monkeypatch.setitem(b.prefs, "core.network.default_schedule", list(schedule))
    network = brian_network(b)
    # Read each active phase once; multiple objects share thresholds/synapses.
    independent = tuple(dict.fromkeys(o.when for o in network.sorted_objects if o.when != "start"))
    model = resolve_model()
    assert model.execution_phases == independent
    extra = independent.index("synapses") < independent.index("thresholds")
    assert model.arrival_delay_steps == model.delay_steps + extra
    assert not model.state_specialization or schedule == ("start", *PHASES, "end")


@pytest.mark.parametrize("missing", ("start", *PHASES))
def test_omitted_required_slots_match_brian_rejection(monkeypatch, missing):
    b = pytest.importorskip("brian2")
    schedule = [slot for slot in ("start", *PHASES, "end") if slot != missing]
    monkeypatch.setitem(b.prefs, "core.network.default_schedule", schedule)
    network = brian_network(b)
    with pytest.raises(KeyError) as independent:
        _ = network.sorted_objects
    with pytest.raises(KeyError) as execution:
        network.run(0*b.ms)
    with pytest.raises(KeyError) as native:
        resolve_model()
    assert native.value.args == independent.value.args == (missing,)
    assert execution.value.args == independent.value.args


def test_ambient_automatic_slots_are_accepted_and_snapshotted(monkeypatch):
    b = pytest.importorskip("brian2")
    schedule = ["start", "before_groups", "after_groups", *PHASES]
    monkeypatch.setitem(b.prefs, "core.network.default_schedule", schedule)
    model = resolve_model({"eq_th": "False"})
    network = brian_network(b)
    assert network.sorted_objects
    # Explicit setters are stricter than the ambient preference used by Eon.
    with pytest.raises(ValueError, match="created automatically"):
        network.schedule = schedule
    monkeypatch.setitem(b.prefs, "core.network.default_schedule", ["start", *PHASES, "end"])
    assert model.schedule == tuple(schedule)
    assert model.signature != resolve_model({"eq_th": "False"}).signature
    c = Connectome(np.arange(1), np.array([], np.int32), np.array([], np.int32),
                   np.array([], np.float64), np.array([], np.float64))
    events = np.ones((1, 2, 1), np.uint8)
    s = Stimulus(events, (0,), (0.,), (9,), 0, 0, None, model.dt_ms)
    expected = reference(c, s, model)
    assert expected.state["v"][0, 0] > model.v_0
    assert tuple(b.prefs.core.network.default_schedule) == ("start", *PHASES, "end")
