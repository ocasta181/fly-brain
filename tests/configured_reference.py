"""Independent CPU Brian replay of the application's configurable model.

No native execution code is imported here.  The neuron equations, linear
integrator, refractory writes, synaptic queue, threshold and reset all execute
in Brian.  Only the Poisson draws are replaced by supplied binary events.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
from hashlib import sha256
from pathlib import Path
import tempfile
from textwrap import dedent

import numpy as np

from mlx_backend.models import Connectome, SpikeEvents, Stimulus


@dataclass(frozen=True)
class ReferenceRun:
    """SI state arrays have shape (trials, neurons).

    Optional traces have shape (trials, steps, neurons) and are sampled in the
    ``end`` phase, after resets.  ``lastspike`` is in seconds, including Brian's
    initial ``-10000`` sentinel.  Spike trial identifiers are the stimulus's actual IDs.
    """

    spikes: SpikeEvents
    state: dict[str, np.ndarray]
    traces: dict[str, np.ndarray] | None
    dt_s: float
    steps: int
    random_states: tuple | None = None


def reference(
    connectome: Connectome,
    stimulus: Stimulus,
    model,
    silenced=(),
    *,
    record_state: bool = False,
    random_states=None,
    cpu_target: str | None = None,
) -> ReferenceRun:
    """Run fresh CPU Brian networks for each trial with identical input events.

    ``connectome.counts * model.w_syn`` supplies the actual edge weights;
    ``weights_mv`` is deliberately unused.  Silencing affects outgoing edges.
    A channel's supplied amplitude is in mV, or defaults to ``w_syn*f_poi``.
    Channels replay individually, in their input creation order, with exactly
    the stateupdate template and synapses phase used by ``PoissonInput``.
    """
    import brian2 as b
    from brian2.devices.device import RuntimeDevice, get_device
    from brian2.codegen.translation import make_statements

    gc.collect()
    if cpu_target is None:
        cpu_target = model.float32_codegen if model.state_dtype == "float32" else "numpy"

    if stimulus.events.ndim != 3:
        raise ValueError("Reference events must have trial, step and channel axes")
    trials, steps, channels = stimulus.events.shape
    if trials != len(stimulus.trial_indices) or channels != len(stimulus.targets):
        raise ValueError("Reference event dimensions differ from stimulus metadata")
    if not np.isclose(stimulus.dt_ms * .001, model.dt_s, rtol=1e-12, atol=0):
        raise ValueError("Reference stimulus and model timesteps differ")
    events = (np.asarray(stimulus.events)
              if isinstance(stimulus.events, np.ndarray)
              else np.asarray(stimulus.events.chunk(0, steps)))
    if events.shape != (trials, steps, channels) or not np.all((events == 0) | (events == 1)):
        raise ValueError("Reference input events must be a binary array")
    amplitudes = (np.full(channels, model.poisson_weight_v)
                  if stimulus.amplitudes_mv is None
                  else np.asarray(stimulus.amplitudes_mv, dtype=np.float64) * .001)
    if amplitudes.shape != (channels,) or not np.all(np.isfinite(amplitudes)):
        raise ValueError("Reference requires one finite amplitude per input channel")

    # Explicit clocks and networks isolate this helper from Brian's magic
    # network.  Save preferences so reference calls do not affect native tests.
    previous = {name: b.prefs[name] for name in (
        "codegen.target", "core.default_float_dtype", "core.default_integer_dtype",
        "legacy.refractory_timing", "codegen.cpp.extra_compile_args",
        "core.network.default_schedule", "codegen.runtime.cython.cache_dir",
        "codegen.loop_invariant_optimisations",
    )}
    switched_device = not isinstance(get_device(), RuntimeDevice)
    if switched_device:
        b.set_device("runtime")
    target = model.reference_codegen_target
    # CPP and Cython share Brian's compiled TimedArray lookup implementation.
    # Actual standalone CPP behavior has its own end-to-end target probe.
    b.prefs.codegen.target = (cpu_target if model.state_dtype == "float32" else
                              "cython" if target in ("cython", "cpp") else "numpy")
    # Freeze the reference's actual arithmetic contract, including when the
    # caller subsequently changes ambient Brian preferences. Brian's in-memory
    # Cython cache omits flags, so distinguish policies in source as well.
    b.prefs.core.network.default_schedule = list(model.schedule)
    if target in ("cython", "cpp") or cpu_target == "cython":
        policy = repr((target, model.reference_compile_args, model.float32_compile_args,
                       model.state_dtype, model.float32_optimise))
        base = Path(b.prefs.codegen.runtime.cython.cache_dir or tempfile.gettempdir())
        b.prefs.codegen.runtime.cython.cache_dir = str(base/"reference-profiles"/sha256(policy.encode()).hexdigest())
    extension_manager = original_extension = None
    if target in ("cython", "cpp"):
        from brian2.codegen.runtime.cython_rt.extension_manager import cython_extension_manager
        extension_manager = cython_extension_manager
        original_extension = extension_manager.create_extension
        policy = repr((target, model.reference_compile_args))
        def create_extension(code, *args, **kwargs):
            return original_extension("# Frozen reference policy: "+policy+"\n"+dedent(code), *args, **kwargs)
        extension_manager.create_extension = create_extension
        b.prefs.codegen.cpp.extra_compile_args = list(model.reference_compile_args)
    b.prefs.core.default_float_dtype = np.dtype(model.state_dtype).type
    b.prefs.core.default_integer_dtype = np.dtype(model.integer_dtype).type
    b.prefs.legacy.refractory_timing = model.legacy_refractory_timing
    # Brian's cached make_statements(optimise=True) key omits the current
    # loop-invariant preference. Give every float32 reference fresh statement
    # translation, as in an independent process, and restore the caller's cache.
    statement_cache = make_statements._cache if model.state_dtype == "float32" else None
    saved_statements = dict(statement_cache) if statement_cache is not None else None
    if statement_cache is not None:
        statement_cache.clear()
    if model.state_dtype == "float32":
        b.prefs.codegen.loop_invariant_optimisations = model.float32_optimise
        if cpu_target != "numpy":
            b.prefs.codegen.cpp.extra_compile_args = list(model.float32_compile_args)
    states: dict[str, list[np.ndarray]] = {}
    traces: dict[str, list[np.ndarray]] = {}
    spike_trials, spike_neurons, spike_steps = [], [], []
    ambient_random=np.random.get_state()if random_states is not None else None
    final_random=[]
    if random_states is not None and len(random_states)!=len(stimulus.trial_indices):
        raise ValueError('Reference random snapshots must match trial count')
    try:
        for trial_row, trial_id in enumerate(stimulus.trial_indices):
            if random_states is not None:np.random.set_state(random_states[trial_row])
            clock = b.Clock(dt=model.dt_s * b.second, name="configured_reference_clock")
            namespace = model.brian_namespace()
            # Input names are private to the helper and cannot shadow a model
            # namespace value silently.
            event_names = [f"eon_replay_events_{channel}" for channel in range(channels)]
            weight_names = [f"eon_replay_weight_{channel}" for channel in range(channels)]
            if set(event_names + weight_names) & namespace.keys():
                raise ValueError("Model namespace collides with reference replay inputs")
            scalar_voltage = 'shared' in b.Equations(model.eqs)['v'].flags
            for channel, (event_name, weight_name) in enumerate(zip(event_names, weight_names)):
                # A zero-step run still needs valid TimedArray storage for the
                # compiler; Network.run(0) never evaluates this sentinel.
                values = events[trial_row, :, channel] if steps else np.zeros(1)
                if model.state_dtype == "float32" and cpu_target == "numpy" and not scalar_voltage:
                    # Actual PoissonInput returns an integer event *vector*.
                    # A scalar TimedArray(t) changes NumPy's float32 promotion.
                    values = np.broadcast_to(values[:, None], (len(values), len(connectome.neuron_ids)))
                namespace[event_name] = b.TimedArray(values, dt=clock.dt)
                namespace[weight_name] = amplitudes[channel] * b.volt
            neurons = b.NeuronGroup(
                len(connectome.neuron_ids), model.eqs, method="linear",
                threshold=model.eq_th, reset=model.eq_rst, refractory="rfc",
                namespace=namespace, clock=clock, dtype={
                    equation.varname: np.dtype(
                        model.integer_dtype if equation.var_type == "integer" else
                        bool if equation.var_type == "boolean" else model.state_dtype,
                    ) for equation in b.Equations(model.eqs).values()
                },
                name="configured_reference_neurons",
            )
            if model.legacy_refractory_timing:
                # Brian 2.8's string-duration legacy branch returns a bare
                # comparison. Repair only its missing assignment, independently
                # of native model/compiler code; Brian generates the predicate
                # and executes every simulation stage on the CPU.
                original_refractory = neurons.state_updater._get_refractory_code

                def assigned_refractory(run_namespace):
                    code = original_refractory(run_namespace)
                    return code if code.startswith("not_refractory =") else "not_refractory = " + code

                neurons.state_updater._get_refractory_code = assigned_refractory
            neurons.v = model.v_0 * b.volt
            neurons.g = 0
            neurons.rfc = model.t_rfc * b.second
            synapses = b.Synapses(
                neurons, neurons, "w : volt", on_pre="g_post += w",
                delay=model.t_dly * b.second, clock=clock,
                name="configured_reference_synapses*",
            )
            if len(connectome.sources):
                synapses.connect(i=connectome.sources, j=connectome.destinations)
                weights = np.asarray(connectome.counts, dtype=np.float64) * model.w_syn
                synapses.w = np.where(np.isin(connectome.sources, silenced), 0, weights) * b.volt
            else:
                synapses.connect(condition=False)
            runners = []
            for channel, target in enumerate(stimulus.targets):
                subgroup = neurons[int(target):int(target) + 1]
                subgroup.rfc = 0 * b.second
                event_args = "t, i" if model.state_dtype == "float32" and cpu_target == "numpy" and not scalar_voltage else "t"
                runners.append(subgroup.run_regularly(
                    f"v += {event_names[channel]}({event_args}) * {weight_names[channel]}",
                    when="synapses", order=channel,
                    name="configured_reference_input*",
                ))
            spikes = b.SpikeMonitor(neurons, name="configured_reference_spikes*")
            state_names = tuple(dict.fromkeys(
                [name for name, equation in neurons.equations.items()
                 if equation.type != "subexpression"] + ["lastspike", "not_refractory"]
            ))
            objects = [neurons, synapses, spikes, *runners]
            if record_state:
                monitor = b.StateMonitor(
                    neurons, state_names, record=True, when="end",
                    name="configured_reference_state*",
                )
                objects.append(monitor)
            network = b.Network(*objects)
            network.run(steps * clock.dt, namespace={})
            final_random.append(np.random.get_state())
            times = np.rint(np.asarray(spikes.t_) / model.dt_s).astype(np.int64)
            spike_trials.append(np.full(len(times), trial_id, dtype=np.int64))
            spike_neurons.append(np.asarray(spikes.i[:], dtype=np.int64))
            spike_steps.append(times)
            for name in state_names:
                values = np.asarray(getattr(neurons, name + "_")[:])
                states.setdefault(name, []).append(np.broadcast_to(values, (len(neurons),)).copy())
                if record_state:
                    values = np.asarray(getattr(monitor, name + "_"))
                    traces.setdefault(name, []).append(values.T.copy())
    finally:
        if ambient_random is not None:np.random.set_state(ambient_random)
        if extension_manager is not None:
            extension_manager.create_extension = original_extension
        for name, value in previous.items():
            b.prefs[name] = value
        if switched_device:
            b.reset_device()
    return ReferenceRun(
        SpikeEvents(*(np.concatenate(values).astype(np.int64, copy=False)
                      for values in (spike_trials, spike_neurons, spike_steps))),
        {name: np.stack(values) for name, values in states.items()},
        {name: np.stack(values) for name, values in traces.items()} if record_state else None,
        model.dt_s, steps,tuple(final_random),
    )
