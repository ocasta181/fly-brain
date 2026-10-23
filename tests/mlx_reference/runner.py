from hashlib import sha256
from time import perf_counter

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray

from . import core
from .bucketed import prepare
from .core import boolean_input, evaluate
from mlx_backend.models import (
    Connectome,
    NeuronRows64,
    SimulationRun,
    SpikeEvents,
    Stimulus,
    TrialIds64,
)


def run(
    connectome: Connectome,
    stimulus: Stimulus,
    silenced: tuple[int, ...],
    *,
    precision: str,
    engine: str = "bucketed",
    record: bool = True,
    spike_sink=None,
    model=None,
) -> SimulationRun:
    if spike_sink is not None and not record:
        raise ValueError("A spike sink requires recording to be enabled")
    if engine != "bucketed":
        raise ValueError(f"Unknown reference engine: {engine}")
    if model is not None and not model.state_specialization:
        raise ValueError("The test oracle requires the fixed model")
    if model is not None and model.dt_ms != stimulus.dt_ms:
        raise ValueError("Stimulus timestep differs from the resolved model")
    if stimulus.dt_ms != core.DT_MS or (
        stimulus.amplitudes_mv is not None
        and tuple(stimulus.amplitudes_mv) != (68.75,) * len(stimulus.targets)
    ):
        raise ValueError("The fixed-model oracle requires default timestep and input amplitudes")
    mx.reset_peak_memory()
    started = perf_counter()
    execution = prepare(connectome, stimulus.targets, silenced, precision)
    state = core.initial_state(execution.network, len(stimulus.trial_indices))
    with mx.stream(mx.gpu):
        evaluate(
            state.voltage_offset_mv,
            state.synaptic_mv,
            state.last_spike_step,
            state.queue,
        )
    setup_s = perf_counter() - started
    steps = stimulus.events.shape[1]
    first_step_s = simulation_s = collection_s = 0.0
    collected: list[tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]] = []
    trial_ids = np.array(stimulus.trial_indices, dtype=np.int64)
    input_hashes = [sha256() for _ in trial_ids]
    spool_io_s = 0.0
    started_loop = perf_counter()
    for begin in range(0, steps, 256):
        end = min(begin + 256, steps)
        raw = stimulus.events
        chunk = raw.chunk(begin, end) if hasattr(raw, "chunk") else raw[:, begin:end]
        if (
            chunk.dtype != np.uint8
            or chunk.shape != (len(trial_ids), end - begin, len(stimulus.targets))
            or chunk.max(initial=0) > 1
        ):
            raise ValueError("Invalid canonical stimulus chunk")
        for trial, hasher in enumerate(input_hashes):
            hasher.update(np.ascontiguousarray(chunk[trial]).data)
        block: list[mx.array] = []
        for step in range(begin, end):
            started_step = perf_counter()
            with mx.stream(mx.gpu):
                events = boolean_input(chunk[:, step - begin, :].astype(np.bool_))
                state, trace = execution.advance(state, events)
                evaluate(
                    state.voltage_offset_mv,
                    state.synaptic_mv,
                    state.last_spike_step,
                    state.queue,
                )
            elapsed = perf_counter() - started_step
            if step == 0:
                first_step_s = elapsed
            else:
                simulation_s += elapsed
            if record:
                block.append(trace.spikes)
        if record:
            started_collection = perf_counter()
            with mx.stream(mx.gpu):
                raster = np.asarray(mx.stack(block), dtype=np.bool_)
            local_steps, local_trials, neurons = np.nonzero(raster)
            batch = (
                    trial_ids[local_trials],
                    neurons.astype(np.int64),
                    (local_steps + begin).astype(np.int64),
                )
            exporting = 0.0
            if spike_sink is None:
                collected.append(batch)
            else:
                exporting = float(spike_sink.append(SpikeEvents(*batch)))
                if not np.isfinite(exporting) or exporting < 0:
                    raise ValueError("Spike sink returned invalid I/O timing")
                spool_io_s += exporting
            collection_s += perf_counter() - started_collection - exporting
        print(f'Simulated {min(begin + 256, steps)}/{steps} steps', flush=True)
    simulation_wall_s = perf_counter() - started_loop - spool_io_s
    if record and collected:
        started_collection = perf_counter()
        trials = np.concatenate([part[0] for part in collected])
        neurons = np.concatenate([part[1] for part in collected])
        spike_steps = np.concatenate([part[2] for part in collected])
        order = np.lexsort((neurons, spike_steps, trials))
        spikes = SpikeEvents(
            TrialIds64(trials[order]), NeuronRows64(neurons[order]), spike_steps[order]
        )
        collection_s += perf_counter() - started_collection
    else:
        spikes = SpikeEvents(
            TrialIds64(np.array([], np.int64)),
            NeuronRows64(np.array([], np.int64)),
            np.array([], np.int64),
        )
    return SimulationRun(
        spikes,
        {
            'setup_s': setup_s,
            'first_step_s': first_step_s,
            'warm_simulation_s': simulation_s,
            'simulation_wall_s': simulation_wall_s,
            'collection_s': collection_s,
            'compilation_s': 0.0,
            'spike_spool_io_s': spool_io_s,
        },
        mx.get_peak_memory(),
        str(mx.device_info()['device_name']),
        input_sha256=tuple(hasher.hexdigest() for hasher in input_hashes),
    )
