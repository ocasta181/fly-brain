"""Eon benchmark entry points for the optional Apple MLX backend.

GPL-2.0-or-later. See mlx_backend and the independent regression fixtures.
"""

import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

from benchmark import (
    BACKEND_NAMES,
    N_RUN_VALUES,
    T_RUN_VALUES_SEC,
    BenchmarkLogger,
    get_experiment,
    get_spike_output_path,
    path_comp,
    path_con,
    print_summary_table,
    save_result_csv,
    spike_io_enabled,
)


def spike_table(connectome, runs, exp_name):
    import numpy as np
    import pyarrow as pa
    from mlx_backend.export import _unmapped_identifier_rows

    trial = np.concatenate([run.spikes.trials for run in runs])
    neurons = np.concatenate([run.spikes.neurons for run in runs])
    steps = np.concatenate([run.spikes.steps for run in runs])
    unmapped = _unmapped_identifier_rows(connectome.neuron_ids)
    if unmapped is not None:
        missing = np.flatnonzero(unmapped[neurons])
        if missing.size:
            first = missing[np.lexsort((neurons[missing], trial[missing]))[0]]
            raise KeyError(int(neurons[first]))
    order = np.lexsort((neurons, steps, trial))
    seconds = np.concatenate([
        run.spikes.steps.astype(np.float64) * getattr(run, "dt_s", 0.0001)
        for run in runs
    ])[order]
    identifiers = pa.array(connectome.neuron_ids[neurons[order]])
    schema = pa.schema(
        [
            ("t", pa.float64()),
            ("time_ms", pa.float64()),
            ("trial", pa.int64()),
            ("neuron_index", pa.int64()),
            ("flywire_id", pa.array(connectome.neuron_ids).type),
            ("exp_name", pa.string()),
        ]
    )
    return pa.table(
        {
            "t": seconds,
            "time_ms": seconds * 1000,
            "trial": trial[order],
            "neuron_index": neurons[order],
            "flywire_id": identifiers,
            "exp_name": [exp_name] * len(seconds),
        },
        schema=schema,
    )


def run_single_benchmark(
    t_run_sec,
    n_run,
    experiment,
    logger,
    run_idx=None,
    total_runs=None,
    run_label=None,
    round_idx=None,
    *,
    seed=20261004,
    stimulus_path=None,
    engine="window",
    strategy="batch",
    stream_output=None,
):
    """Run independent trials with the selected internal execution strategy.

    Saved input is the retained stimulus.json/stimulus.npz directory. Shorter
    horizons and trial counts select verified prefixes of that archive.
    """
    import mlx_engine

    started = perf_counter()
    record = spike_io_enabled()
    steps = mlx_engine.dimensions(t_run_sec, n_run)
    mlx_engine.configure()
    exp_name = f"mlx_t{t_run_sec}s_n{n_run}"
    path = get_spike_output_path(exp_name, run_label, round_idx) if record else None
    sidecar = path.with_suffix(".mlx.json") if record else None
    if record and (path.exists() or sidecar.exists()):
        raise FileExistsError(
            f"MLX output already exists; choose a fresh run label: {path}"
        )

    import numpy as np

    effective_seed = seed
    if stimulus_path is None and round_idx is not None and round_idx > 1:
        if type(seed) is not int or seed < 0:
            raise ValueError("MLX seed must be a nonnegative integer")
        effective_seed = int(
            np.random.SeedSequence([seed, round_idx]).generate_state(
                1, dtype=np.uint64
            )[0]
        )

    spec = mlx_engine.experiment_spec(experiment)
    connectome = mlx_engine.load_inputs(path_comp, path_con)
    inputs, silenced = mlx_engine.stimuli(
        connectome, spec, steps, n_run, effective_seed, stimulus_path
    )
    if strategy not in {"batch", "sequential"}:
        raise ValueError("Unknown MLX trial strategy")
    batched = engine == "window" and strategy == "batch" and n_run > 1
    workloads = [mlx_engine.combine_trials(inputs)] if batched else inputs
    if stream_output is not None and type(stream_output) is not bool:
        raise TypeError("stream_output must be a boolean or None")
    spooled = record and (
        stream_output if stream_output is not None
        else any(hasattr(stimulus.events, "chunk") for stimulus in workloads)
    )
    spool = None
    if spooled:
        from mlx_backend.export import SpoolSpikes

        spool = SpoolSpikes(connectome, (trial for item in inputs for trial in item.trial_indices), exp_name)
    setup = perf_counter() - started
    simulation = 0.0
    jit = 0.0
    postprocessing = 0.0
    runs = []
    collection = save_time = 0.0
    n_spikes = n_active = 0
    try:
        for index, stimulus in enumerate(workloads):
            logger.log(
                f"MLX {exp_name}: {'batch' if batched else 'fresh trial'} {index + 1}/{len(workloads)}"
            )
            execution_options = dict(engine=engine, record=record)
            if spool is not None:
                execution_options["spike_sink"] = spool
            run_started = perf_counter()
            result = mlx_engine.execute(connectome, stimulus, silenced, **execution_options)
            engine_elapsed = perf_counter() - run_started
            setup += result.timings["setup_s"]
            preparation = result.timings.get("jit_and_warmup_s", 0.0)
            export_io = result.timings.get("spike_spool_io_s", 0.0)
            jit += preparation
            simulation += result.timings["simulation_wall_s"]
            save_time += export_io
            postprocessing += max(
                0.0,
                engine_elapsed - result.timings["setup_s"] - preparation
                - result.timings["simulation_wall_s"] - export_io,
            )
            runs.append(result)
        if spool is not None:
            exported = spool.finish(path)
            collection = postprocessing + exported["collection_s"]
            save_time += exported["save_s"]
            n_spikes, n_active = spool.n_spikes, spool.n_active_neurons
        elif record:
            import pyarrow.parquet as pq

            collecting = perf_counter()
            table = spike_table(connectome, runs, exp_name)
            collection = postprocessing + perf_counter() - collecting
            n_spikes = table.num_rows
            n_active = int(np.unique(table["flywire_id"].to_numpy()).size)
            path.parent.mkdir(parents=True, exist_ok=True)
            saving = perf_counter()
            with path.open("xb") as destination:
                pq.write_table(table, destination, compression="brotli")
            save_time += perf_counter() - saving
    finally:
        if spool is not None:
            cleaning = perf_counter()
            spool.close()
            save_time += perf_counter() - cleaning
    spike_digest = None
    if record:
        from mlx_backend.inputs import file_sha256

        hashing = perf_counter()
        spike_digest = file_sha256(path)
        save_time += perf_counter() - hashing
    elapsed = perf_counter() - started
    timings = {
        "network_creation_total": setup,
        "device_build": jit,
        "simulation_total": simulation,
        "simulation_avg_per_trial": simulation / n_run,
        "result_collection": collection,
        "result_save": save_time,
        "total_elapsed": elapsed,
        "realtime_ratio": t_run_sec * n_run / simulation,
        "realtime_ratio_total": t_run_sec * n_run / elapsed,
    }
    result = {
        "t_run_sec": t_run_sec,
        "n_run": n_run,
        "n_spikes": n_spikes,
        "n_active_neurons": n_active,
        "status": "success",
        "timings": timings,
        "backend_key": "mlx",
        "experiment_name": experiment["name"],
        "experiment_key": experiment["key"],
        "run_label": run_label or "",
        "round": round_idx or "",
        "spike_path": str(path) if record else "",
    }
    if not record:
        logger.log("MLX spike probing/output disabled; no Parquet written")
        return result

    import resource

    pin = connectome.pin or mlx_engine.input_pin()
    consumed_hashes = [digest for run in runs for digest in (run.input_sha256 or ())]
    event_hashes = consumed_hashes if len(consumed_hashes) == n_run else [x.sha256 for x in inputs]
    if any(digest is None for digest in event_hashes):
        raise RuntimeError("Streamed events were not fully fingerprinted during execution")
    metadata = {
        "result": result,
        "execution_mode": "batched-fresh-trials"
        if batched
        else "sequential-fresh-singletons",
        "engine": engine,
        "device": runs[0].device_name,
        "dt_ms": 0.1,
        "seed": inputs[0].seed,
        "base_seed": seed,
        "effective_seed": inputs[0].seed,
        "round": round_idx,
        "record_spikes": record,
        "output_mode": "bounded-spool" if spooled else "retained",
        "generator": "PCG64",
        "seed_tuples": [[x.seed, x.generator_code, x.trial_indices[0]] for x in inputs],
        "canonical_event_sha256": event_hashes,
        "stimulus_source": str(Path(stimulus_path).resolve())
        if stimulus_path
        else "canonical-generation",
        "activated_ids": list(spec.activated_ids),
        "activation_classes": list(spec.activation_classes),
        "rates_hz": list(spec.rates_hz),
        "silenced_ids": list(spec.silenced_ids),
        "input_sha256": {
            "completeness": pin.completeness_sha256,
            "connectivity": pin.connectivity_sha256,
        },
        "spike_sha256": spike_digest,
        "mlx_peak_allocation_bytes": max(x.peak_device_bytes for x in runs),
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "per_trial_timings": [
            {key: value / n_run for key, value in runs[0].timings.items()}
            for _ in inputs
        ]
        if batched
        else [x.timings for x in runs],
        "per_trial_timing_scope": "amortized batch timing"
        if batched
        else "independent trial timing",
        "engine_timings": [x.timings for x in runs],
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("mlx", "mlx-metal", "numpy", "pyarrow")
        },
        "MLX_ENABLE_TF32": "0",
        "compilation": "custom-metal-jit" if engine == "window" else "disabled",
        "timing_scope": "Whole-loop simulation includes input staging/fingerprinting, dispatch, synchronization and block collection; scratch export I/O is charged to result_save. Window-engine JIT preparation uses disposable state and is separate. Final Arrow preparation, Parquet export and file hashing are outside simulation; total ends after export cleanup, before JSON sidecar/logging.",
        "memory_scope": "Process ru_maxrss and MLX allocator peaks are distinct, non-additive scopes; neither establishes physical unified-memory usage.",
        "validation_scope": "Replay parity and independent-cohort agreement are separate claims. This record describes the selected engine, workload and seed; independent-cohort P9 equivalence remains unestablished.",
    }
    with sidecar.open("x") as destination:
        json.dump(metadata, destination, indent=2, allow_nan=False)
        destination.write("\n")
    logger.log(f"MLX wrote {n_spikes} spikes to {path}")
    return result


def run_all_benchmarks(
    t_run_values=None,
    n_run_values=None,
    experiment=None,
    logger=None,
    run_label=None,
    round_idx=None,
):
    if logger is None:
        logger = BenchmarkLogger()
    if experiment is None:
        experiment = get_experiment()
    times = T_RUN_VALUES_SEC if t_run_values is None else t_run_values
    counts = N_RUN_VALUES if n_run_values is None else n_run_values
    parameters = [(duration, count) for count in counts for duration in times]
    results = []
    for index, (duration, count) in enumerate(parameters, 1):
        result = run_single_benchmark(
            duration,
            count,
            experiment,
            logger,
            index,
            len(parameters),
            run_label,
            round_idx,
        )
        save_result_csv(BACKEND_NAMES["mlx"], result)
        results.append(result)
    print_summary_table(results, BACKEND_NAMES["mlx"], logger)
    return results
