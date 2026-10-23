"""Optional batched float32 Metal runner for the current Eon model."""

from __future__ import annotations

import math
import operator
import os
import tempfile
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter as time

from benchmark import (
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

_BACKEND_NAME = 'MLX float32 (Metal)'
_CHUNK_SIZE = 32
_ARROW_ROWS = 65_536


def run_single_benchmark(
    t_run_sec: float, n_run: int, experiment: Mapping, logger: BenchmarkLogger,
    run_idx: int | None = None, total_runs: int | None = None,
    run_label: str | None = None, round_idx: int | None = None,
) -> dict[str, object]:
    """Run all trials once on Metal; optional experiment ``seed`` defaults to 0.

    CSV simulation time is synchronized native execution, including GPU spike
    extraction. Input staging, host collection and export have separate timers.
    ``runtime_s`` includes those runtime costs and overlaps their phase timers.
    """
    started = time()
    timings = {}
    result = {
        't_run_sec': t_run_sec,
        'n_run': n_run,
        'n_active_neurons': 0,
        'n_spikes': 0,
        'backend_key': 'mlx32',
        'experiment_name': experiment['name'],
        'experiment_key': experiment['key'],
        'run_label': run_label or '',
        'round': round_idx or '',
        'spike_path': '',
        'timings': timings,
    }
    writer = None
    temporary_path = None
    sink_collection_s = 0.0
    sink_export_s = 0.0
    rows_written = 0

    try:
        dependency_start = time()
        import numpy as np
        import pandas as pd

        from mlx32.engine import run_batch
        from mlx32.model import Drive, Model, make_graph

        timings['dependency_setup'] = time() - dependency_start
        model = Model()
        if isinstance(n_run, (bool, np.bool_)):
            raise ValueError('n_run must be a positive integer')
        n_run = operator.index(n_run)
        if not 0 < n_run <= 2**31 - 1:
            raise ValueError('n_run must be a positive int32 integer')
        if isinstance(t_run_sec, (bool, np.bool_)):
            raise ValueError('t_run_sec must be finite and positive')
        duration = float(t_run_sec)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('t_run_sec must be finite and positive')
        ticks = duration / model.dt_s
        if not math.isfinite(ticks):
            raise ValueError('t_run_sec exceeds the supported tick range')
        steps = round(ticks)
        if not 0 < steps <= 2**31 - 1 or not math.isclose(
            ticks, steps, rel_tol=0.0, abs_tol=4 * math.ulp(ticks)
        ):
            raise ValueError('t_run_sec must be an integral number of dt ticks')
        seed_round = 0 if round_idx is None else round_idx
        exp_name = f'mlx32_t{t_run_sec}s_n{n_run}'
        record_spikes = spike_io_enabled()
        run_info = f'[{run_idx}/{total_runs}] ' if run_idx else ''
        logger.log(f'{run_info}BENCHMARK: {_BACKEND_NAME}, '
                   f't_run={t_run_sec}s, n_run={n_run}')
        logger.log(f'Spike probing/output: '
                   f'{"enabled" if record_spikes else "disabled"}')

        load_start = time()
        df_comp = pd.read_csv(path_comp, index_col=0)
        df_con = pd.read_parquet(path_con)
        timings['data_load'] = time() - load_start
        graph_start = time()
        graph = make_graph(
            df_comp.index.to_numpy(copy=True),
            df_con['Presynaptic_Index'].to_numpy(),
            df_con['Postsynaptic_Index'].to_numpy(),
            df_con['Excitatory x Connectivity'].to_numpy(),
        )
        timings['graph_creation'] = time() - graph_start
        del df_comp, df_con
        mapping_start = time()
        flyid2i = {flyid: i for i, flyid in enumerate(graph.neuron_ids)}
        exc = [flyid2i[flyid] for flyid in experiment['neu_exc']]
        exc2 = [flyid2i[flyid] for flyid in experiment['neu_exc2']]
        silenced = [flyid2i[flyid] for flyid in experiment['neu_slnc']]
        drive = Drive(
            np.array(exc + exc2, dtype=np.int32),
            np.array([experiment['stim_rate']] * len(exc) + [0.0] * len(exc2)),
            seed=experiment.get('seed', 0),
        )
        logger.log(f'Seed: {drive.seed}; round: {seed_round}')
        timings['id_mapping'] = time() - mapping_start
        timings['adapter_setup'] = time() - started

        output_setup_start = time()
        if record_spikes:
            import pyarrow as pa
            import pyarrow.parquet as pq

            schema = pa.schema([
                ('t', pa.float64()), ('time_ms', pa.float64()),
                ('trial', pa.int64()), ('neuron_index', pa.int64()),
                ('flywire_id', pa.int64()), ('exp_name', pa.string()),
            ])
            path_save = get_spike_output_path(exp_name, run_label, round_idx)
            path_save.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary_name = tempfile.mkstemp(
                prefix=f'.{exp_name}.', suffix='.parquet.tmp', dir=path_save.parent,
            )
            temporary_path = Path(temporary_name)
            os.close(fd)
            writer = pq.ParquetWriter(temporary_path, schema, compression='brotli')

        timings['output_setup'] = time() - output_setup_start

        def write_spikes(events: np.ndarray) -> None:
            nonlocal rows_written, sink_collection_s, sink_export_s
            for offset in range(0, len(events), _ARROW_ROWS):
                batch = events[offset:offset + _ARROW_ROWS]
                collection_start = time()
                try:
                    seconds = batch[:, 0].astype(np.float64) * model.dt_s
                    table = pa.Table.from_arrays([
                        pa.array(seconds), pa.array(seconds * 1000.0),
                        pa.array(batch[:, 1]), pa.array(batch[:, 2]),
                        pa.array(graph.neuron_ids[batch[:, 2]]),
                        pa.array([exp_name] * len(batch), type=pa.string()),
                    ], schema=schema)
                finally:
                    sink_collection_s += time() - collection_start
                export_start = time()
                try:
                    writer.write_table(table, row_group_size=_ARROW_ROWS)
                finally:
                    sink_export_s += time() - export_start
                rows_written += len(batch)

        run_start = time()
        try:
            native = run_batch(
                graph, steps, drive, trials=tuple(range(n_run)), model=model,
                chunk_size=_CHUNK_SIZE, record=record_spikes,
                round_idx=seed_round, silenced=silenced,
                spike_sink=write_spikes if record_spikes else None,
                max_recorded_events=1_000_000,
            )
        finally:
            timings['native_call'] = time() - run_start
        timings.update(native.timings)
        timings['network_creation_total'] = (
            timings['adapter_setup'] + native.timings['setup_compile_s']
            - native.timings['compile_warmup_s']
        )
        timings['device_build'] = native.timings['compile_warmup_s']
        timings['simulation_total'] = native.timings['simulation_s']
        timings['spike_extraction_total'] = max(
            0.0, native.timings['collection_s'] - sink_collection_s - sink_export_s,
        )

        summary_start = time()
        if record_spikes:
            if rows_written != int(np.sum(native.counts, dtype=np.int64)):
                raise RuntimeError('streamed spike count differs from native counts')
            result['n_spikes'] = rows_written
            result['n_active_neurons'] = int(
                np.count_nonzero(np.any(native.counts != 0, axis=0))
            )
        timings['result_collection'] = (
            sink_collection_s + native.timings['snapshot_s'] + time() - summary_start
        )

        publish_start = time()
        if writer is not None:
            writer.close()
            writer = None
            os.replace(temporary_path, path_save)
            temporary_path = None
            result['spike_path'] = str(path_save)
        timings['result_save'] = (
            timings['output_setup'] + sink_export_s + time() - publish_start
        )
        timings['total_elapsed'] = time() - started
        simulated_time = duration * n_run
        timings['realtime_ratio'] = (
            simulated_time / timings['simulation_total']
            if timings['simulation_total'] > 0 else float('inf')
        )
        timings['realtime_ratio_total'] = simulated_time / timings['total_elapsed']
        timings['phase_total'] = sum(timings[key] for key in (
            'network_creation_total', 'device_build', 'input_generation_staging_s',
            'simulation_total', 'spike_extraction_total',
            'result_collection', 'result_save',
        ))
        result['status'] = 'success'
        logger.log(f'Simulation: {timings["simulation_total"]:.3f}s; '
                   f'total: {timings["total_elapsed"]:.3f}s; '
                   f'spikes: {result["n_spikes"]}')
        if result['spike_path']:
            logger.log(f'Output file: {result["spike_path"]}')
    except Exception as error:
        timings['total_elapsed'] = time() - started
        result.update(status=f'error: {error}', n_spikes=0,
                      n_active_neurons=0, spike_path='')
        logger.log(f'ERROR: {error}')
        logger.log_raw(traceback.format_exc())
    finally:
        timings['sink_collection'] = sink_collection_s
        timings['sink_export'] = sink_export_s
        try:
            if writer is not None:
                writer.close()
        except Exception as error:
            result.update(status=f'error: parquet close failed: {error}',
                          n_spikes=0, n_active_neurons=0, spike_path='')
            logger.log(f'ERROR: parquet close failed: {error}')
        finally:
            try:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
            except OSError as error:
                result.update(status=f'error: temporary output cleanup failed: {error}',
                              n_spikes=0, n_active_neurons=0, spike_path='')
                logger.log(f'ERROR: temporary output cleanup failed: {error}')
        if result.get('status') != 'success':
            timings['total_elapsed'] = time() - started

    return result


def run_all_benchmarks(
    t_run_values: Sequence[float] | None = None,
    n_run_values: Sequence[int] | None = None,
    experiment: Mapping | None = None, logger: BenchmarkLogger | None = None,
    run_label: str | None = None, round_idx: int | None = None,
) -> list[dict[str, object]]:
    """Run the existing parameter grid through the optional Metal backend."""
    if t_run_values is None:
        t_run_values = T_RUN_VALUES_SEC
    if n_run_values is None:
        n_run_values = N_RUN_VALUES
    if experiment is None:
        experiment = get_experiment()
    if logger is None:
        logger = BenchmarkLogger()
    total_runs = len(t_run_values) * len(n_run_values)
    results = []
    for n_run in n_run_values:
        for t_run_sec in t_run_values:
            result = run_single_benchmark(
                t_run_sec, n_run, experiment, logger,
                run_idx=len(results) + 1, total_runs=total_runs,
                run_label=run_label, round_idx=round_idx,
            )
            results.append(result)
            save_result_csv(_BACKEND_NAME, result)
    print_summary_table(results, _BACKEND_NAME, logger)
    return results
