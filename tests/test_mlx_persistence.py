"""Metadata rollback and adapter recovery without native Metal execution."""

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def persistence_case(tmp_path, monkeypatch):
    import benchmark
    import run_mlx

    monkeypatch.setattr(benchmark, "path_res", tmp_path / "results")
    monkeypatch.setattr(benchmark, "csv_path", tmp_path / "benchmark.csv")
    monkeypatch.delenv(benchmark.SPIKE_IO_ENV_VAR, raising=False)
    logger = SimpleNamespace(log=lambda *args: None, log_raw=lambda *args: None)
    experiment = {
        "name": "Persistence case", "key": "persistence", "neu_exc": [17],
        "neu_exc2": [], "neu_slnc": [], "stim_rate": 100.0,
    }
    return SimpleNamespace(
        benchmark=benchmark, adapter=run_mlx, logger=logger,
        experiment=experiment, root=tmp_path,
    )


@pytest.fixture
def fake_runtime(persistence_case, monkeypatch):
    import numpy as np
    import pandas as pd
    from mlx_backend import engine

    case = persistence_case
    completeness = case.root / "completeness.csv"
    connectivity = case.root / "connectivity.parquet"
    pd.DataFrame({"complete": [True]}, index=[17]).to_csv(completeness)
    pd.DataFrame({name: pd.Series(dtype="int64") for name in (
        "Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity",
    )}).to_parquet(connectivity, index=False)
    monkeypatch.setattr(case.adapter, "path_comp", completeness)
    monkeypatch.setattr(case.adapter, "path_con", connectivity)

    def run_batch(graph, steps, drive, **options):
        trials = options["trials"]
        if options["record"]:
            options["spike_sink"](np.array([[1, trial, 0] for trial in trials]))
        timings = dict.fromkeys((
            "setup_compile_s", "compile_warmup_s", "runtime_s", "simulation_s",
            "input_generation_staging_s", "collection_s", "snapshots_s",
        ), 0.0)
        timings["simulation_s"] = 0.000001
        return SimpleNamespace(
            counts=np.ones((len(trials), graph.n_neurons), dtype=np.int64),
            timings=timings,
        )

    monkeypatch.setattr(engine, "run_batch", run_batch)


def _prior_result(case):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = case.benchmark.path_res / "persist" / "unrelated.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"preserve": [7]}), path)
    result = {
        "n_run": 9, "t_run_sec": 0.1, "backend_key": "mlx",
        "experiment_name": case.experiment["name"], "experiment_key": "persistence",
        "run_label": "persist", "round": 3, "spike_path": str(path),
        "n_spikes": 2, "n_active_neurons": 1, "status": "success", "timings": {},
    }
    case.benchmark.save_result_csv(case.benchmark.BACKEND_NAMES["mlx"], result)
    return result


def _files(root):
    return {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _rows(path):
    with path.open(newline="") as saved:
        return list(csv.DictReader(saved))


def _fail_metadata_once(case, monkeypatch, target, operation):
    benchmark = case.benchmark
    failed = False
    if operation == "write":
        real_writerows = csv.DictWriter.writerows
        columns = benchmark.CSV_COLUMNS if target == "csv" else benchmark.MANIFEST_COLUMNS

        def writerows(writer, rows):
            nonlocal failed
            if not failed and writer.fieldnames == columns:
                failed = True
                writer.writerow(dict.fromkeys(columns, "partial"))
                raise OSError(f"injected {target} write failure")
            return real_writerows(writer, rows)

        monkeypatch.setattr(csv.DictWriter, "writerows", writerows)
    else:
        real_replace = benchmark.os.replace
        destination = (benchmark.csv_path if target == "csv" else
                       benchmark.path_res / "persist" / "manifest.csv")

        def replace(source, path):
            nonlocal failed
            if not failed and Path(path) == destination:
                failed = True
                raise OSError(f"injected {target} replace failure")
            return real_replace(source, path)

        monkeypatch.setattr(benchmark.os, "replace", replace)


@pytest.mark.parametrize("target", ["csv", "manifest"])
@pytest.mark.parametrize("operation", ["write", "replace"])
def test_metadata_failure_preserves_prior_files(
    persistence_case, monkeypatch, target, operation,
):
    case = persistence_case
    prior = _prior_result(case)
    before = _files(case.root)
    _fail_metadata_once(case, monkeypatch, target, operation)
    with pytest.raises(OSError, match=f"injected {target} {operation} failure"):
        case.benchmark.save_result_csv(
            case.benchmark.BACKEND_NAMES["mlx"], dict(prior, n_run=1),
        )
    assert _files(case.root) == before
    assert not list(case.root.rglob(".*"))


def test_failed_manifest_publication_removes_new_csv(persistence_case, monkeypatch):
    case = persistence_case
    _fail_metadata_once(case, monkeypatch, "manifest", "replace")
    result = {
        "n_run": 1, "t_run_sec": 0.0002, "run_label": "persist", "round": 3,
        "spike_path": str(case.benchmark.path_res / "new.parquet"),
    }
    with pytest.raises(OSError, match="injected manifest replace failure"):
        case.benchmark.save_result_csv(case.benchmark.BACKEND_NAMES["mlx"], result)
    assert not case.benchmark.csv_path.exists()
    assert not (case.benchmark.path_res / "persist" / "manifest.csv").exists()
    assert not list(case.root.rglob(".*"))


def test_grid_recovers_owned_output_continues_and_allows_retry(
    persistence_case, fake_runtime, monkeypatch,
):
    case = persistence_case
    prior = _prior_result(case)
    original_parquet = Path(prior["spike_path"]).read_bytes()
    _fail_metadata_once(case, monkeypatch, "manifest", "replace")
    failed_path = case.benchmark.get_spike_output_path("mlx_t0.0002s_n1", "persist", 3)
    later_path = case.benchmark.get_spike_output_path("mlx_t0.0003s_n1", "persist", 3)
    failed, later = case.adapter.run_benchmarks(
        [0.0002, 0.0003], [1], case.experiment, case.logger,
        run_label="persist", round_idx=3,
    )
    assert "result metadata save failed: injected manifest replace failure" in failed["status"]
    assert failed["n_spikes"] == failed["n_active_neurons"] == 0
    assert not failed["spike_path"] and not failed_path.exists()
    assert later["status"] == "success" and later_path.exists()
    assert Path(prior["spike_path"]).read_bytes() == original_parquet
    manifest_path = case.benchmark.path_res / "persist" / "manifest.csv"
    for metadata in (case.benchmark.csv_path, manifest_path):
        rows = _rows(metadata)
        assert len(rows) == 2 and all(row["status"] == "success" for row in rows)
        assert {row["spike_path"] for row in rows} == {prior["spike_path"], str(later_path)}

    retried, = case.adapter.run_benchmarks(
        [0.0002], [1], case.experiment, case.logger, run_label="persist", round_idx=3,
    )
    assert retried["status"] == "success" and failed_path.exists()
    assert len(_rows(case.benchmark.csv_path)) == len(_rows(manifest_path)) == 3
    assert not list(case.root.rglob(".*"))


def test_metadata_failure_does_not_remove_existing_collision(
    persistence_case, fake_runtime, monkeypatch,
):
    case = persistence_case
    _prior_result(case)
    published = case.adapter.run_single_benchmark(
        0.0002, 1, case.experiment, case.logger, run_label="persist", round_idx=3,
    )
    assert published["status"] == "success"
    before = _files(case.root)
    _fail_metadata_once(case, monkeypatch, "csv", "write")
    collision, = case.adapter.run_benchmarks(
        [0.0002], [1], case.experiment, case.logger, run_label="persist", round_idx=3,
    )
    assert "already exists" in collision["status"]
    assert "result metadata save failed" in collision["status"]
    assert _files(case.root) == before
    assert not list(case.root.rglob(".*"))


def test_failed_metadata_rollback_retains_published_output(
    persistence_case, fake_runtime, monkeypatch,
):
    case = persistence_case
    prior = _prior_result(case)
    prior_parquet = Path(prior["spike_path"]).read_bytes()
    real_replace = case.benchmark.os.replace
    manifest = case.benchmark.path_res / "persist" / "manifest.csv"
    publication_failed = False

    def replace(source, destination):
        nonlocal publication_failed
        if Path(destination) == manifest:
            publication_failed = True
            raise OSError("manifest unavailable")
        if publication_failed:
            raise OSError("rollback unavailable")
        return real_replace(source, destination)

    monkeypatch.setattr(case.benchmark.os, "replace", replace)
    result, = case.adapter.run_benchmarks(
        [0.0002], [1], case.experiment, case.logger, run_label="persist", round_idx=3,
    )
    assert "metadata rollback failed: rollback unavailable" in result["status"]
    assert Path(result["spike_path"]).exists()
    assert Path(prior["spike_path"]).read_bytes() == prior_parquet


@pytest.mark.parametrize("run_label", [None, "rounds"])
def test_round_zero_and_absent_round_keep_distinct_metadata_and_paths(
    persistence_case, fake_runtime, run_label,
):
    case = persistence_case
    results = []
    for round_idx in (None, 0):
        result, = case.adapter.run_benchmarks(
            [0.0002], [1], case.experiment, case.logger,
            run_label=run_label, round_idx=round_idx,
        )
        assert result["status"] == "success"
        assert result["round"] == ("" if round_idx is None else 0)
        expected = case.benchmark.get_spike_output_path("mlx_t0.0002s_n1", run_label, round_idx)
        assert result["spike_path"] == str(expected) and expected.exists()
        results.append(result)
    assert results[0]["spike_path"] != results[1]["spike_path"]
    assert type(results[1]["round"]) is int
    case.benchmark.save_result_csv(case.benchmark.BACKEND_NAMES["mlx"], results[1])
    rows = _rows(case.benchmark.csv_path)
    assert len(rows) == 2 and {row["round"] for row in rows} == {"", "0"}
    if run_label:
        rows = _rows(case.benchmark.path_res / run_label / "manifest.csv")
        assert len(rows) == 2 and {row["round"] for row in rows} == {"", "0"}
