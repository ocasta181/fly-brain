"""Bounded real-Metal packaging checks; invoked explicitly on Apple silicon."""

import builtins
import json
import shutil
import subprocess
import sys
from pathlib import Path

import benchmark
import mlx_engine
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from native_execution import require_metal
import run_mlx
from mlx_backend.experiments import SUGAR_IDS
from mlx_backend.models import Connectome

pytestmark = pytest.mark.metal
ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/data/mlx-replay"


@pytest.mark.parametrize("experiment", ["sugar", "p9"])
def test_full_network_saved_input_matches_independent_brian_native_fixture(
    tmp_path, monkeypatch, experiment
):
    original_import = builtins.__import__

    def mlx_only_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "brian2"}
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mlx_only_import)
    monkeypatch.setattr(benchmark, "path_res", tmp_path)
    fixture = FIXTURES / experiment
    result = run_mlx.run_single_benchmark(
        1.0,
        1,
        benchmark.get_experiment(experiment),
        benchmark.BenchmarkLogger(),
        run_label=experiment,
        round_idx=1,
        stimulus_path=fixture,
    )
    table = pq.read_table(result["spike_path"])
    with np.load(fixture / "spikes.npz", allow_pickle=False) as expected:
        steps = np.rint(table["t"].to_numpy() / 0.0001).astype(np.int64)
        rows = table["neuron_index"].to_numpy()
        np.testing.assert_array_equal(rows, expected["neuron_rows"])
        np.testing.assert_array_equal(steps, expected["steps"])
        np.testing.assert_array_equal(
            table["flywire_id"].to_numpy(),
            expected["flywire_ids"],
        )
        assert result["n_spikes"] == len(expected["steps"])
    metadata = json.loads(
        Path(result["spike_path"]).with_suffix(".mlx.json").read_text()
    )
    assert metadata["execution_mode"] == "sequential-fresh-singletons"
    assert metadata["seed_tuples"] == [[20261004, 0 if experiment == "sugar" else 1, 0]]
    assert result["timings"]["simulation_total"] > 0
    assert result["timings"]["total_elapsed"] >= result["timings"]["simulation_total"]


def test_multiple_trials_use_fresh_singletons_and_preserve_first_trial(
    tmp_path, monkeypatch
):
    c = Connectome(
        np.array(SUGAR_IDS, dtype=np.int64),
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([], dtype=np.float64),
    )
    monkeypatch.setattr(mlx_engine, "load_inputs", lambda *args: c)
    monkeypatch.setattr(benchmark, "path_res", tmp_path)
    single = run_mlx.run_single_benchmark(
        0.0064,
        1,
        benchmark.get_experiment(),
        benchmark.BenchmarkLogger(),
        run_label="one",
    )
    multiple = run_mlx.run_single_benchmark(
        0.0064,
        2,
        benchmark.get_experiment(),
        benchmark.BenchmarkLogger(),
        run_label="two",
    )
    first = pq.read_table(single["spike_path"])
    both = pq.read_table(multiple["spike_path"])
    selected = both.filter(pc.equal(both["trial"], 0))
    # Names differ only by the benchmark's trial-count suffix.
    for name in ("t", "time_ms", "trial", "neuron_index", "flywire_id"):
        np.testing.assert_array_equal(first[name].to_numpy(), selected[name].to_numpy())
    assert set(both["trial"].to_pylist()) == {0, 1}
    metadata = json.loads(
        Path(multiple["spike_path"]).with_suffix(".mlx.json").read_text()
    )
    assert len(metadata["per_trial_timings"]) == 2
    assert (
        metadata["canonical_event_sha256"][0] != metadata["canonical_event_sha256"][1]
    )


def test_real_cli_writes_common_csv_manifest_and_typed_spikes(tmp_path):
    # A standalone package copy keeps the upstream benchmark CSV and user's data intact.
    shutil.copytree(
        ROOT / "code", tmp_path / "code", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copy2(ROOT / "main.py", tmp_path / "main.py")
    (tmp_path / "data").mkdir()
    for name in ("2025_Completeness_783.csv", "2025_Connectivity_783.parquet"):
        (tmp_path / "data" / name).symlink_to(ROOT / "data" / name)
    result = subprocess.run(
        [
            sys.executable,
            str(tmp_path / "main.py"),
            "--mlx",
            "--t_run",
            "0.1",
            "--n_run",
            "1",
            "--no_log_file",
            "--run-label",
            "cli",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    (tmp_path / "command.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    import csv

    rows = list(csv.DictReader((tmp_path / "data/benchmark-results.csv").open()))
    assert len(rows) == 1 and rows[0]["backend_key"] == "mlx"
    assert rows[0]["status"] == "success"
    manifest = list(csv.DictReader((tmp_path / "data/results/cli/manifest.csv").open()))
    path = tmp_path / manifest[0]["spike_path"]
    table = pq.read_table(path)
    with np.load(FIXTURES / "sugar/spikes.npz", allow_pickle=False) as expected:
        keep = expected["steps"] < 1000
        np.testing.assert_array_equal(
            table["neuron_index"].to_numpy(), expected["neuron_rows"][keep]
        )
        np.testing.assert_array_equal(
            np.rint(table["t"].to_numpy() / 0.0001).astype(np.int64),
            expected["steps"][keep],
        )


def test_tiny_cli_cpp_and_native_outputs_are_portable(tmp_path):
    """Real CLI/build/export smoke check; independent draws are not a parity test."""
    import csv
    import pandas as pd

    mx = require_metal('Metal is required')
    shutil.copytree(ROOT / "code", tmp_path / "code",
                    ignore=shutil.ignore_patterns("__pycache__", "output*"))
    shutil.copy2(ROOT / "main.py", tmp_path / "main.py")
    data = tmp_path / "data"
    data.mkdir()
    # Reuse the tracked preset IDs with one additional postsynaptic neuron.
    # Only22neurons/oneedge; no production dataset, symlink or private path.
    ids = [*SUGAR_IDS, max(SUGAR_IDS) + 1]
    pd.DataFrame(index=pd.Index(ids, name="flywire_id")).to_csv(
        data / "2025_Completeness_783.csv")
    pd.DataFrame({"Presynaptic_Index": [0], "Postsynaptic_Index": [21],
                  "Excitatory x Connectivity": [200.]}).to_parquet(
        data / "2025_Connectivity_783.parquet")
    # This launcher changes build parallelism/cache location only. The actual
    # main.py parser, dispatchers, CPU standalone runner and native runner execute.
    launcher = """
import runpy, sys
from pathlib import Path
import brian2 as b
b.prefs.devices.cpp_standalone.extra_make_args_unix = ['-j2']
b.prefs.codegen.runtime.cython.cache_dir = str(Path.cwd() / 'cython-cache')
sys.argv = ['main.py', *sys.argv[1:]]
runpy.run_path('main.py', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", launcher, "--brian2-cpu", "--mlx",
         "--t_run", "0.1", "--n_run", "1", "--no_log_file",
         "--run-label", "portable"],
        cwd=tmp_path, text=True, capture_output=True, timeout=80, check=False,
    )
    (tmp_path / "command.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Building C++ standalone code" in result.stdout
    assert (tmp_path / "code/output/main").is_file()
    with (data / "benchmark-results.csv").open() as source:
        rows = list(csv.DictReader(source))
    assert {row["backend_key"] for row in rows} == {"brian2cpp", "mlx"}
    assert len(rows) == 2 and all(row["status"] == "success" for row in rows)
    with (data / "results/portable/manifest.csv").open() as source:
        manifest = list(csv.DictReader(source))
    assert len(manifest) == 2
    for row in manifest:
        path = tmp_path / row["spike_path"]
        table = pq.read_table(path)
        assert table.num_rows > 0
        assert table.schema.field("trial").type == pa.int64()
        assert table.schema.field("neuron_index").type == pa.int64()
        assert table.schema.field("time_ms").type == pa.float64()
        assert set(table["flywire_id"].to_pylist()) <= set(ids)
        assert table.num_rows == int(row["spikes"])
        assert np.all(table["trial"].to_numpy() == 0)
        assert np.all((table["time_ms"].to_numpy() >= 0)
                      & (table["time_ms"].to_numpy() < 100))
    sidecars = list((data / "results/portable").rglob("*.mlx.json"))
    assert len(sidecars) == 1
    metadata = json.loads(sidecars[0].read_text())
    assert metadata["compilation"] == "custom-metal-jit"
    assert metadata["device"] and metadata["record_spikes"]
    assert len(metadata["canonical_event_sha256"]) == 1
