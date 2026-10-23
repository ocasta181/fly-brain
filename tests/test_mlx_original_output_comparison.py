"""Actual original CPU/native writers feed the three real comparison CLIs."""
from native_execution import forbid_network_cpu_steps, require_metal

import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

mx = require_metal('Metal is required', allow_module_level=True)
b = pytest.importorskip("brian2")

from mlx_backend.brian_device import original_api
from original_api_reference import original_runtime, load_original_model

pytestmark = pytest.mark.metal
ROOT = Path(__file__).resolve().parents[1]
DURATION = .0008
TRIALS = 2


def _write_original(directory, paths, ids, active, native, monkeypatch):
    """Observe real run_exp without replacing simulation or output generation."""
    original = load_original_model()
    snapshots = []
    get_spikes = original.get_spk_trn

    def observe(monitor):
        group = monitor.source
        snapshots.append({
            name: np.array(owner.variables[variable].get_value(), copy=True)
            for name, owner, variable in [
                *((name, group, name) for name in
                  ("v", "g", "rfc", "lastspike", "not_refractory")),
                ("i", monitor, "i"), ("t", monitor, "t"),
                ("count", monitor, "count"), ("N", monitor, "N"),
                ("timestep", group.clock, "timestep"),
            ]
        })
        return get_spikes(monitor)

    monkeypatch.setattr(original, "get_spk_trn", observe)
    backend = "mlx" if native else "brian2cpp"
    name = f"{backend}_t{DURATION}s_n{TRIALS}"
    directory.mkdir(parents=True, exist_ok=True)
    with original_runtime(original_api, seed=719, native=native) as device:
        with monkeypatch.context() as trap:
            if native:
                before_run = b.Network.before_run

                forbid_network_cpu_steps(b.Network, trap, 'CPU timestep code executed', before=before_run)
            original.run_exp(
                name, list(ids) if active else [], directory, *paths,
                params={**original.default_params, "t_run": DURATION*b.second,
                        "n_run": TRIALS, "r_poi": 10000*b.Hz}, n_proc=1,
            )
            if native:
                assert device.native_runs == TRIALS
                assert device.native_steps == 8*TRIALS
    assert len(snapshots) == TRIALS
    path = directory / f"{name}.parquet"
    frame = pd.read_parquet(path)
    assert list(frame.columns) == ["t", "trial", "flywire_id", "exp_name"]
    # Independent scheduling expectation: p=1 inputs arrive after thresholds;
    # reset clears input on spike steps, so spikes occur at 1,3,5,7.
    if active:
        assert set(frame.flywire_id) == set(ids)
        assert set(frame.trial) == {0, 1}
        assert set(frame.exp_name) == {name}
        for trial in range(TRIALS):
            for neuron_id in ids:
                times = frame.loc[(frame.trial == trial) &
                                  (frame.flywire_id == neuron_id), "t"].to_numpy()
                np.testing.assert_array_equal(times, np.arange(1, 8, 2)*.0001)
    else:
        assert frame.empty
        # This is the actual original empty writer's historical schema.
        assert frame.flywire_id.dtype == np.dtype("float64")
    return path, snapshots


def _cli(module, results, *arguments):
    # Each real CLI runs in a fresh process; only its result root is redirected.
    script = (
        "import sys,importlib; from pathlib import Path; "
        "sys.path.insert(0,sys.argv[1]); import benchmark; "
        "benchmark.path_res=Path(sys.argv[2]); "
        "module=importlib.import_module(sys.argv[3]); "
        "sys.argv=[module.__file__,*sys.argv[4:]]; module.main()"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(ROOT/"code"), str(results), module,
         "--run-label", "original", "--t_run", str(DURATION), "--n_run",
         str(TRIALS), *arguments], capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("kind", ["int64", "uint64", "string"])
def test_actual_original_writers_and_comparison_clis(tmp_path, monkeypatch, kind):
    ids = {
        "int64": [-7, 720575940624963786, 720575940630233916],
        "uint64": [2**63+1, 2**63+7, 2**64-1],
        "string": ["left", "middle", "right"],
    }[kind]
    nodes, edges = tmp_path/"neurons.csv", tmp_path/"edges.parquet"
    pd.DataFrame({"label": ["A", "B", "C"]},
                 index=pd.Index(ids, name="flywire_id")).to_csv(nodes)
    pd.DataFrame({"Presynaptic_Index": [0, 1, 2],
                  "Postsynaptic_Index": [0, 1, 2],
                  "Excitatory x Connectivity": [0., 0., 0.]}).to_parquet(edges)
    outputs = {}
    # Identifier coverage uses active outputs. Only string IDs need additional
    # simulations: the original empty schema discards every identifier dtype.
    for active in ([True, False] if kind == "string" else [True]):
        for native in (False, True):
            path, snapshots = _write_original(
                tmp_path/f"source-{active}-{native}", (nodes, edges), ids,
                active, native, monkeypatch,
            )
            outputs[active, native] = path
            if not native:
                expected = snapshots
            else:
                for actual, wanted in zip(snapshots, expected):
                    assert actual.keys() == wanted.keys()
                    for key in wanted:
                        assert actual[key].dtype == wanted[key].dtype, key
                        assert actual[key].shape == wanted[key].shape, key
                        assert actual[key].tobytes() == wanted[key].tobytes(), key
                cpu = pd.read_parquet(outputs[active, False]).drop(columns="exp_name")
                gpu = pd.read_parquet(path).drop(columns="exp_name")
                pd.testing.assert_frame_equal(gpu, cpu, check_exact=True)

    pairs = [(True, True)]
    if kind == "string":
        pairs += [(False, False), (True, False), (False, True)]
    results = tmp_path/"results"
    for round_idx, (cpu_active, gpu_active) in enumerate(pairs, 1):
        directory = results/"original"/f"round_{round_idx:02d}"
        directory.mkdir(parents=True)
        for active, native in ((cpu_active, False), (gpu_active, True)):
            source = outputs[active, native]
            shutil.copyfile(source, directory/source.name)
        cpu_count, gpu_count = 24*cpu_active, 24*gpu_active
        shared = cpu_active and gpu_active
        ground = directory/"ground.json"
        _cli("compare_ground_truth", results, "--round", str(round_idx), "-o", str(ground))
        metric = json.loads(ground.read_text())["comparisons"]["MLX (Apple GPU)"]
        assert metric["spike_count"] == {
            "ground_truth": cpu_count, "backend": gpu_count,
            "ratio": (gpu_count/cpu_count if cpu_count else None),
        }
        assert metric["active_neurons"]["jaccard"] == float(shared)
        assert metric["firing_rate"]["mean_pct_diff"] == (0. if shared else None)
        assert {row["flywire_id"] for row in metric["top_deviations"]} == (set(ids) if shared else set())

        pair_dir = directory/"pairwise"
        _cli("compare_spike_outputs", results, "--round", str(round_idx),
             "--tolerance-ms", "0.000001", "-o", str(pair_dir))
        pair, = json.loads((pair_dir/"pairwise_summary.json").read_text())
        assert (pair["spikes_a"], pair["spikes_b"]) == (cpu_count, gpu_count)
        assert pair["timing_matches"] == (24 if shared else 0)
        assert pair["timing_f1"] == float(shared)
        assert pair["active_jaccard"] == float(shared)
        assert pair["rate_rmse_hz_shared"] == (0. if shared else None)

    for include_timing in (False, True):
        destination = results/("timing" if include_timing else "rates")
        _cli("compare_backend_to_brian2", results, "--backend", "mlx", "--rounds",
             *(str(i) for i in range(1, len(pairs)+1)), "--tolerance-ms", "0.000001",
             "-o", str(destination), *(["--include-timing"] if include_timing else []))
        summaries = json.loads((destination/"mlx_vs_brian2_rate_summary.json").read_text())
        assert len(summaries) == len(pairs)
        for summary, (cpu_active, gpu_active) in zip(summaries, pairs):
            shared = cpu_active and gpu_active
            assert summary["spikes_reference"] == 24*cpu_active
            assert summary["spikes_comparison"] == 24*gpu_active
            assert summary["active_jaccard"] == float(shared)
            assert summary["rate_rmse_hz_shared"] == (0. if shared else None)
            assert summary["timing_matches"] == ((24 if shared else 0) if include_timing else None)
        parity = pd.read_csv(destination/"mlx_vs_brian2_rate_parity.csv", dtype={"flywire_id": str})
        assert set(parity.flywire_id) == {str(value) for value in ids}
        assert set(parity["round"]) == ({1, 3, 4} if kind == "string" else {1})
        for row in parity.itertuples():
            cpu_active, gpu_active = pairs[row.round-1]
            assert row.reference_rate_hz == 5000*cpu_active
            assert row.comparison_rate_hz == 5000*gpu_active
