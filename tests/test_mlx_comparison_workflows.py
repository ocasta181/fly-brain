"""Real comparison CLIs consume native MLX and original Brian output schemas."""
from state_assertions import assert_spikes_equal

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from native_execution import require_metal
mx = require_metal('Metal is required', allow_module_level=True)
pytest.importorskip("brian2")

from configured_reference import reference
from mlx_backend import configured
from mlx_backend.model import resolve_model
from mlx_backend.models import Connectome, Stimulus
from run_mlx import spike_table

pytestmark = pytest.mark.metal
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["int64", "uint64", "string"])
def paired_outputs(request, tmp_path):
    if request.param == "int64":
        ids = np.array([720575940624963786, 720575940630233916, -7], np.int64)
    elif request.param == "uint64":
        ids = np.array([2**63 + 1, 2**63 + 7, 2**64 - 1], np.uint64)
    else:
        ids = np.array(["left", "middle", "right"])
    c = Connectome(ids, np.array([0, 1, 2], np.int32),
                   np.array([1, 2, 0], np.int32),
                   np.array([400.125, -99.875, 200.25]), np.zeros(3))
    model = resolve_model({})
    events = np.zeros((2, 100, 3), np.uint8)
    for channel, interval in enumerate((3, 7, 11)):
        events[:, channel::interval, channel] = 1
    events[1, :20] = 0
    s = Stimulus(events, (0, 1, 2), (100., 100., 100.), (5, 7), 0, 0,
                 sha256(events.tobytes()).hexdigest(), model.dt_ms, (80., 80., 80.))
    expected = reference(c, s, model)
    actual = configured.run_prepared(configured.prepare(c, s, model), s)
    assert_spikes_equal(actual.spikes, expected.spikes)
    assert len(set(expected.spikes.neurons)) == 3

    output = tmp_path / "results" / "custom" / "round_01"
    output.mkdir(parents=True)
    spike_table(c, [actual], "mlx_t0.01s_n2").to_pandas().to_parquet(
        output / "mlx_t0.01s_n2.parquet", index=False)
    # Original run_exp has no neuron_index or explicit time-unit column.
    pd.DataFrame({
        "t": expected.spikes.steps.astype(np.float64) * model.dt_s,
        "trial": expected.spikes.trials,
        "flywire_id": ids[expected.spikes.neurons],
        "exp_name": "brian2cpp_t0.01s_n2",
    }).to_parquet(output / "brian2cpp_t0.01s_n2.parquet", index=False)
    return tmp_path / "results", ids, len(expected.spikes.steps)


def run_cli(module, results, *args):
    # Invoke each real main() in a fresh process, with only its result root
    # redirected so tests cannot touch a user's recorded benchmark outputs.
    code = (
        "import sys,importlib; from pathlib import Path; "
        "sys.path.insert(0,sys.argv[1]); import benchmark; "
        "benchmark.path_res=Path(sys.argv[2]); "
        "module=importlib.import_module(sys.argv[3]); "
        "sys.argv=[module.__file__,*sys.argv[4:]]; module.main()"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(ROOT / "code"), str(results), module,
         "--run-label", "custom", "--t_run", "0.01", "--n_run", "2", *args],
        text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_all_comparison_clis_preserve_identifiers_and_exact_results(paired_outputs):
    results, ids, count = paired_outputs
    ground_truth = results / "ground-truth.json"
    run_cli("compare_ground_truth", results, "--round", "1", "-o", str(ground_truth))
    metrics = json.loads(ground_truth.read_text())["comparisons"]["MLX (Apple GPU)"]
    assert metrics["spike_count"] == {"ground_truth": count, "backend": count, "ratio": 1.0}
    assert metrics["firing_rate"]["mean_pct_diff"] == 0
    assert {row["flywire_id"] for row in metrics["top_deviations"]} == set(ids)

    pair_dir = results / "pairwise"
    run_cli("compare_spike_outputs", results, "--round", "1", "--tolerance-ms",
            "0.000001", "-o", str(pair_dir))
    pair, = json.loads((pair_dir / "pairwise_summary.json").read_text())
    assert pair["timing_matches"] == count
    assert pair["timing_f1"] == 1
    assert pair["rate_rmse_hz_shared"] == 0

    for include_timing in (False, True):
        output = results / ("timed" if include_timing else "rates")
        options = ["--include-timing"] if include_timing else []
        run_cli("compare_backend_to_brian2", results, "--backend", "mlx",
                "--rounds", "1", "--tolerance-ms", "0.000001", "-o", str(output),
                *options)
        summary, = json.loads((output / "mlx_vs_brian2_rate_summary.json").read_text())
        assert summary["spikes_reference"] == summary["spikes_comparison"] == count
        assert summary["active_jaccard"] == 1
        assert summary["rate_rmse_hz_shared"] == 0
        assert summary["timing_matches"] == (count if include_timing else None)
        # Read as text: large uint64 identifiers must never pass through float.
        parity = pd.read_csv(output / "mlx_vs_brian2_rate_parity.csv",
                             dtype={"flywire_id": str})
        assert set(parity.flywire_id) == {str(value) for value in ids}


def test_legacy_pytorch_time_units_do_not_depend_on_latest_spike(tmp_path, monkeypatch):
    import benchmark
    from compare_spike_outputs import load_spikes

    monkeypatch.setattr(benchmark, "path_res", tmp_path)
    pd.DataFrame({"trial": [0, 0], "flywire_id": ["left", "right"],
                  "t": [0., .1]}).to_parquet(tmp_path / "pytorch_t1.0s_n1.parquet")
    spikes, _ = load_spikes("pytorch", 1., 1)
    np.testing.assert_array_equal(spikes.time_s, [0., .0001])
