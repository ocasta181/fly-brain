"""Compare P9 stimulation with and without P9 outgoing synaptic transmission.

From the repository root: uv run --no-project python examples/stimulation.py
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from benchmark import BenchmarkLogger, get_experiment, spike_io_enabled  # noqa: E402
from run_mlx import run_single_benchmark  # noqa: E402


def main() -> None:
    import pandas as pd

    if not spike_io_enabled():
        raise SystemExit("Unset FLY_BRAIN_DISABLE_SPIKE_IO to export example spikes.")

    p9 = get_experiment("p9") | {"seed": 0}
    label = "stimulation_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    logger = BenchmarkLogger()
    conditions = (
        ("control", []),
        ("outgoing_silenced", p9["neu_exc"]),
    )

    print("P9 stimulation: 100 Hz, 0.1 seconds, one trial per condition.")
    print("Outgoing silencing zeros P9 synaptic output; P9 cells can still spike.")
    # The input RNG uses seed, round and trial, independently of the run label.
    for condition, silenced in conditions:
        experiment = p9 | {"neu_slnc": silenced}
        result = run_single_benchmark(
            0.1, 1, experiment, logger,
            run_label=f"{label}_{condition}", round_idx=0,
        )
        if result["status"] != "success":
            raise SystemExit(f"{condition}: {result['status']}")

        path = Path(result["spike_path"])
        events = pd.read_parquet(path)
        print(
            f"\n{condition}: {len(events):,} spike events, "
            f"{events['flywire_id'].nunique():,} active cells"
        )
        print(f"Output: {path}")
        preview = events[["time_ms", "trial", "flywire_id"]].head(5)
        print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
