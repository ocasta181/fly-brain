# FlyBrain Metal64

**Float64 FlyWire v783 simulation on Apple silicon.**

Simulate a leaky integrate-and-fire model of the adult fruit fly brain:
138,639 neurons and 15,091,983 weighted edges from the
[FlyWire](https://flywire.ai/) v783 connectome. Stimulate neurons, suppress their
outgoing synapses, and inspect the resulting spike times and firing rates.
Native Metal kernels run through [MLX](https://github.com/ml-explore/mlx), with
software binary64 arithmetic from [VF64-metal](https://github.com/Lulzx/VF64-metal).

FlyBrain Metal64 is an independently maintained fork of
[Eon Systems' fly-brain](https://github.com/eonsystemspbc/fly-brain), focused on
running its fixed LIF model on Apple silicon.

## Install

Use macOS on Apple silicon with [uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/ocasta181/fly-brain.git fly-brain-metal64
cd fly-brain-metal64
uv venv --python 3.10.14
uv pip install -r requirements-mlx.txt
uv run --no-project python code/mlx_backend/vf64.py --install
```

The last command installs the pinned VF64 shader and license outside this
repository. Simulation verifies their hashes and reads them without network
access. To use an existing checkout, set `VF64_SOURCE_DIR` to VF64-metal commit
`8dae385a16493282e2a3e90e4cd1aa1250b36dc5`.

## Run

```sh
uv run --no-project python main.py
```

The default runs one Sugar stimulation trial for 0.1 biological seconds on the
Apple GPU, with spike recording and Parquet export enabled. For a one-second
P9 stimulation with four independent trials:

```sh
uv run --no-project python main.py --mlx --experiment p9 --t_run 1 --n_run 4 --run-label p9
```

The console reports the spike file path. Timing rows are saved to
`data/benchmark-results.csv`; labeled runs also produce a manifest under
`data/results/<label>/`. Spike Parquet files contain one row per event, with
`time_ms`, `trial`, `neuron_index`, `flywire_id`, and `exp_name` columns. `t`
retains the event time in seconds for compatibility.
Unlabeled MLX runs receive unique timestamped labels. If you supply
`--run-label`, choose a new value for each saved run; existing spike files are
protected from overwrite.

Use `--disable-spike-io` for counts without spike recording or export. Run
`uv run --no-project python main.py --help` for the full CLI.

## Stimulate, silence, and inspect spikes

```sh
uv run --no-project python examples/stimulation.py
```

The [example](examples/stimulation.py) stimulates P9 at 100 Hz for 0.1 seconds
in a control and an outgoing-silenced condition, using the same Poisson input
in both. It exports spikes to Parquet, reads them back, and prints event and
active-cell counts plus the first spike rows. Outgoing silencing sets the
selected neurons' outgoing synaptic weights to zero; those neurons can still
receive input and spike.

## Performance

On an M5 Max using AC power, warm full-v783 simulations of one biological
second were **7.04–13.28× faster than Brian2 CPU** and **1.19–2.38× faster than
the [Rust F32 implementation](https://github.com/mehrantsi/flyBrain)**. These
are paired geometric means from 20 rounds per comparison across Sugar and P9
stimulation with one or four trials. Setup and compilation are excluded.

![Warm full-v783 speedups on an AC-powered M5 Max, across Sugar and P9 with one or four trials](https://raw.githubusercontent.com/ocasta181/fly-brain/5071c55a217348cdbaa5f2e637b23c507e1783bc/mlx-throughput-pr11-20261027.png)

The [summary CSV](https://raw.githubusercontent.com/ocasta181/fly-brain/5071c55a217348cdbaa5f2e637b23c507e1783bc/mlx-throughput-pr11-20261027.csv)
contains the means and simultaneous 95% confidence intervals. All eight lower
bounds exceed 1×. Measurements use runtime source commit
[`c36e7f6`](https://github.com/ocasta181/fly-brain/commit/c36e7f6f698a44d4a54d179c2dfd5bd28f3ce48d).

Both sides of the CPU comparison time spike recording and Brotli Parquet
export through file close. Its one-trial baseline is the original Brian2 C++
standalone implementation; four trials use Brian2 Cython with four workers.
The Rust comparison is count-only and excludes export on both sides. Rust
also has different refractory behavior and spike trajectories, so its result
compares implementations rather than isolating the cost of float64.

## Accuracy and scope

The [maintained tests](tests/test_mlx_fixed_model.py) compare actual native
Metal execution with an independent Brian2 2.8.0 C++ standalone reference on
Apple arm64. Under identical supplied input tapes, they require matching raw
float64 state bits and ordered spikes, including delayed propagation,
refractory behavior, silencing, and independent trials.

This backend implements the fixed Eon LIF model, with a default 0.1 ms
timestep. It does not execute arbitrary Brian2 equations. Its seeded PCG64
Poisson inputs are reproducible by round and trial, but differ from Brian2's
random streams; matching seeds alone do not produce an exact cross-backend
comparison.

To run the maintained qualification, install the test dependencies in the
same environment:

```sh
uv pip install -r requirements-mlx-test.txt
uv run --no-project pytest -q
```

## Original backends and attribution

The inherited Brian2, Brian2CUDA, PyTorch, NEST GPU, GeNN, and Brian2GeNN
backends remain available through explicit CLI flags. Their setup and
benchmark workflows are documented in the
[original project](https://github.com/eonsystemspbc/fly-brain#installation).
The Apple installation above supplies the Metal backend's dependencies.

The model is based on Shiu et al.,
[*A leaky integrate-and-fire computational model based on the connectome of
the entire adult Drosophila brain reveals insights into sensorimotor
processing*](https://www.biorxiv.org/content/10.1101/2023.05.02.539144v1)
([published version, Nature 2024](https://www.nature.com/articles/s41586-024-07763-9)).

GPU propagation, recording, and bounded export mechanics were adapted from
[Kisame76/drosophila-brain-mlx](https://github.com/Kisame76/drosophila-brain-mlx)
through the independent float32 Eon implementation. VF64-metal supplies the
unchanged external arithmetic shader. See the
[third-party notices](code/mlx_backend/third_party/NOTICE) for revisions and
license terms.

## License

This project retains [GPL-2.0-or-later](LICENSE), except where otherwise
noted. Third-party components retain their original licenses, including the
[MIT-licensed Shiu et al. materials](code/paper-phil-drosophila/LICENSE),
[VF64's BSD-3-Clause license](code/mlx_backend/third_party/VF64-LICENSE), and
the [Kisame runtime license](code/mlx_backend/third_party/Kisame76-LICENSE).
