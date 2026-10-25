# Float64 MLX performance and verification

Measured October 24, 2026, on an Apple M5 Max with 128 GiB RAM, macOS 26.4,
battery power and default power mode (`powermode 0`). The tested implementation
is commit `1225f248299fa40ca1febb1a064163fb52c03588`.

## Warm performance

Each trial runs one biological second on the complete FlyWire v783 graph:
138,639 neurons and 15,091,983 connections. Each case uses three fixed adjacent
CPU/MLX pairs in CPU-first, MLX-first, CPU-first order, after disposable warm-ups.
The base seed is `20261030`. All trials record complete six-column Brotli Parquet.

The CPU uses the original C++ standalone route for one trial. Four trials use
the original 14-slot loky configuration with four prepared Brian Cython workers.
MLX uses the production adapter and VF64 Metal engine, with distinct PCG64 inputs.
Brian and PCG64 streams differ; this performance cohort does not test input parity.

Warm time includes input generation, simulation, recording, result collection,
complete output and publication. Native time also includes physical-state snapshots
and engine cleanup. It excludes data/graph setup, compilation and disposable warm-up.

| Case | Mean CPU s | Mean MLX s | Wall change | Paired speedup | Simultaneous 95% lower | MLX aggregate × real time |
|---|---:|---:|---:|---:|---:|---:|
| Sugar, B1 | 1.671 | 0.507 | -69.7% | 3.30× | 2.93× | 1.97× |
| P9, B1 | 1.691 | 0.423 | -75.0% | 4.02× | 2.88× | 2.37× |
| Sugar, B4 | 2.575 | 1.384 | -46.3% | 1.86× | 1.74× | 2.89× |
| P9, B4 | 2.409 | 1.206 | -49.9% | 2.00× | 1.76× | 3.32× |

Means are arithmetic. Wall change is `100*(mean_MLX/mean_CPU - 1)`.
Speedup is the geometric mean of the three paired CPU/MLX wall-time ratios.
Bounds use log ratios, Student t with two degrees of freedom, and Bonferroni
adjustment across four one-sided claims (`t=6.205346816570693`). All lower bounds
exceed 1.0. Aggregate real-time throughput is `trials / mean_MLX_seconds`; it is
not the latency of one trial within a batch. These are same-Mac CPU comparisons;
they do not establish an advantage over GeNN/CUDA.

![Warm CPU and MLX comparisons](mlx-performance.png)

Fresh-worker MLX preparation plus compilation took 7.20–8.54 seconds across
the measured runs. C++ compilation took 3.01–7.34 seconds. The Cython comparison
used a private verified binary cache; its worker preparation occurs before the
common release barrier. Warm speedups do not describe cold start.

## Individual paired observations

| Case | Round | Order | CPU s | MLX s |
|---|---:|---|---:|---:|
| Sugar, B1 | 1 | cpu / vf64 | 1.619568 | 0.483788 |
| Sugar, B1 | 2 | vf64 / cpu | 1.756610 | 0.552522 |
| Sugar, B1 | 3 | cpu / vf64 | 1.637467 | 0.483634 |
| P9, B1 | 1 | cpu / vf64 | 1.672505 | 0.396187 |
| P9, B1 | 2 | vf64 / cpu | 1.725301 | 0.477850 |
| P9, B1 | 3 | cpu / vf64 | 1.675323 | 0.393735 |
| Sugar, B4 | 1 | cpu / vf64 | 2.518064 | 1.382154 |
| Sugar, B4 | 2 | vf64 / cpu | 2.626200 | 1.388987 |
| Sugar, B4 | 3 | cpu / vf64 | 2.580385 | 1.380725 |
| P9, B4 | 1 | cpu / vf64 | 2.409504 | 1.226014 |
| P9, B4 | 2 | vf64 / cpu | 2.392425 | 1.149931 |
| P9, B4 | 3 | cpu / vf64 | 2.425934 | 1.242246 |

The initial AC-only preflight stopped before collecting observations. The battery
cohort then completed both B1 cases but stopped on an administrative process-liveness
probe after a successful B4 CPU disposable warm-up. A source-bound continuation
revalidated every completed B1 pair and collected the predeclared B4 cases.
Only the disposable warm-up repeated. No primary sample was retried, added or
dropped. Both watchdogs confirmed that all owned process groups were gone.
The numerical sources, observation code, seeds, pair order and inference stayed
identical. Raw receipts, failed attempts and cumulative runtime remain in ignored
`executions/vf64-replacement-20261024/`; no campaign runner is shipped.

## Correctness and resources

Actual Brian2 2.8.0 C++ execution and actual Metal execution agreed on raw binary64
state, refractory flags, counts, pending delay state and ordered spikes under
identical supplied inputs. Controls include original, nondefault and zero-rest
parameters, distinct trials, duplicate channels/edges, cancellation and boundaries.
The full v783 shared-input check covers 43 ticks and 68 ordered spikes.

A 43-tick full graph resource check exercises B1, B4 and B32. B32 peak active Metal
memory was 873,029,468 bytes; active allocations returned to zero after release.
The stable allocator cache retained 951,698,272 bytes. Streaming matched native
counts and event order. Recording-off, repeat runs, and recovery after an injected
sink failure preserved the same physical state. These are allocation observations,
not B32 throughput measurements.

The maintained regressions also exercise native adapter output, dependency integrity,
metadata rollback, bounded dense extraction, and preserved default CLI dispatch.
Numerical qualification uses NumPy 1.26.4, MLX/Metal 0.32.3, Brian2 2.8.0,
Python 3.10.14 and Apple Clang 21. The C++ flags are `-O3 -ffast-math
-fno-finite-math-only -march=native -std=c++11`.

VF64 source is pinned to `8dae385a16493282e2a3e90e4cd1aa1250b36dc5`.
CPU numerical source is unchanged: `code/run_brian2_cuda.py` SHA-256
`0184fb8bbd28e36643a0bcffbe64675d58cb1bcce27cbe1fc08deb8cb1777722`.
