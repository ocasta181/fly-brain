# Native MLX support: maintainer review

The opt-in backend runs supported Eon simulations on Apple silicon Metal while
preserving the CPU default, original Brian objects and incremental outputs.
This is a substantial native subsystem; the source map below separates its
integration, execution, arithmetic and independent proofs.

## Value and comparator scope

| Evidence | Result | Comparator and limits |
|---|---|---|
| Representative timing | 16/16 cases meet the 2× median floor: 2.029962–7.279297×. One meets the 5× target; 15 meet the accepted 2–5× band. | Independent Brian2 **Cython runtime** workers versus native Metal; matched inputs, work and recording; one warmup/five fresh pairs and reported 95% intervals. Desktop qualification, without quiet-machine qualification. |
| Numerical correctness | 96 exact matched state/spike pairs and two recorded preflights for recording-off. | Exact raw bits, ordered spikes and pending-state equivalence in the named cases. |
| Native workflows | 28 observations, 27 recorded cells, 54 exports; four resumed rounds preserve their prefixes; largest 1000s/32-trial recording on/off equality. | Two default and 76 paper cross-product cells remain unrun. The original omitted-default API (`1s`, `30` trials, `n_proc=-1`) completed on the full graph with native workers and 38190 output rows; this proves workflow/output integrity, separately from numerical parity. |
| Independent-input P9 | Passes at 9216 trials per arm with fixed margins and Brian controls. | Scientific runtime remains `5feffb5dcd32`; later applicability rests on scoped source review. |
| Semantic and recording regressions | 22 named complex/cast/NumPy-action/failure-prefix/RNG checks; six recorder and four custom arithmetic/clock-guard checks. | Each group qualifies its identified repair and target/dtype cases. |
| IEEE parameters and delays | 20 physical-parameter pairs plus ten delay items: 29 CPU/Metal pairs and one guard in total. | The initialization repair preserves accepted values such as `v_0=NaN*mV`. Checks retain exact raw snapshots, spikes, clocks, RNG and continuation; raw delay identities and metadata remain distinct. |

Eon's CLI uses **C++ standalone for one CPU trial** and runtime/joblib for
multiple CPU trials. The 16-case result is a Brian Cython runtime comparison.
The tiny check below exercises the actual one-trial C++ CLI build/output path
and native CLI path with independent Poisson draws. Separate tests compare
exact NumPy continuation and real C++ state, recording and pending events.

Measurements retain their original source identities. Earlier finite-workload
qualification through `cead41b4ac4b`, later focused repairs and the independent
review support qualified baseline `7033ff667b0196ec13de8bcc342fdff6ba70a291` through
source-applicability proofs. This is not a fresh campaign over every accepted
configuration. Resulting source identities, comparisons and receipts stay in
the separate private handoff.

## Boundary coverage and comparison rules

The physical-parameter pairs cover six NumPy64 NaN cases and 14
NumPy32/Cython64/Cython32 boundary cases.

At positive timesteps, NaN and negative delays, including `-Inf`, use zero
effective arrival delay, as the original CPU route does; `+Inf` is rejected.
The ten delay items comprise four NumPy/Cython NaN-delay pairs at 32/64 bits,
four finite-negative pairs spanning `-0.025ms` and `-0.2ms`, one NumPy64
negative-infinite pair and one guard/identity check.

Twenty original-API checks cover empty clocks, optional reset, unused namespace
and signed clocks: 16 exact snapshot/RNG pairs and four error equivalences.
The five signed-clock NumPy64 pairs include negative-timestep `+Inf` delay,
negative absolute indices and live delayed continuation after an empty rewind.
The positive-timestep rejection rule does not apply to those accepted cases.

Eighteen configurations were frozen before inspection: 17 actual native
simulations and one matching Brian preflight rejection passed. A post-freeze
reset-error case and 77 NumPy division-dispatch pairs qualify the error-message
repair. Signed error-prefix checks preserve failure ordering at tick −1 and
across −1→0. Earlier failed witnesses remain beside their focused repair proofs
in the private evidence.

Numerical comparisons preserve exact finite bits, ordered spikes, clocks and
random streams. Only proven allocation-dependent NaN payload differences may
compare by NaN status; deterministic payloads keep exact checks. Named cases
do not establish every object graph, dtype, expression or parameter combination.
Generic user Python callbacks remain outside Eon's contract and are rejected.

## Tiny clean-checkout check

Requires native Apple silicon macOS, working Metal, Python 3.10.14, `uv` and Xcode
Command Line Tools (`clang++`/`make`). From the checkout root:

```sh
uv venv --python 3.10.14 .venv
uv pip install --python .venv/bin/python -r requirements-mlx-verify.txt
.venv/bin/python -c 'import mlx.core as mx; assert mx.metal.is_available()'
.venv/bin/python -m pytest -q \
  'tests/test_mlx_brian_device.py::test_real_original_objects_and_requested_vs_clock_time[durations1]' \
  tests/test_mlx_components.py::test_actual_cpp_component_state_recording_and_continuation \
  tests/test_mlx_metal.py::test_tiny_cli_cpp_and_native_outputs_are_portable
```

The command must report **3 passed**, with no skips. Inputs are generated from
tracked fixtures/helpers: 4–5 neurons for exact checks and 22 for the CLI, with no
production dataset, private path or pre-existing environment. The CLI builds
with two compiler jobs and writes only to its temporary package copy. This
check was installed and run from a fresh tracked-source snapshot. Both exact
checks passed. The CLI check passed after a test-only relative-path correction
and one-selector retry in that same environment; the first failed receipt is
retained. No existing environment or production dataset was used.

## Review in coherent groups

| Boundary | Review entrypoints |
|---|---|
| Optional integration and outputs | [`main.py`](../main.py), [`run_mlx.py`](../code/run_mlx.py), [`mlx_engine.py`](../code/mlx_engine.py), [`runner.py`](../code/mlx_backend/runner.py), dependency pins |
| Original Brian objects, continuation and failure replay | [`brian_device.py`](../code/mlx_backend/brian_device.py), [`brian_errors.py`](../code/mlx_backend/brian_errors.py) |
| Model, compilation and expressions | [`model.py`](../code/mlx_backend/model.py), [`compiler.py`](../code/mlx_backend/compiler.py), [`expression_lowering.py`](../code/mlx_backend/expression_lowering.py), arithmetic/policy modules |
| Native state, scheduling, inputs and recording | [`configured.py`](../code/mlx_backend/configured.py), [`components.py`](../code/mlx_backend/components.py), input/RNG/export modules |
| NumPy action delivery | [`numpy_policy_actions.py`](../code/mlx_backend/numpy_policy_actions.py) |
| Independent proofs | [Original-API tests](../tests/test_mlx_brian_device.py), [custom arithmetic](../tests/test_mlx_custom_cython_state.py), [recording](../tests/test_mlx_sparse_recording.py), corresponding `tests/test_mlx_*.py`, bundled fixtures and the tiny command above |

Full matrices, case intervals, commands, failed attempts and source-bound receipts
remain in the private evidence pack. These are review groups; they do not
require rewriting existing history.
