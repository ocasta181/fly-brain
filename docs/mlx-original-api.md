# Original Brian API on MLX

The opt-in MLX device executes Eon's original model module with real Brian
`NeuronGroup`, `Synapses`, `PoissonInput`, `SpikeMonitor` and `Network` objects.
Brian handles units and object construction; MLX/Metal executes the simulation.

The current source implements Brian's built-in equation functions with
float32/float64, integer and Boolean persistent state. Dynamic complex
intermediates use paired native real and imaginary components, including
state-, index- and time-dependent expressions and destination conversions.
Generic user Python callbacks remain outside Eon's contract and are rejected.

For actual NumPy code objects, native arithmetic and casts produce floating
status for `ignore`, `warn`, `raise`, `call`, `log` and `print`. The host delivers
warnings, exceptions and configured actions at the ordered expression boundary.
Ambient NumPy settings do not restrict actual Cython or C++ code objects.

The [maintainer review note](mlx-review.md) records numerical comparison rules,
accepted IEEE parameters and signed-clock behavior, named boundary checks and
source applicability. These qualifications apply to the identified cases.

Select the device before constructing objects. With `original` referring to
Eon's model module, the context manager also installs the device in loky workers:

```python
from mlx_backend.brian_device import original_api

with original_api():
    original.run_exp(
        "experiment", excited_ids, result_directory,
        completeness_csv, connectivity_parquet,
        params=parameters, n_proc=2,
    )
```

For serial construction, `brian2.set_device(MLXDevice())` also works. Original
model functions retain their arguments and return types. Successive
`Network.run` calls preserve pending delayed events and import state and weight
edits. Requested time and rounded clock time retain Brian's distinct meanings.
Monitor and input active flags are respected. `SpikeMonitor(record=False)`
retains counts with bounded GPU storage and no host spike records.

Poisson inputs preserve Brian's channel order, eligibility and RNG stream.
Positive-delay windows combine target-neuron input sampling with neuron updates
where possible; wider layouts use a separate native prepass. Uniform tapes use
bounded chunks, committing only consumed draws. Zero delay uses a native probe.

CPU target, compiler flags, stored dtypes and preparation temporary dtypes are
captured separately to preserve their arithmetic behavior. Native preparation
keeps actual Cython objects and source while skipping unused CPU timestep
compilation. The raw-RNG sampler compiles normally.

`native_runs`, `native_steps` and `last_timings` provide execution diagnostics.
Independent CPU comparisons are in `tests/test_mlx_brian_device.py`; their
reference networks use Eon's unchanged model module. Unsupported Brian object
graphs raise explicit errors before native execution.

Source and independent-test entrypoints are in the
[maintainer review map](mlx-review.md#review-in-coherent-groups).
