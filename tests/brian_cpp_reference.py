"""Independent, actual Brian 2.8.0 C++ observations of the fixed Eon model."""

from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np


@dataclass(frozen=True)
class ReferenceModel:
    dt_s: float = 0.0001
    v_rest_v: float = -52 * 0.001
    v_reset_v: float = -52 * 0.001
    threshold_v: float = -0.045
    tau_m_s: float = 0.02
    tau_g_s: float = 0.005
    refractory_steps: int = 22
    delay_steps: int = 18
    weight_v: float = 0.000275
    input_v: float = 0.06875


@dataclass(frozen=True)
class ReferenceResult:
    v: np.ndarray
    g: np.ndarray
    lastspike: np.ndarray
    not_refractory: np.ndarray
    counts: np.ndarray
    delay_ring: np.ndarray
    spikes: np.ndarray
    v_trace: np.ndarray
    g_trace: np.ndarray
    eligible_trace: np.ndarray
    receipt_path: Path


def _save_receipt(root: Path, receipt: dict[str, object]) -> None:
    sources = sorted(path for path in (root / "build").rglob("*") if path.is_file()
                     and (path.suffix in {".cpp", ".h"} or path.name == "makefile"))
    receipt["generated_sources"] = {
        str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
        for path in sources
    }
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


def run_reference(
    n_neurons: int,
    edges: Sequence[tuple[int, int, int]],
    targets: Sequence[int],
    input_tape: np.ndarray,
    *,
    output_directory: Path,
    model: ReferenceModel = ReferenceModel(),
    silenced: Sequence[int] = (),
) -> ReferenceResult:
    """Compile one disconnected C++ network for all trial tapes.

    SpikeGeneratorGroup replaces only the input draws. Actual Brian equations,
    conditional synaptic writes, event queues, refractory predicates, threshold,
    reset, and end-of-tick monitors produce every observed physical state.
    No production code, prepared coefficient, or host recurrence is used.
    """
    import brian2 as brian
    from brian2.codegen.cpp_prefs import get_compiler_and_args
    from brian2.devices.device import reset_device

    if brian.__version__ != "2.8.0":
        raise RuntimeError(f"The CPU oracle requires Brian2 2.8.0, found {brian.__version__}")
    tape = np.asarray(input_tape)
    if tape.ndim != 3 or tape.shape[2] != len(targets):
        raise ValueError("reference tape must have shape (steps, trials, channels)")
    if tape.dtype not in (np.dtype(np.uint8), np.dtype(bool)) or np.any(tape > 1):
        raise ValueError("reference tape must be binary uint8 or bool")
    steps, batch, channels = tape.shape
    if n_neurons <= 0 or steps <= 0 or batch <= 0:
        raise ValueError("the compiled oracle requires positive dimensions")
    root = Path(output_directory).resolve()
    if root.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("the CPU build directory must be outside the checkout")
    root.mkdir(parents=True, exist_ok=False)
    np.savez(root / "inputs.npz", edges=np.asarray(edges, dtype=np.int64).reshape(-1, 3),
             targets=np.asarray(targets, dtype=np.int64), input_tape=tape,
             silenced=np.asarray(silenced, dtype=np.int64))
    settings = {
        "core.default_float_dtype": np.float64,
        "legacy.refractory_timing": False,
        "devices.cpp_standalone.extra_make_args_unix": ["-j2"],
        "devices.cpp_standalone.openmp_threads": 0,
    }
    previous_prefs = {key: brian.prefs[key] for key in settings}
    thread_names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
    previous_env = {key: os.environ.get(key) for key in thread_names}
    started = perf_counter()
    receipt = {
        "backend": "brian2_cpp_standalone", "brian2": brian.__version__,
        "model": asdict(model), "shape": [steps, batch, n_neurons],
        "oracle_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs_sha256": sha256((root / "inputs.npz").read_bytes()).hexdigest(),
        "platform": platform.platform(), "machine": platform.machine(),
        "spike_phase": ["thresholds", 1], "state_phase": ["end", 0],
        "build_directory": str(root / "build"), "status": "setup",
    }
    try:
        for key, value in settings.items():
            brian.prefs[key] = value
        for key in thread_names:
            os.environ[key] = "1"
        brian.set_device("cpp_standalone", build_on_run=False)
        brian.device.reinit()
        brian.device.activate(build_on_run=False)
        clock = brian.Clock(dt=model.dt_s * brian.second, name="fixed_cpu_clock*")
        namespace = {
            "v_0": model.v_rest_v * brian.volt,
            "v_rst": model.v_reset_v * brian.volt,
            "v_th": model.threshold_v * brian.volt,
            "t_mbr": model.tau_m_s * brian.second,
            "tau": model.tau_g_s * brian.second,
            "input_v": model.input_v * brian.volt,
        }
        neurons = brian.NeuronGroup(
            batch * n_neurons,
            """dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
            dg/dt = -g / tau : volt (unless refractory)
            rfc : second""",
            threshold="v > v_th", reset="v = v_rst; w = 0; g = 0 * mV",
            refractory="rfc", method="linear", clock=clock, namespace=namespace,
            dtype={key: np.float64 for key in ("v", "g", "rfc", "lastspike")},
            name="fixed_cpu_neurons*",
        )
        neurons.v = namespace["v_0"]
        neurons.g = 0 * brian.volt
        neurons.rfc = model.refractory_steps * clock.dt
        neurons.lastspike = -1e4 * brian.second
        neurons.not_refractory = True
        synapses = brian.Synapses(
            neurons, neurons, "w : volt", on_pre="g += w",
            delay=model.delay_steps * clock.dt, dtype=np.float64,
            name="fixed_cpu_synapses*",
        )
        ordered_edges = sorted(enumerate(edges), key=lambda item: (item[1][0], item[0]))
        if ordered_edges:
            pre, post, weights = [], [], []
            for position in range(batch):
                for _, (source, target, contacts) in ordered_edges:
                    pre.append(position * n_neurons + source)
                    post.append(position * n_neurons + target)
                    weights.append(0.0 if source in silenced else contacts * model.weight_v)
            synapses.connect(i=pre, j=post)
            synapses.w = np.asarray(weights) * brian.volt
        else:
            synapses.active = False
        objects = [neurons, synapses]
        if channels:
            ticks, positions, occurrences = np.nonzero(tape)
            input_indices = positions * channels + occurrences
            inputs = brian.SpikeGeneratorGroup(
                batch * channels, input_indices, ticks * clock.dt,
                clock=clock, when="thresholds", order=-1, name="fixed_cpu_inputs*",
            )
            input_synapses = brian.Synapses(
                inputs, neurons, on_pre="v_post += input_v", namespace=namespace,
                name="fixed_cpu_input_edges*",
            )
            destination = [position * n_neurons + target
                           for position in range(batch) for target in targets]
            input_synapses.connect(i=np.arange(batch * channels), j=destination)
            neurons.rfc[destination] = 0 * brian.second
            objects.extend((inputs, input_synapses))
        spikes = brian.SpikeMonitor(neurons, when="thresholds", order=1)
        states = brian.StateMonitor(
            neurons, ("v", "g", "lastspike", "not_refractory"),
            record=True, when="end", order=0,
        )
        network = brian.Network(*objects, spikes, states)
        network.schedule = ["start", "groups", "thresholds", "synapses", "resets", "end"]
        network.run(steps * clock.dt, namespace={})
        compiler, flags = get_compiler_and_args()
        receipt.update(compiler=compiler, compiler_flags=flags,
                       schedule=str(network.scheduling_summary()),
                       environment={key: os.environ.get(key) for key in
                                    (*thread_names, "CC", "CXX", "CFLAGS", "CPPFLAGS", "LDFLAGS")})
        (root / "brian-preferences.txt").write_text(brian.prefs.as_file)
        receipt["status"] = "building"
        _save_receipt(root, receipt)
        brian.device.build(directory=str(root / "build"), run=False, with_output=False)
        from brian2.utils.logger import std_silent

        build_output = Path(std_silent.dest_fname_stdout).read_text()
        (root / "build.log").write_text(build_output)
        (root / "build.stderr.log").write_text(Path(std_silent.dest_fname_stderr).read_text())
        commands = [line for line in build_output.splitlines() if " -c " in line]
        if not commands:
            raise RuntimeError("CPU oracle did not retain its actual compiler command")
        command = shlex.split(commands[0])[:1]
        version = subprocess.run(command + ["--version"], check=True,
                                 capture_output=True, text=True, timeout=5)
        receipt.update(compiler_command=command, compiler_version=version.stdout,
                       compiler_path=shutil.which(command[0]), build_commands=commands,
                       executable_sha256=sha256((root / "build/main").read_bytes()).hexdigest())
        receipt["status"] = "running"
        _save_receipt(root, receipt)
        brian.device.run(directory=str(root / "build"), with_output=False)
        flat = np.asarray(spikes.i, dtype=np.int64)
        ticks = np.rint(np.asarray(spikes.t / clock.dt)).astype(np.int64)
        events = np.column_stack((ticks, flat // n_neurons, flat % n_neurons))
        events = events[np.lexsort((events[:, 2], events[:, 1], events[:, 0]))]

        def snapshot(name: str, unit: object) -> np.ndarray:
            return np.asarray(getattr(neurons, name) / unit,
                              dtype=np.float64).reshape(batch, n_neurons).copy()

        def trace(name: str, unit: object) -> np.ndarray:
            return np.asarray(getattr(states, name) / unit, dtype=np.float64).T.reshape(
                steps, batch, n_neurons,
            ).copy()

        ring = np.zeros((model.delay_steps, batch, n_neurons), dtype=np.uint8)
        for tick, position, neuron in events:
            if tick >= max(0, steps - model.delay_steps):
                ring[tick % model.delay_steps, position, neuron] = 1
        result = ReferenceResult(
            v=snapshot("v", brian.volt), g=snapshot("g", brian.volt),
            lastspike=snapshot("lastspike", brian.second),
            not_refractory=np.asarray(neurons.not_refractory, dtype=bool).reshape(
                batch, n_neurons).copy(),
            counts=np.asarray(spikes.count, dtype=np.int64).reshape(batch, n_neurons).copy(),
            delay_ring=ring, spikes=events, v_trace=trace("v", brian.volt),
            g_trace=trace("g", brian.volt),
            eligible_trace=np.asarray(states.not_refractory, dtype=bool).T.reshape(
                steps, batch, n_neurons).copy(), receipt_path=root / "receipt.json",
        )
        np.savez(root / "observations.npz", **{
            name: value for name, value in vars(result).items() if isinstance(value, np.ndarray)
        }, lastspike_trace=trace("lastspike", brian.second))
        receipt.update(status="complete", total_s=perf_counter() - started,
                       observations_sha256=sha256((root / "observations.npz").read_bytes()).hexdigest())
        _save_receipt(root, receipt)
        return result
    except BaseException as error:
        receipt.update(status="failed", error=repr(error), total_s=perf_counter() - started)
        _save_receipt(root, receipt)
        raise
    finally:
        reset_device()
        for key, value in previous_prefs.items():
            brian.prefs[key] = value
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
