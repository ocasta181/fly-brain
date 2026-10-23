"""Apple MLX adapter for the validated GPL-2.0-or-later runtime."""

import hashlib
import importlib.metadata
import json
import math
import os
import platform
from numbers import Real
from operator import index
from pathlib import Path

MAX_EAGER_EVENT_BYTES = 64 * 1024 * 1024


def configure():
    """Check the supported platform before importing the optional Metal runtime."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("MLX requires native Apple silicon macOS")
    if os.environ.setdefault("MLX_ENABLE_TF32", "0") != "0":
        raise ValueError("MLX requires MLX_ENABLE_TF32=0")
    for name, version in (
        ("mlx", "0.32.3"),
        ("mlx-metal", "0.32.3"),
        ("numpy", "1.26.4"),
        ("pyarrow", "25.0.1"),
    ):
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f"MLX requires {name}=={version}")


def dimensions(duration, trials, *, model=None):
    if isinstance(duration, bool) or not isinstance(duration, Real):
        raise TypeError("Duration must be a finite nonnegative number")
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("Duration must be finite and nonnegative")
    if isinstance(trials, bool) or getattr(getattr(trials, "dtype", None), "kind", None) == "b":
        raise TypeError("Trial count must be a positive integer")
    trials = index(trials)
    if trials < 1:
        raise ValueError("Trial count must be a positive integer")
    if model is None:
        from mlx_backend.model import resolve_model

        model = resolve_model(dt_ms=0.1)
    return model.computed_steps(duration)


def input_pin():
    from mlx_backend.models import InputPin

    return InputPin(
        completeness_sha256="52b0ac6094cd32c546f8d4c341e094376f48f4e791f8db9b166de5dff8199ea4",
        connectivity_sha256="efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347",
        neurons=138639,
        edges=15091983,
    )


def load_inputs(completeness, connectivity):
    from mlx_backend.inputs import load_connectome

    return load_connectome(Path(completeness), Path(connectivity))


def experiment_spec(experiment):
    """Preserve preset streams while accepting custom activation and silencing."""
    from mlx_backend.experiments import EXPERIMENTS
    from mlx_backend.models import Experiment

    key = experiment.get("key", "custom")
    if not isinstance(key, str) or not key:
        raise ValueError("Experiment key must be a nonempty string")
    first = tuple(experiment.get("neu_exc", ()))
    second = tuple(experiment.get("neu_exc2", ()))
    rates = (float(experiment.get("stim_rate", 0)),) * len(first) + (
        float(experiment.get("stim_rate2", 0)),
    ) * len(second)
    if any(not math.isfinite(rate) or rate < 0 for rate in rates):
        raise ValueError("Stimulus rates must be finite and nonnegative")
    original = EXPERIMENTS.get(key)
    if original is not None:
        code, generator = original.code, original.generator_code
    else:
        code = generator = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")
    candidate = Experiment(
        key,
        code,
        generator,
        first + second,
        rates,
        tuple(experiment.get("neu_slnc", ())),
        len(first),
    )
    if original is not None and (
        candidate.activated_ids == original.activated_ids
        and candidate.rates_hz == original.rates_hz
        and candidate.silenced_ids == original.silenced_ids
        and candidate.activation_classes == original.activation_classes
    ):
        return original
    return candidate


def read_replay(directory, spec, targets, steps, trials, *, pin=None, dt_ms=0.1):
    """Validate complete saved bytes before selecting a horizon/trial prefix."""
    import numpy as np

    directory = Path(directory)
    metadata = json.loads((directory / "stimulus.json").read_text())
    artifact = directory / "stimulus.npz"
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != metadata["artifact_sha256"]:
        raise ValueError("Saved MLX stimulus archive hash differs")
    pin = input_pin() if pin is None else pin
    expected = {
        "format_version": 1,
        "layout": ["trial", "step", "channel"],
        "dt_ms": dt_ms,
        "generator": "PCG64",
        "numpy": "1.26.4",
        "dtype": "|u1",
        "targets": list(targets),
        "activated_ids": list(spec.activated_ids),
        "activation_classes": list(spec.activation_classes),
        "rates_hz": list(spec.rates_hz),
        "amplitudes_mv": [68.75] * len(targets),
        "experiment": spec.name,
        "experiment_code": spec.code,
        "generator_code": spec.generator_code,
        "silenced_ids": list(spec.silenced_ids),
        "input_sha256": {
            "completeness": pin.completeness_sha256,
            "connectivity": pin.connectivity_sha256,
        },
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Saved MLX stimulus setup differs from the pinned experiment")
    with np.load(artifact, allow_pickle=False) as saved:
        if saved.files != ["events"]:
            raise ValueError(
                "Saved MLX stimulus must contain only canonical event bits"
            )
        events = saved["events"]
    trial_ids = metadata["trial_indices"]
    seed = metadata["seed"]
    if (
        events.dtype != np.uint8
        or events.ndim != 3
        or list(events.shape) != metadata["shape"]
        or events.shape[2] != len(targets)
        or steps > events.shape[1]
        or trials > events.shape[0]
        or events.shape[0] != len(trial_ids)
        or type(seed) is not int
        or seed < 0
        or any(
            type(index) is not int or not 0 <= index <= 2**63 - 1 for index in trial_ids
        )
        or len(set(trial_ids)) != len(trial_ids)
        or np.any(events > 1)
        or hashlib.sha256(events.tobytes()).hexdigest()
        != metadata["canonical_event_sha256"]
        or [hashlib.sha256(row.tobytes()).hexdigest() for row in events]
        != metadata["per_trial_event_sha256"]
        or [[seed, spec.generator_code, index] for index in trial_ids]
        != metadata["seed_tuples"]
    ):
        raise ValueError(
            "Saved MLX stimulus shape, identity, horizon or event bytes differ"
        )
    selected = events[:trials, :steps]
    selected.setflags(write=False)
    return selected, tuple(trial_ids[:trials]), seed


def stimuli(connectome, spec, steps, trials, seed, replay=None, *, stream=None, dt_ms=0.1):
    import numpy as np
    from mlx_backend.models import Stimulus, TrialId
    from mlx_backend.stimuli import SeededEvents, neuron_indices, trial_events, uniforms

    if type(seed) is not int or seed < 0:
        raise ValueError("MLX seed must be a nonnegative integer")
    targets = neuron_indices(connectome, spec.activated_ids)
    if replay is not None:
        events, trial_ids, seed = read_replay(
            replay, spec, targets, steps, trials, pin=connectome.pin, dt_ms=dt_ms
        )
    else:
        events = None
        trial_ids = tuple(range(trials))
    if stream is None:
        stream = events is None and trials * steps * len(targets) > MAX_EAGER_EVENT_BYTES
    result = []
    for row, trial in enumerate(trial_ids):
        if stream and events is None:
            native = SeededEvents(steps, spec.rates_hz, (trial,), seed, spec.generator_code, dt_ms)
            digest = None
        else:
            bits = (
                events[row]
                if events is not None
                else trial_events(
                    spec.rates_hz,
                    uniforms((seed, spec.generator_code, trial), (steps, len(targets))),
                    dt_ms=dt_ms,
                )
            )
            native = np.asarray(bits[None], dtype=np.uint8)
            # Hashing already materializes these exact bytes. Retain that
            # immutable owner so warm calls need no defensive alias snapshot.
            payload = native.tobytes()
            native = np.frombuffer(payload, np.uint8).reshape(native.shape)
            digest = hashlib.sha256(payload).hexdigest()
        result.append(
            Stimulus(
                native,
                targets,
                spec.rates_hz,
                (TrialId(trial),),
                seed,
                spec.generator_code,
                digest,
                dt_ms=dt_ms,
            )
        )
    return result, neuron_indices(connectome, spec.silenced_ids)


def combine_trials(inputs):
    """Combine independently generated trials without changing their identities."""
    import numpy as np
    from mlx_backend.models import Stimulus
    from mlx_backend.stimuli import CombinedEvents

    first = inputs[0]
    if any(
        (x.targets, x.rates_hz, x.seed, x.generator_code, x.events.shape[1:], x.dt_ms, x.amplitudes_mv)
        != (
            first.targets,
            first.rates_hz,
            first.seed,
            first.generator_code,
            first.events.shape[1:],
            first.dt_ms,
            first.amplitudes_mv,
        )
        for x in inputs
    ):
        raise ValueError(
            "Batched trials require identical model and input configuration"
        )
    if (
        any(hasattr(x.events, "chunk") for x in inputs)
        or sum(x.events.size for x in inputs) > MAX_EAGER_EVENT_BYTES
    ):
        events = CombinedEvents(tuple(x.events for x in inputs))
        digest = None
    else:
        events = np.concatenate([x.events for x in inputs], axis=0)
        payload = events.tobytes()
        events = np.frombuffer(payload, np.uint8).reshape(events.shape)
        digest = hashlib.sha256(payload).hexdigest()
    return Stimulus(
        events,
        first.targets,
        first.rates_hz,
        tuple(i for x in inputs for i in x.trial_indices),
        first.seed,
        first.generator_code,
        digest,
        dt_ms=first.dt_ms,
        amplitudes_mv=first.amplitudes_mv,
    )


def execute(connectome, stimulus, silenced, *, engine="window", record=True, spike_sink=None,
            model=None):
    from mlx_backend.runner import run

    return run(
        connectome, stimulus, silenced, precision="0", engine=engine, record=record,
        spike_sink=spike_sink, model=model,
    )
