from dataclasses import replace
from functools import lru_cache
from hashlib import sha256

from .models import (
    Connectome,
    SimulationRun,
    Stimulus,
)


@lru_cache(maxsize=32)
def _bind_default_target(model, target, compiler, flags, optimise):
    """Freeze the default model's actual Brian arithmetic before native use."""
    identity = repr((model.signature, target, compiler, flags, optimise))
    return replace(model, reference_codegen_target=target,
                   reference_cpp_compiler=compiler, reference_compile_args=flags,
                   reference_optimise=optimise,
                   signature=(sha256(identity.encode()).hexdigest()
                              if model.cacheable else None),
                   default_specialization=False, state_specialization=False)


def _exact_default_model(model):
    if not model.state_specialization:
        return model
    import brian2 as b
    from brian2.codegen.generators import CPPCodeGenerator, CythonCodeGenerator, NumpyCodeGenerator
    from brian2.codegen.cpp_prefs import get_compiler_and_args
    from brian2.devices.device import get_device

    generator = get_device().code_object_class().generator_class
    target = next(name for base, name in ((NumpyCodeGenerator, "numpy"),
                      (CythonCodeGenerator, "cython"), (CPPCodeGenerator, "cpp"))
                  if issubclass(generator, base))
    compiler, flags = get_compiler_and_args() if target in ("cython", "cpp") else (None, ())
    bind = _bind_default_target if model.cacheable else _bind_default_target.__wrapped__
    return bind(model, target, compiler, tuple(flags),
                bool(b.prefs.codegen.loop_invariant_optimisations))


def run(
    connectome: Connectome,
    stimulus: Stimulus,
    silenced: tuple[int, ...],
    *,
    precision: str,
    engine: str = "window",
    record: bool = True,
    spike_sink=None,
    model=None,
) -> SimulationRun:
    if spike_sink is not None and not record:
        raise ValueError("A spike sink requires recording to be enabled")
    if engine not in ("window", "configured"):
        raise ValueError(f"Unknown MLX engine: {engine}")
    if precision != "0":
        raise ValueError("Configured execution requires the pinned precision configuration")
    from .configured import run as run_configured
    from .model import resolve_model

    model = resolve_model(dt_ms=stimulus.dt_ms) if model is None else model
    if model.dt_ms != stimulus.dt_ms:
        raise ValueError("Stimulus timestep differs from the resolved model")
    # The legacy selector remains accepted by callers, but its binary32
    # resting-offset state can shift a float64 Brian threshold crossing.
    # Execute every production step with the existing exact native compiler.
    return run_configured(connectome, stimulus, silenced,
                          model=_exact_default_model(model),
                          record=record, spike_sink=spike_sink)
