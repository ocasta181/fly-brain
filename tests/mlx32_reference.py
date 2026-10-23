"""Independent Brian CPU observations for the fixed Eon float32 slice."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReferenceModel:
    dt_s: float = 0.0001
    v_rest_v: float = -0.052
    v_reset_v: float = -0.052
    threshold_v: float = -0.045
    tau_m_s: float = 0.020
    tau_g_s: float = 0.005
    refractory_steps: int = 22
    delay_steps: int = 18
    weight_v: float = 0.000275
    input_v: float = 0.06875


@dataclass(frozen=True)
class ReferenceResult:
    v: np.ndarray
    g: np.ndarray
    counts: np.ndarray
    events: np.ndarray  # (trial, neuron, tick), in tick/neuron order
    v_trace: np.ndarray  # Post-reset observations, indexed (tick, neuron).
    g_trace: np.ndarray
    eligible_trace: np.ndarray
    spike_phase: tuple[str, int] = ("thresholds", 1)
    state_phase: tuple[str, int] = ("end", 0)

    def raw_bits(self) -> dict[str, list[int]]:
        unsigned = np.uint32 if self.v.dtype == np.float32 else np.uint64
        return {
            "v": self.v.view(unsigned).tolist(),
            "g": self.g.view(unsigned).tolist(),
        }


def run_reference(
    n_neurons: int,
    edges: tuple[tuple[int, int, int], ...],
    targets: tuple[int, ...],
    input_tape: np.ndarray,
    *,
    model: ReferenceModel = ReferenceModel(),
    dtype: type = np.float32,
    silenced: tuple[int, ...] = (),
    trial: int = 0,
) -> ReferenceResult:
    """Run Brian equations, not native coefficients or a Python recurrence.

    Each occurrence uses an actual PoissonInput with probability one and a
    binary TimedArray weight. This replaces only its random draw, preserving
    its conditional writes and its synapses-before-resets delivery phase.
    """
    import brian2 as brian
    from brian2.devices.device import reset_device

    tape = np.asarray(input_tape)
    if tape.ndim != 2 or tape.shape[1] != len(targets):
        raise ValueError("input tape must have one column per channel occurrence")
    if tape.dtype not in (np.dtype(np.uint8), np.dtype(bool)):
        raise ValueError("input tape must contain binary uint8 or bool values")
    if np.any((tape != 0) & (tape != 1)):
        raise ValueError("input tape must be binary")
    scalar_dtype = np.dtype(dtype).type
    if scalar_dtype not in (np.float32, np.float64):
        raise ValueError("Brian controls require float32 or float64")
    previous_target = brian.prefs.codegen.target
    previous_dtype = brian.prefs.core.default_float_dtype
    previous_refractory = brian.prefs.legacy.refractory_timing
    try:
        brian.set_device("runtime")
        brian.prefs.codegen.target = "numpy"
        brian.prefs.core.default_float_dtype = scalar_dtype
        brian.prefs.legacy.refractory_timing = False
        clock = brian.Clock(dt=model.dt_s * brian.second)
        namespace = {
            "v_0": model.v_rest_v * brian.volt,
            "v_rst": model.v_reset_v * brian.volt,
            "v_th": model.threshold_v * brian.volt,
            "t_mbr": model.tau_m_s * brian.second,
            "tau": model.tau_g_s * brian.second,
            "input_v": model.input_v * brian.volt,
        }
        neurons = brian.NeuronGroup(
            n_neurons,
            """
            dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
            dg/dt = -g / tau : volt (unless refractory)
            rfc : second
            """,
            threshold="v > v_th",
            reset="v = v_rst; w = 0; g = 0 * mV",
            refractory="rfc",
            method="linear",
            clock=clock,
            dtype=scalar_dtype,
            namespace=namespace,
            name="mlx32_reference_neurons*",
        )
        neurons.v = namespace["v_0"]
        neurons.g = 0 * brian.volt
        neurons.rfc = model.refractory_steps * clock.dt
        synapses = brian.Synapses(
            neurons,
            neurons,
            "w : volt",
            on_pre="g += w",
            delay=model.delay_steps * clock.dt,
            dtype=scalar_dtype,
            name="mlx32_reference_synapses*",
        )
        if edges:
            pre, post, contacts = map(np.asarray, zip(*edges))
            synapses.connect(i=pre, j=post)
            weights = contacts * model.weight_v
            weights[np.isin(pre, silenced)] = 0
            synapses.w = weights * brian.volt
        else:
            synapses.active = False
        inputs = []
        for channel, target in enumerate(targets):
            tape_name = f"fixed_channel_{channel}"
            values = tape[:, channel] if len(tape) else np.zeros(1)
            neurons.namespace[tape_name] = brian.TimedArray(values, dt=clock.dt)
            inputs.append(
                brian.PoissonInput(
                    neurons[target : target + 1],
                    target_var="v",
                    N=1,
                    rate=1 / clock.dt,
                    weight=f"input_v * {tape_name}(t)",
                    when="synapses",
                    order=channel,
                )
            )
            neurons.rfc[target] = 0 * brian.second
        spikes = brian.SpikeMonitor(neurons, when="thresholds", order=1)
        states = brian.StateMonitor(
            neurons, ("v", "g", "not_refractory"), record=True,
            when="end", order=0,
        )
        network = brian.Network(neurons, synapses, spikes, states, *inputs)
        network.schedule = ["start", "groups", "thresholds", "synapses", "resets", "end"]
        network.run(len(tape) * clock.dt, namespace={})
        ticks = np.rint(np.asarray(spikes.t / clock.dt)).astype(np.int64)
        neuron_indices = np.asarray(spikes.i, dtype=np.int64)
        order = np.lexsort((neuron_indices, ticks))
        events = np.column_stack(
            (np.full(len(ticks), trial, dtype=np.int64), neuron_indices, ticks)
        )[order]
        return ReferenceResult(
            v=np.asarray(neurons.v / brian.volt, dtype=scalar_dtype).copy(),
            g=np.asarray(neurons.g / brian.volt, dtype=scalar_dtype).copy(),
            counts=np.asarray(spikes.count, dtype=np.int64).copy(),
            events=events,
            v_trace=np.asarray(states.v / brian.volt, dtype=scalar_dtype).T.copy(),
            g_trace=np.asarray(states.g / brian.volt, dtype=scalar_dtype).T.copy(),
            eligible_trace=np.asarray(states.not_refractory, dtype=bool).T.copy(),
        )
    finally:
        brian.prefs.codegen.target = previous_target
        brian.prefs.core.default_float_dtype = previous_dtype
        brian.prefs.legacy.refractory_timing = previous_refractory
        reset_device()
