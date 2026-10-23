"""Synchronize actual NumPy error callbacks at native operation boundaries.

Only host policy delivery crosses this boundary. Numerical expressions, their
deferred NumPy locals, the random stream, and subsequent timesteps stay native.
The context is per Network.run, never retained by a compiled-program cache.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace

import mlx.core as mx
import numpy as np


_current = ContextVar('eon_numpy_policy_context', default=None)


@dataclass
class _Stage:
    state: object
    name: str
    spikes: object
    changed_spikes: bool = False


@contextmanager
def stage(state, name):
    context = _current.get()
    boundary = _Stage(state, name, state.spikes)
    if context is None:
        yield boundary
        return
    previous = context.stage
    context.stage = boundary
    try:
        yield boundary
    finally:
        context.stage = previous


def threshold_completed(state):
    context = _current.get()
    if context is not None:
        context.threshold_completed(state)


def refreshed(prepared):
    context = _current.get()
    return prepared if context is None else context.refreshed(prepared)


def stopped():
    context = _current.get()
    return context is not None and (context.network._stopped or
                                   type(context.network)._globally_stopped)


def deliver(prepared, name, status, values, random, host_views):
    """Return the updated native RNG and the original policy exception."""
    from .numpy_policy_actions import deliver as action
    context = _current.get()
    categories = ((1, 'divide'), (2, 'over'), (4, 'under'), (8, 'invalid'))
    policy = np.geterr()
    callback = any(status & bit and policy[kind] in ('call', 'log')
                   for bit, kind in categories)
    if context is None or not callback:
        try:
            action(name, status)
        except Exception as error:
            return random, error
        return random, None
    return context.host_action(prepared, values, random, host_views, lambda: action(name, status))


def host_action(prepared, values, random, host_views, action):
    """Share ordered host policy delivery with a native cast-warning marker."""
    context = _current.get()
    if context is not None:
        return context.host_action(prepared, values, random, host_views, action)
    try:
        action()
    except Exception as error:
        return random, error
    return random, None


def failed_phase_state(prepared, state, synapse_spikes, synapses_completed):
    """Retain Brian's advanced synaptic queue when its clock did not tick."""
    from .brian_errors import _ReplayModel, _enqueue, _rescale_words, _shift_queue_back
    model = prepared.model
    lag = model.execution_phases.index('synapses') < model.execution_phases.index('thresholds')
    if not synapses_completed and not lag:
        return state
    window = prepared.window
    part = replace(prepared, window=1, model=_ReplayModel(model))
    queue = replace(state, history=_rescale_words(state.history, window, 1),
                    due=_rescale_words(state.due, window, 1))
    if synapses_completed:
        queue = _enqueue(part, queue, synapse_spikes, state.step)
        queue = replace(queue,
            history={key:word for key,word in queue.history.items()
                     if key >= state.step-model.arrival_delay_steps+1},
            due={key:word for key,word in queue.due.items() if key > state.step})
        queue = replace(queue, history=_shift_queue_back(queue.history, 1),
                        due=_shift_queue_back(queue.due, 1))
    if lag:
        queue = _enqueue(part, queue, state.spikes, state.step)
    return replace(state, history=_rescale_words(queue.history, 1, window),
                   due=_rescale_words(queue.due, 1, window))


class PolicyContext:
    def __init__(self, network, group, synapses, monitors, clock, device):
        self.network, self.group, self.synapses = network, group, synapses
        self.monitors, self.clock, self.device = tuple(monitors), clock, device
        self.stage = None
        self.pending = []
        self.weights = None
        self.callback_count = 0

    def __enter__(self):
        self.token = _current.set(self)
        return self

    def __exit__(self, *error):
        _current.reset(self.token)

    def refreshed(self, prepared):
        return prepared if self.weights is None else replace(prepared, weights=self.weights)

    def threshold_completed(self, state):
        observers = tuple((monitor, bool(monitor.record))
                          for monitor in self.monitors if monitor.active)
        if observers:
            self.pending.append((state.step, state.spikes, observers))

    def flush_monitors(self):
        """Commit exactly the monitors scheduled after completed thresholds."""
        if not self.pending:
            return
        updates = {}
        for step, bits, observers in self.pending:
            rows = np.flatnonzero(np.asarray(bits)).astype(np.int32)
            for monitor, recording in observers:
                rows_list, steps_list, counts = updates.setdefault(
                    id(monitor), ([], [], np.zeros(len(self.group), np.int64)))
                counts += np.bincount(rows, minlength=len(self.group))
                if recording:
                    rows_list.append(rows)
                    steps_list.append(np.full(rows.size, step, np.int64))
        for monitor in self.monitors:
            if id(monitor) not in updates:
                continue
            rows_list, steps_list, counts = updates[id(monitor)]
            old_size = int(monitor.variables['N'].get_value()[0])
            total = int(counts.sum())
            if rows_list:
                rows = np.concatenate(rows_list)
                steps = np.concatenate(steps_list)
                # N counts events even when recording is disabled. Brian's
                # record flag is fixed on construction, so recorded size=N.
                monitor.resize(old_size + rows.size)
                monitor.variables['i'].get_value()[old_size:] = rows
                np.multiply(steps, float(self.clock.dt),
                            out=monitor.variables['t'].get_value()[old_size:])
            monitor.variables['count'].get_value()[:] += counts
            monitor.variables['N'].get_value()[0] = np.asarray(old_size + total).astype(np.int32)
        self.pending.clear()

    @staticmethod
    def _decode(raw, variable, shadow=False):
        raw = np.asarray(raw)
        dtype = np.dtype(variable.dtype)
        if shadow:
            if dtype.kind == 'f':
                raw = raw.view(np.float64)
            elif dtype.kind == 'i':
                raw = raw.view(np.int64)
            elif dtype.kind == 'b':
                return raw != 0
            with np.errstate(all='ignore'):
                return raw.astype(dtype)
        if dtype.kind == 'f':
            return raw.view(dtype)
        if dtype.kind == 'b' and variable.storage_dtype == 'uint8':
            return raw.view(np.bool_)
        return raw

    @staticmethod
    def _encode(host, variable, shadow=False):
        host = np.asarray(host)
        if shadow:
            if host.dtype.kind == 'f':
                with np.errstate(all='ignore'):
                    return mx.array(host.astype(np.float64).view(np.uint64))
            return mx.array(host.astype(np.uint64))
        if host.dtype.kind == 'f':
            host = host.view(np.uint32 if host.dtype.itemsize == 4 else np.uint64)
        elif variable.dtype == 'bool' and variable.storage_dtype == 'uint8':
            host = host.view(np.uint8)
        return mx.array(host)

    def host_action(self, prepared, values, random, host_views, action):
        from .expression_poisson_input import export_random, import_random
        boundary = self.stage
        if boundary is None:
            raise RuntimeError('A NumPy callback requires its native stage boundary')
        views = dict(host_views)
        for key, variable in prepared.program.variables.items():
            if not variable.persistent:
                continue
            source = views.get(key, key)
            host = self._decode(values[source], variable, shadow=source != key)
            self.group.variables[key].get_value()[:] = host
        visible = (np.empty(0, np.int32) if boundary.spikes is None else
                   np.flatnonzero(np.asarray(boundary.spikes)).astype(np.int32))
        eventspace = self.group.variables['_spikespace'].get_value()
        eventspace[:visible.size] = visible
        eventspace[-1] = visible.size
        step = boundary.state.step
        self.clock.variables['timestep'].get_value()[0] = step
        self.clock.variables['t'].get_value()[0] = step * prepared.model.dt_s
        self.network.t_ = step * prepared.model.dt_s
        self.flush_monitors()
        export_random(random, self.device)
        before_weights = self.synapses.variables['w'].get_value().copy()
        error = None
        try:
            self.callback_count += 1
            action()
        except Exception as caught:
            error = caught
        finally:
            # Original physical arrays and true read-only aliases observe
            # callback mutations. NumPy's copied/rebound locals stay intact.
            for key, variable in prepared.program.variables.items():
                if not variable.persistent:
                    continue
                host = self.group.variables[key].get_value()
                values[key] = self._encode(host, variable)
                source = views.get(key, key)
                if source != key:
                    values[source] = self._encode(host, variable, shadow=True)
            count = int(eventspace[-1])
            changed = np.asarray(eventspace[:count], np.int32)
            if not np.array_equal(changed, visible):
                boundary.spikes = mx.array(np.isin(np.arange(prepared.neurons), changed).astype(np.uint32))
                boundary.changed_spikes = True
            imported = import_random(prepared, self.device)
            random = replace(random, stream=imported.stream, buffers=imported.buffers, pool=None)
            weights = self.synapses.variables['w'].get_value()
            if weights.tobytes() != before_weights.tobytes():
                from .configured import _ordered_layout
                sources = self.synapses.variables['_synaptic_pre'].get_value()
                destinations = self.synapses.variables['_synaptic_post'].get_value()
                order = _ordered_layout(sources, destinations, prepared.neurons)[0]
                encoded = np.asarray(weights, np.float64)[order].view(np.uint64)
                self.weights = mx.array(encoded if encoded.size else np.zeros(1, np.uint64))
        return random, error
