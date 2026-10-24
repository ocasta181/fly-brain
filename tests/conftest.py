"""Shared actual-Metal execution witness; independent Brian references stay separate."""

import ast
from functools import partial
import importlib
import inspect
import sys

import pytest


@pytest.fixture(scope="module")
def native_launches():
    return []


@pytest.fixture(scope="module")
def native(native_launches):
    import brian2 as brian
    import mlx.core as mx

    assert mx.metal.is_available(), "This qualification requires an actual Metal device"
    launches = native_launches
    factory = mx.fast.metal_kernel

    def witnessed_factory(*args, **kwargs):
        kernel = factory(*args, **kwargs)

        def witnessed_launch(*launch_args, **launch_kwargs):
            outputs = kernel(*launch_args, **launch_kwargs)
            mx.eval(outputs)
            name, logical_ticks = kwargs["name"], 1
            inputs = launch_kwargs["inputs"]
            if name == "eon32_state":
                assert len(inputs[5].shape) == len(inputs[6].shape) == 3
                logical_ticks, batch, n = inputs[5].shape
                assert inputs[6].shape[:2] == (logical_ticks, batch)
                assert all(value.shape == (batch, n) for value in inputs[:5])
                assert all(value.shape == (batch, n) for value in outputs[:5])
                assert outputs[5].shape == (logical_ticks, batch, n)
                if len(outputs) == 7:
                    assert outputs[6].shape == (logical_ticks, batch, (n + 31) // 32)
            elif name == "eon32_propagate":
                assert len(inputs[0].shape) == 3
                logical_ticks, batch, n = inputs[0].shape
                assert outputs[0].shape == inputs[0].shape
                assert launch_kwargs["grid"] == (n, 8, logical_ticks * batch)
            launches.append((name, logical_ticks))
            return outputs

        return witnessed_launch

    def reject_cpu_steps(*args, **kwargs):
        raise AssertionError("A native run attempted Brian CPU timestepping")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mx.fast, "metal_kernel", witnessed_factory)
        previously_loaded = "mlx32.engine" in sys.modules
        api = importlib.import_module("mlx32.model")
        engine = importlib.import_module("mlx32.engine")
        if previously_loaded:
            engine = importlib.reload(engine)
        tree = ast.parse(inspect.getsource(engine))
        assert any(
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", getattr(node.func, "id", "")) == "metal_kernel"
            for node in ast.walk(tree)
        ), "The engine must construct a native Metal kernel"

        def checked_run(method, graph, steps, drive, **kwargs):
            before = len(launches)
            with pytest.MonkeyPatch.context() as guard:
                guard.setattr(brian.Network, "run", reject_cpu_steps)
                guard.setattr(brian, "run", reject_cpu_steps)
                result = getattr(engine, method)(graph, steps, drive, **kwargs)
            if graph.n_neurons and steps:
                chunk = kwargs.get("chunk_size", 32)
                warm_lengths = sorted({min(chunk, steps), steps % chunk} - {0})
                chunk_lengths = [min(chunk, steps - tick) for tick in range(0, steps, chunk)]
                block_size = min(kwargs.get("model", api.Model()).delay_steps, 32)
                blocks = [
                    min(block_size, length - offset)
                    for length in warm_lengths + chunk_lengths
                    for offset in range(0, length, block_size)
                ]
            else:
                warm_lengths, blocks = [], []
            for name in ("eon32_state", "eon32_propagate"):
                observed = [ticks for kernel, ticks in launches[before:] if kernel == name]
                expected_ticks = steps + sum(warm_lengths) if graph.n_neurons else 0
                assert sum(observed) == expected_ticks, f"Requested Metal logical ticks did not execute: {name}"
                assert observed == blocks, f"Requested Metal physical blocks did not execute: {name}"
            return result

        yield api, partial(checked_run, "run"), partial(checked_run, "run_batch")
