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
            launches.append(kwargs["name"])
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
                warmup = sum({min(chunk, steps), steps % chunk} - {0})
                state_updates = launches[before:].count("eon32_state")
                assert state_updates == steps + warmup, "Requested Metal state updates did not execute"
            return result

        yield api, partial(checked_run, "run"), partial(checked_run, "run_batch")
