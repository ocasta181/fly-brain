"""Keep real Brian preparation while forbidding CPU timesteps on native paths."""


def forbidden_call(message):
    def forbidden(*args, **kwargs):
        raise AssertionError(message)
    return forbidden


def _setattr(owner, attribute, value, monkeypatch):
    (setattr if monkeypatch is None else monkeypatch.setattr)(owner, attribute, value)


def forbid_call(owner, attribute, message, monkeypatch=None):
    _setattr(owner, attribute, forbidden_call(message), monkeypatch)


def _forbid_runners(runners, message, monkeypatch):
    forbidden = forbidden_call(message)
    for obj in runners:
        for codeobj in obj.code_objects:
            _setattr(codeobj, "run", forbidden, monkeypatch)


def forbid_cpu_steps(network, monkeypatch=None, message="CPU timestep executed", *, runners=None,
                     patch_preparation=False):
    before = network.before_run

    def before_run(namespace):
        before(namespace)
        _forbid_runners(network.sorted_objects if runners is None else runners(), message, monkeypatch)

    _setattr(network, "before_run", before_run, monkeypatch if patch_preparation else None)


def forbid_network_cpu_steps(network_type, monkeypatch=None, message="CPU timestep executed", *, before=None,
                             patch_runners=True):
    before = network_type.before_run if before is None else before

    def before_run(network, namespace):
        before(network, namespace)
        _forbid_runners(network.sorted_objects, message, monkeypatch if patch_runners else None)

    _setattr(network_type, "before_run", before_run, monkeypatch)


def forbid_model_cpu_steps(fixture, monkeypatch, message="CPU timestep executed", *, include_inputs=False,
                           patch_preparation=False):
    def runners():
        return (fixture.neurons.state_updater, fixture.neurons.thresholder["spike"],
                fixture.neurons.resetter["spike"], *fixture.synapses._pathways,
                *(fixture.inputs if include_inputs else ()))

    forbid_cpu_steps(fixture.network, monkeypatch, message, runners=runners,
                     patch_preparation=patch_preparation)


def require_metal(message, *, allow_module_level=False):
    """Require optional Metal execution at the caller's existing gate."""
    __tracebackhide__ = True
    import pytest

    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip(message, allow_module_level=allow_module_level)
    return mx


def forbid_cpu_code_objects(monkeypatch, message):
    from brian2.codegen.runtime.numpy_rt.numpy_rt import NumpyCodeObject
    from brian2.codegen.runtime.cython_rt.cython_rt import CythonCodeObject

    forbidden = forbidden_call(message)
    monkeypatch.setattr(NumpyCodeObject, "run", forbidden)
    monkeypatch.setattr(CythonCodeObject, "run", forbidden)
