"""Dense ordinary expression output uses bounded chronological recording."""

import numpy as np
import pytest


@pytest.mark.metal
@pytest.mark.parametrize("target", ["numpy", "cython"])
@pytest.mark.parametrize("case", ["dense", "sparse", "fallback"])
@pytest.mark.parametrize("delay", [0., .2], ids=["zero-delay", "two-step-delay"])
def test_original_dense_recording_dispatch(tmp_path, monkeypatch, target, case, delay):
    import brian2 as b
    import original_api_reference as reference
    from mlx_backend import expression_recording as recording
    from original_api_protocol import check_original_api_runs, observe_run
    size = 64

    monkeypatch.setattr(reference, "write_tiny_dataset", reference.synthetic_dataset(size, labelled=False))
    monkeypatch.setattr(recording, "MAX_EVENTS", size*3)
    collect = recording.collect
    calls = []

    def observed(*args):
        calls.append(True)
        return None if case == "fallback" else collect(*args)

    monkeypatch.setattr(recording, "collect", observed)
    x = np.full(size, .75 if case != "sparse" else -.75)
    x[3] = .75

    def build(mode):
        fixture = reference.build_original_api_fixture(
            tmp_path/str(mode=='native'), parameters=dict(
                eqs="v:volt\ng:volt\nrfc:second\nx:1\na:1",
                eq_th="x>.5", eq_rst="a=a+.125;v=v_rst;g=0*mV",
                t_rfc=0*b.ms, t_dly=delay*b.ms),
            exc=(), exc2=(), silenced=(), initial_state={"x": x})
        return fixture

    def exercise(fixture, mode):
        for duration in (0., .9, .2, .7):
            yield observe_run(fixture,duration,rng=True)
        if mode=='native':assert bool(calls) == (case != "sparse")
    check_original_api_runs(build,exercise,monkeypatch,seed=953,target=target)
