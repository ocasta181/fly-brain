"""Actual Cython state arithmetic for a custom threshold on canonical LIF."""
from state_assertions import assert_state_bytes_equal
from hashlib import sha256

import numpy as np
import pytest

from native_execution import require_metal
mx = require_metal('Metal is required', allow_module_level=True)
b = pytest.importorskip("brian2")

from configured_reference import reference
from mlx_backend import configured
from mlx_backend.model import DEFAULT_EQUATIONS, resolve_model
from mlx_backend.models import Connectome, Stimulus

pytestmark = pytest.mark.metal


def test_cython_custom_threshold_preserves_canonical_voltage_raw_bits(tmp_path):
    previous = b.prefs.codegen.runtime.cython.cache_dir
    b.prefs.codegen.runtime.cython.cache_dir = str(tmp_path / "cython-cache")
    try:
        model = resolve_model({
            "eqs": DEFAULT_EQUATIONS + "counter : integer\nflag : boolean\n",
            "eq_rst": "v=v_rst; g=0*mV; counter+=1; flag=not flag",
            "eq_th": "(v > v_th) or (flag and (counter > 2))",
        }, dt_ms=.1, cpu_target="cython")
        graph = Connectome(np.array(["source", "target"]), np.array([0], np.int32),
            np.array([1], np.int32), np.array([185], np.int32), np.array([50.875]))
        events = np.zeros((1, 21, 2), np.uint8)
        events[0, 0, 0] = 1
        stimulus = Stimulus(events, (0, 1), (200., 0.), (0,), 0, 0,
            sha256(events.tobytes()).hexdigest(), model.dt_ms, (68.75, 68.75))
        expected = reference(graph, stimulus, model, cpu_target="cython")
        # The source spike arrives at step19. Step20 applies the sensitive
        # state update while the target remains below threshold, so reset
        # cannot conceal a voltage error.
        assert expected.spikes.neurons.tolist() == [0]
        assert expected.spikes.steps.tolist() == [1]
        assert expected.state["g"][0, 1] > 0
        assert expected.state["v"][0, 1] < model.v_th
        prepared = configured.prepare(graph, stimulus, model)
        actual, state = configured.run_prepared(prepared, stimulus, return_state=True)
        observed = configured.host_state(state, prepared)
        assert set(observed) == set(expected.state)
        assert_state_bytes_equal({name: observed[name] for name in expected.state}, expected.state)
        for name in ("trials", "neurons", "steps"):
            assert getattr(actual.spikes, name).tobytes() == getattr(expected.spikes, name).tobytes()
        assert any((tmp_path / "cython-cache").rglob("*.so"))
    finally:
        b.prefs.codegen.runtime.cython.cache_dir = previous


def test_custom_threshold_still_fires_at_physical_rest(tmp_path):
    previous = b.prefs.codegen.runtime.cython.cache_dir
    b.prefs.codegen.runtime.cython.cache_dir = str(tmp_path / 'cython-cache')
    try:
        model = resolve_model({
            'eqs': DEFAULT_EQUATIONS + 'counter : integer\nflag : boolean\n',
            'eq_rst': 'v=v_rst; g=0*mV; counter+=1; flag=not flag',
            'eq_th': '(v > v_th) or (flag and (counter > 2))',
        }, dt_ms=.1, cpu_target='cython')
        graph = Connectome(np.array(['source','rest']), np.array([0],np.int32),
            np.array([1],np.int32),np.array([0],np.int32),np.array([0.]))
        events = np.zeros((1,12,1),np.uint8)
        events[0,[0,3,6],0] = 1
        stimulus = Stimulus(events,(0,),(200.,),(0,),0,0,
            sha256(events.tobytes()).hexdigest(),model.dt_ms,(68.75,))
        expected = reference(graph,stimulus,model,cpu_target='cython',record_state=True)
        assert expected.spikes.steps.tolist() == [1,4,7,8]
        assert expected.traces['v'][0,7,0].tobytes() == np.float64(model.v_0).tobytes()
        assert expected.traces['g'][0,7,0] == 0
        assert expected.traces['flag'][0,7,0] and expected.traces['counter'][0,7,0] == 3
        prepared = configured.prepare(graph,stimulus,model)
        assert prepared.program.canonical_custom_lif
        assert prepared.program.equilibrium is not None
        assert prepared.program.refractory_elapsed_threshold is not None
        assert not prepared.compact_idle
        actual,state = configured.run_prepared(prepared,stimulus,return_state=True)
        observed = configured.host_state(state,prepared)
        assert_state_bytes_equal({name: observed[name] for name in expected.state}, expected.state)
        for name in ('trials','neurons','steps'):
            assert getattr(actual.spikes,name).tobytes() == getattr(expected.spikes,name).tobytes()
        assert any((tmp_path/'cython-cache').rglob('*.so'))
    finally:
        b.prefs.codegen.runtime.cython.cache_dir = previous


@pytest.mark.parametrize('write',['rfc=0*ms','lastspike=t'])
def test_custom_reset_clock_writes_keep_release_cache_disabled(write):
    model = resolve_model({
        'eqs': DEFAULT_EQUATIONS + 'counter : integer\nflag : boolean\n',
        'eq_rst': 'v=v_rst; g=0*mV; counter+=1; flag=not flag; '+write,
        'eq_th': '(v > v_th) or (flag and (counter > 2))',
    },dt_ms=.1,cpu_target='cython')
    from mlx_backend.compiler import compile_model
    program = compile_model(model,neurons=2)
    assert program.refractory_elapsed_threshold is None
