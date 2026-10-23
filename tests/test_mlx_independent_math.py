"""Zero-edge math execution preserves exact state and every spike word."""
from state_assertions import assert_recorded_spikes_equal
from dataclasses import replace

import numpy as np
import pytest


@pytest.mark.metal
@pytest.mark.parametrize('target',['numpy','cython'])
def test_binary64_independent_math_words_inputs_and_continuation(monkeypatch,target):
    b=pytest.importorskip('brian2')
    mx=pytest.importorskip('mlx.core')
    from configured_reference import reference
    from mlx_backend import configured
    from mlx_backend.model import resolve_model
    from mlx_backend.models import Connectome
    from test_mlx_configured import inputs

    monkeypatch.setitem(b.prefs,'codegen.target',target)
    monkeypatch.setitem(b.prefs,'core.default_float_dtype',np.float64)
    model=resolve_model({'eqs':'v:volt\ng:volt\nrfc:second\na:1\nb:1\nc:1\nd:1\nphase:integer\nchecksum:integer\nflag:boolean\n',
        'eq_th':'(v>v_th) or (t<dt) or ((t>=31*dt) and (t<34*dt)) or (t>=1023*dt)',
        't_rfc':0*b.ms,'t_dly':3.3*b.ms,
        'eq_rst':'v=v_rst;g=0*mV;phase+=1;a=log10(1+phase/512);b=sinh(phase/512);c=cosh(phase/512);d=tanh(phase/512);checksum+=int(a*1024)+int(b*1024)+int(c*1024)+int(d*1024);flag=not flag'})
    empty=np.empty(0,np.int32)
    graph=Connectome(np.arange(3),empty,empty,empty,np.empty(0))
    stimulus=inputs(model,steps=1057,trials=2,amplitudes=(60.125,-2.75,72.5))
    expected=reference(graph,stimulus,model)
    prepared=configured.prepare(graph,stimulus,model,layout_key=('independent-math',target))
    constrained=configured.prepare(graph,stimulus,model,allow_independent=False,
                                   layout_key=('independent-math',target))
    assert prepared.independent_neurons and prepared.window==32
    assert not constrained.independent_neurons and constrained.window==32

    def assert_state(state):
        actual=configured.host_state(state,prepared)
        for name,wanted in expected.state.items():
            assert actual[name].dtype==wanted.dtype,name
            assert actual[name].tobytes()==wanted.tobytes(),name

    for record in (True,False):
        result,state=configured.run_prepared(prepared,stimulus,record=record,
            return_state=True,chunk_steps=2048)
        assert_recorded_spikes_equal(result, expected, record)
        assert_state(state)
        assert not state.history and not state.due

    # Explicit continuation crosses31/32 and1023/1024 while retaining all words.
    partial=replace(prepared,window=1024)
    state=configured.initial_state(partial,2)
    events=mx.array(stimulus.events.reshape(-1))
    seen=[]
    for length in (31,1,1,991,33):
        begin=state.step
        state,bits=configured.advance(partial,state,events,1057,length)
        configured.evaluate_state(state)
        words=np.asarray(bits).reshape(-1,6)
        for bit in range(length):
            seen.extend((stimulus.trial_indices[row//3],row%3,begin+bit)
                        for row in np.flatnonzero(words[bit//32]&(1<<(bit%32))))
    assert sorted(seen,key=lambda row:(row[0],row[2],row[1]))==list(zip(
        expected.spikes.trials,expected.spikes.neurons,expected.spikes.steps))
    assert_state(state)
