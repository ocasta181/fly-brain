"""Actual Brian NumPy state arithmetic retains each finite operation's bits."""
from native_execution import forbid_model_cpu_steps
from contextlib import nullcontext
import json
import uuid

import brian2 as b
import numpy as np
import pytest

from mlx_backend.brian_device import original_api
from mlx_backend.model import DEFAULT_EQUATIONS, DEFAULT_RESET
from original_api_reference import (
    assert_snapshot_equal, build_original_api_fixture, cpu_runtime,
)


pytestmark = pytest.mark.metal


@pytest.mark.parametrize('durations', [(.2,), (.1, .1)], ids=['window', 'continuation'])
def test_original_numpy_finite_g_drive_precision(tmp_path, monkeypatch, durations):
    expected = []
    observed_native = []
    for native in (False, True):
        with cpu_runtime(target='numpy', float_dtype=np.float64):
            monkeypatch.setitem(b.prefs, 'codegen.loop_invariant_optimisations', False)
            equations = DEFAULT_EQUATIONS.replace(
                '-g / tau', '(-g + drive(t,column)*mV) / tau') + 'column:integer\n'
            parameters = dict(
                eqs=equations, eq_th='i<2', eq_rst=DEFAULT_RESET,
                t_rfc=0*b.ms,
                drive=b.TimedArray([[1., 2.], [3., 4.]], dt=.1*b.ms),
            )
            with original_api(workers=False) if native else nullcontext() as device:
                fixture = build_original_api_fixture(
                    tmp_path/('metal' if native else 'cpu'),
                    parameters=parameters, exc=(), exc2=(), silenced=(),
                    initial_state=dict(
                        column=[0]*4, g=np.array([1., 2., 3., 4.])*b.mV,
                        v=np.array([-51., -50., -49., -48.])*b.mV,
                    ),
                )
                if native:
                    forbid_model_cpu_steps(fixture, monkeypatch, 'Native state update used a CPU timestep', patch_preparation=True)
                for step, duration in enumerate(durations):
                    snapshot = fixture.run_ms(duration)
                    label = 'native' if native else 'cpu'
                    np.savez(tmp_path/f'{label}-step-{step+1}.npz', **snapshot.state)
                    if native:
                        observed_native.append(snapshot)
                        program = device._sessions[fixture.network].prepared.program
                        (tmp_path/'native-state.metal').write_text(program.state_code)
                        (tmp_path/'native-statements.txt').write_text(repr(program.state_statements))
                    else:
                        expected.append(snapshot)
                        (tmp_path/'cpu-state.txt').write_text(
                            str(fixture.neurons.state_updater.codeobj.code))
    differences = []
    for step, (actual, reference) in enumerate(zip(observed_native, expected), 1):
        for name in reference.state:
            a, z = actual.state[name], reference.state[name]
            if a.dtype.kind != 'f':
                continue
            words = np.uint32 if a.dtype.itemsize == 4 else np.uint64
            for index in np.flatnonzero(a.view(words) != z.view(words)):
                differences.append(dict(step=step, name=name, index=int(index),
                    native=float(a[index]), cpu=float(z[index]),
                    native_bits=hex(int(a.view(words)[index])),
                    cpu_bits=hex(int(z.view(words)[index]))))
    (tmp_path/'differences.json').write_text(json.dumps(differences, indent=2)+'\n')
    for actual, reference in zip(observed_native, expected):
        assert_snapshot_equal(actual, reference)


@pytest.mark.parametrize('warmup_optimise', [True, False])
@pytest.mark.parametrize('record', [True, False])
@pytest.mark.parametrize('dtype', [np.float64, np.float32])
def test_actual_numpy_preserves_cached_statement_selection(
        tmp_path, monkeypatch, warmup_optimise, record, dtype):
    """Changing a preference must not replace the actual prepared CPU code."""
    prefix = 'state_cache_'+uuid.uuid4().hex[:8]
    drive, column = prefix+'_drive', prefix+'_column'
    equations = DEFAULT_EQUATIONS.replace(
        '-g / tau', f'(-g + {drive}(t,{column})*mV) / tau') + column+':integer\n'
    parameters = dict(eqs=equations, eq_th='i<2', eq_rst=DEFAULT_RESET,
        t_rfc=0*b.ms, **{drive:b.TimedArray([[1., 2.], [3., 4.]], dt=.1*b.ms)})
    initial = {column:[0]*4, 'g':np.array([1., 2., 3., 4.])*b.mV,
               'v':np.array([-51., -50., -49., -48.])*b.mV}
    expected = []
    for native in (False, True):
        with cpu_runtime(target='numpy', float_dtype=dtype):
            monkeypatch.setitem(b.prefs, 'codegen.loop_invariant_optimisations', warmup_optimise)
            with original_api(workers=False) if native else nullcontext() as device:
                # Device class is part of Brian's cache identity. Prime each
                # selected device with the same zero-timestep preparation.
                warmup = build_original_api_fixture(tmp_path/f'warmup-{native}',
                    parameters=parameters, exc=(), exc2=(), silenced=(), initial_state=initial)
                warmup.run_ms(0.)
                assert warmup.network.t_ == 0.
                if native:
                    assert device.native_steps == 0
                monkeypatch.setitem(b.prefs, 'codegen.loop_invariant_optimisations', not warmup_optimise)
                fixture = build_original_api_fixture(tmp_path/str(native),
                    parameters=parameters, exc=(), exc2=(), silenced=(), initial_state=initial)
                if not record:
                    fixture.network.remove(fixture.monitor)
                    fixture.monitor = b.SpikeMonitor(fixture.neurons, record=False)
                    fixture.network.add(fixture.monitor)
                if native:
                    forbid_model_cpu_steps(fixture, monkeypatch, 'Native cached state used a CPU timestep', patch_preparation=True)
                for step in range(2):
                    snapshot = fixture.run_ms(.1)
                    label = 'native' if native else 'cpu'
                    np.savez(tmp_path/f'{label}-cached-{step+1}.npz', **snapshot.state)
                    source = str(fixture.neurons.state_updater.codeobj.code)
                    (tmp_path/f'{label}-prepared.txt').write_text(source)
                    assert ('_lio_' in source) is warmup_optimise
                    if native:
                        session = device._sessions[fixture.network]
                        frozen = session.prepared.model.analysis.typed_statements
                        assert frozen is not None
                        assert bool(frozen['state'][0]) is warmup_optimise
                        (tmp_path/'frozen-statements.txt').write_text(repr({
                            phase:[[statement.__dict__ for statement in section]
                                   for section in sections]
                            for phase, sections in frozen.items()}))
                        (tmp_path/'model-signature.txt').write_text(session.prepared.model.signature)
                        (tmp_path/'native-state.metal').write_text(session.prepared.program.state_code)
                        assert_snapshot_equal(snapshot, expected[step])
                    else:
                        expected.append(snapshot)
