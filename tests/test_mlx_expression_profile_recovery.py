"""Interrupted real compiler output must not publish a ready expression profile."""
from native_execution import forbid_network_cpu_steps
from pathlib import Path
import json
import os
import pickle
import subprocess
import sys

import numpy as np
import pytest
from original_api_reference import (cpu_runtime, load_original_model, write_tiny_dataset,
                                    capture_rng_state, assert_rng_state_equal)


def _trial(paths, phase):
    import brian2 as b
    original = load_original_model()
    observations = []
    collect = original.get_spk_trn
    def capture(monitor):
        group = monitor.source
        names = [name for name, eq in group.equations.items() if eq.type != 'subexpression']
        names += ['lastspike', 'not_refractory']
        observations.append(dict(
            state={name: group.variables[name].get_value().copy() for name in names},
            i=np.array(monitor.i[:]), t=np.asarray(monitor.t[:]).copy(),
            count=np.array(monitor.count[:]),
            clock=group.clock.variables['timestep'].get_value().copy(), rng=capture_rng_state()))
        return collect(monitor)
    original.get_spk_trn = capture
    eqs = 'v:volt\ng:volt\nrfc:second'
    params = dict(original.default_params, eqs=eqs+'\na:1', eq_th='True',
                  eq_rst='v=v_rst;g=0*mV;a=.13+rand()+.37',
                  t_run=.2*b.ms, t_rfc=0*b.ms, r_poi=3700*b.Hz, r_poi2=1400*b.Hz)
    original.run_trial([0, 0], [2], [3], *paths, params)
    return observations[0]


# Runs inside the real profile worker, after Clang has produced the real MIR.
# Both the legacy direct write and the atomic temporary write are interrupted.
_INTERRUPT = '''
import json,sys
from pathlib import Path
from mlx_backend.expression_profile_worker import main
write = Path.write_text
def interrupted(path, text, *args, **kwargs):
    if path.name == 'metadata.json' or (path.name.startswith('metadata.json.') and path.suffix == '.tmp'):
        metadata = json.loads(text)
        mir = path.parent / (metadata['stage'] + '.mir')
        assert mir.stat().st_size > 0
        assert '-stop-after=machine-combiner' in metadata['command']
        write(Path(sys.argv[2]), json.dumps(dict(stage=metadata['stage'], command=metadata['command'], mir_bytes=mir.stat().st_size)))
        write(path, text[:len(text)//2], *args, **kwargs)
        raise RuntimeError('Controlled interruption during metadata publication')
    return write(path, text, *args, **kwargs)
Path.write_text = interrupted
main(sys.argv[1])
'''


@pytest.mark.metal
def test_interrupted_profile_recovers_original_trial(tmp_path, monkeypatch):
    phase = 'caller'
    paths = write_tiny_dataset(tmp_path)
    oracle = tmp_path / 'cpu.pkl'
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(Path(__file__).parent)
    subprocess.run([sys.executable, str(Path(__file__).resolve()), str(oracle), phase,
                    *(str(path) for path in paths)], env=environment, check=True)
    expected = pickle.loads(oracle.read_bytes())
    from mlx_backend.brian_device import original_api
    from mlx_backend import expression_profile
    cache = tmp_path / 'profiles'
    monkeypatch.setenv('EON_MLX_EXPRESSION_CACHE', str(cache))
    expression_profile._apply.cache_clear()
    real_run = subprocess.run
    marker = tmp_path / 'interrupted.json'
    calls = []
    def interrupt_once(command, *args, **kwargs):
        if '-m' in command and 'mlx_backend.expression_profile_worker' in command:
            calls.append(command)
            if len(calls) == 1:
                command = [sys.executable, '-c', _INTERRUPT, command[-1], str(marker)]
        return real_run(command, *args, **kwargs)
    monkeypatch.setattr(subprocess, 'run', interrupt_once)
    with cpu_runtime(seed=941, target='cython') as b:
        before = b.Network.before_run
        forbid_network_cpu_steps(b.Network, monkeypatch, 'Native trial executed a CPU timestep', before=before,
                                 patch_runners=False)
        with original_api(workers=False):
            device = b.get_device()
            with pytest.raises(Exception, match='Compiled arithmetic code generation failed'):
                _trial(paths, phase)
            assert device.native_runs == 0
            assert not list(cache.rglob('metadata.json'))
            evidence = json.loads(marker.read_text())
            assert evidence['stage'] == 'reset'
            locks = {path: path.stat().st_ino for path in cache.rglob('profile.lock')}
            b.seed(941)
            actual = _trial(paths, phase)
            assert device.native_runs == 1
            assert all(path.stat().st_ino == inode for path, inode in locks.items())
    for name, value in expected['state'].items():
        assert actual['state'][name].dtype == value.dtype
        assert actual['state'][name].tobytes() == value.tobytes(), name
    for name in ('i', 't', 'count', 'clock'):
        np.testing.assert_array_equal(actual[name], expected[name])
    assert_rng_state_equal(actual['rng'], expected['rng'])
    assert len(calls) == 2
    assert not list(cache.rglob('*.tmp'))
    assert expected['i'].size == 8
    print('Profile interruption recovery:', phase, evidence, 'exact states/spikes/counts/clock/physical RNG')


if __name__ == '__main__':
    with cpu_runtime(seed=941, target='cython') as b:
        result = _trial(tuple(Path(p) for p in sys.argv[3:]), sys.argv[2])
        assert not any(name == 'mlx' or name.startswith('mlx.') for name in sys.modules)
        Path(sys.argv[1]).write_bytes(pickle.dumps(result))
