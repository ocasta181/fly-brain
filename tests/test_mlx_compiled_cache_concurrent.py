"""Actual original trials sharing a cold compiled-integer cache."""
from native_execution import forbid_network_cpu_steps
from pathlib import Path
import os
import time

import numpy as np
import pytest
from original_api_reference import (cpu_runtime, load_original_model, write_tiny_dataset,
                                    capture_rng_state, assert_rng_state_equal)


def _rendezvous(directory, name):
    directory = Path(directory)
    (directory / f'{name}-{os.getpid()}').touch()
    deadline = time.monotonic() + 15
    while len(list(directory.glob(name + '-*'))) < 2:
        if time.monotonic() > deadline:
            raise AssertionError('Both cache workers must reach ' + name)
        time.sleep(.01)


def _trial(paths, case="roundtrip"):
    import brian2 as b
    b.prefs.core.default_float_dtype = np.float32 if case == "paired" else np.float64
    original = load_original_model()
    observations = []
    collect = original.get_spk_trn
    def capture(monitor):
        group = monitor.source
        names = [name for name, eq in group.equations.items() if eq.type != 'subexpression']
        names += ['lastspike', 'not_refractory']
        observations.append({
            'state': {name: group.variables[name].get_value().copy() for name in names},
            'i': np.array(monitor.i[:]), 't': np.asarray(monitor.t[:]).copy(),
            'count': np.array(monitor.count[:]),
            'clock': group.clock.variables['timestep'].get_value().copy(),
            'rng': capture_rng_state(),
        })
        return collect(monitor)
    original.get_spk_trn = capture
    reset = {'roundtrip': 'x=x-0.25;a=int(x);q+=1',
             'boxed': 'q=(q+4294967297)%7;x=x-0.25;a=int(x)',
             'paired': 'x=x-0.25;a=sin(x)+cos(x)'}[case]
    params = dict(original.default_params, eqs=original.default_params['eqs']+'\na : 1\nx : 1\nq : integer\n',
                  eq_th='True', eq_rst='v=v_rst;g=0*mV;'+reset,
                  t_run=.2*b.ms, t_rfc=0*b.ms, r_poi=3700*b.Hz, r_poi2=1400*b.Hz)
    original.run_trial([0, 0], [2], [3], *paths, params)
    return observations[0]


def _worker(paths, directory, interrupt=False, case="roundtrip"):
    import subprocess
    import brian2 as b
    from brian2.devices.device import get_device
    from mlx_backend import compiled_integer, compiled_integer_frontend
    from mlx_backend.brian_device import MLXDevice
    directory = Path(directory)
    compiled_integer._CACHE = directory / compiled_integer._CACHE.name
    compiled_integer_frontend._CACHE = directory / compiled_integer_frontend._CACHE.name
    device = get_device()
    assert type(device) is MLXDevice
    b.seed(941)
    b.prefs.codegen.runtime.cython.cache_dir = str(directory / 'cython')
    # Hold successful legacy compiler invocations before their fixed-path
    # rename: both real compilers must finish before either publishes it.
    # This schedule reproduces a writer collision without modifying output.
    run = subprocess.run
    interrupted = False
    class InterruptedBuild(RuntimeError):
        pass
    def coordinated(command, *args, **kwargs):
        nonlocal interrupted
        result = run(command, *args, **kwargs)
        if '-o' in command:
            output = Path(command[command.index('-o')+1])
            if interrupt and not interrupted and output.suffix == '.tmp' and result.returncode == 0:
                interrupted = True
                (directory / f'interrupted-{os.getpid()}').write_text(str(output))
                raise InterruptedBuild('Compiler stopped before publication')
            if output.name.endswith('.pending.ll') and result.returncode == 0:
                _rendezvous(directory, output.stem)
        return result
    subprocess.run = coordinated
    before = b.Network.before_run
    forbid_network_cpu_steps(b.Network, None, 'Native trial executed a CPU timestep', before=before)
    _rendezvous(directory, 'trial')
    start = time.perf_counter()
    try:
        result = _trial(paths, case)
    except InterruptedBuild:
        assert device.native_runs == 0
        b.seed(941)
        result = _trial(paths, case)
    assert device.native_runs == 1
    return os.getpid(), time.perf_counter()-start, result


@pytest.mark.metal
@pytest.mark.parametrize("case,interrupt,legacy", [("roundtrip",False,False),
    ("roundtrip",True,False),("boxed",False,True),("paired",False,True)])
def test_cold_original_trials_share_compiled_cache(tmp_path, case, interrupt, legacy):
    from joblib import Parallel, delayed, parallel_backend
    from joblib.externals.loky import get_reusable_executor
    from mlx_backend.brian_device import original_api
    paths = write_tiny_dataset(tmp_path)
    with cpu_runtime(seed=941, target='cython') as b:
        expected = _trial(paths, case)
    legacy_files = []
    if legacy:
        from mlx_backend import compiled_integer, compiled_integer_frontend, clang32
        from unittest.mock import patch
        # Discover exact real-model cache keys, then retain truncated artifacts
        # in their previous namespaces. The next native trials must rebuild.
        with cpu_runtime(seed=941, target='cython') as b:
            previous_cache = b.prefs.codegen.runtime.cython.cache_dir
            b.prefs.codegen.runtime.cython.cache_dir = str(tmp_path / 'cython')
            try:
                with patch.object(compiled_integer, '_CACHE', tmp_path / compiled_integer._CACHE.name), \
                     patch.object(compiled_integer_frontend, '_CACHE', tmp_path / compiled_integer_frontend._CACHE.name):
                    with original_api(workers=False):
                        _trial(paths, case)
            finally:
                b.prefs.codegen.runtime.cython.cache_dir = previous_cache
        for directory in (tmp_path / compiled_integer._CACHE.name,
                          tmp_path / compiled_integer_frontend._CACHE.name,
                          tmp_path / 'cython' / 'native-float32-v2'):
            if not directory.exists():
                continue
            old = directory.with_name(directory.name.removesuffix('-v2'))
            directory.rename(old)
            for artifact in old.iterdir():
                if artifact.is_file():
                    artifact.write_text('')
                    legacy_files.append(artifact)
        assert legacy_files
    with cpu_runtime(seed=941, target='cython') as b:
        with original_api(workers=True):
            try:
                with parallel_backend('loky', n_jobs=2):
                    actual = Parallel(batch_size=1, pre_dispatch=2)(
                        delayed(_worker)(paths, tmp_path, interrupt=interrupt, case=case) for _ in range(2))
            finally:
                get_reusable_executor().shutdown(wait=True, kill_workers=True)
    assert len({pid for pid, _, _ in actual}) == 2
    for pid, seconds, got in actual:
        for name, want in expected['state'].items():
            assert got['state'][name].dtype == want.dtype
            assert got['state'][name].tobytes() == want.tobytes(), name
        for name in ('i', 't', 'count', 'clock'):
            np.testing.assert_array_equal(got[name], expected[name])
        assert_rng_state_equal(got['rng'], expected['rng'])
    if interrupt:
        assert list(tmp_path.glob('interrupted-*'))
        assert not list(tmp_path.rglob('*.tmp'))
    assert all(path.read_bytes() == b'' for path in legacy_files)
    if case == 'boxed':
        assert list(tmp_path.rglob('*.python-types-v3.json'))
    if case == 'paired':
        assert list((tmp_path / 'cython' / 'native-float32-v2').glob('*.s'))
    assert expected['i'].size == 8
    print('Concurrent trial participants:', [(pid, seconds) for pid, seconds, _ in actual])
