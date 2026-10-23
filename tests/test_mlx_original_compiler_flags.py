"""The original Eon equations retain the selected Cython compiler policy."""
from native_execution import forbid_cpu_steps, require_metal
from hashlib import sha256

import numpy as np
import pytest

mx = require_metal('Metal is required', allow_module_level=True)

from test_mlx_brian_device import _run_pair

pytestmark = pytest.mark.metal


@pytest.mark.parametrize("policy", ["strict", "no_contract", "strict_reset_zero_delay"])
def test_original_default_equations_keep_nondefault_compiler_flags(
    tmp_path, monkeypatch, policy,
):
    import brian2 as b
    from brian2.codegen.cpp_prefs import get_compiler_and_args

    flags = get_compiler_and_args()[1]
    flags = ([flag for flag in flags if flag != "-ffast-math"]
             + ["-fno-fast-math", "-ffp-contract=off"] if policy.startswith("strict")
             else [*flags, "-ffp-contract=off"])
    monkeypatch.setitem(b.prefs, "codegen.cpp.extra_compile_args", flags)
    monkeypatch.setitem(b.prefs, "codegen.runtime.cython.cache_dir", str(tmp_path/"cython"))
    from brian2.codegen.runtime.cython_rt.extension_manager import cython_extension_manager
    create_extension = cython_extension_manager.create_extension
    policy_key = sha256(repr(flags).encode()).hexdigest()
    compiled_flags = []
    def compile_selected_policy(code, *args, **kwargs):
        # Brian's normal extension key omits flags and cache directory. A
        # comment isolates this test's actual binary without changing code.
        assert tuple(kwargs["extra_compile_args"]) == tuple(flags)
        compiled_flags.append(tuple(kwargs["extra_compile_args"]))
        return create_extension(code+"\n# eon-test-compiler-policy:"+policy_key+"\n", *args, **kwargs)
    monkeypatch.setattr(cython_extension_manager, "create_extension", compile_selected_policy)
    monkeypatch.setattr(cython_extension_manager, "_code_cache", {})
    def trap_cpu_timesteps(fixture):
        forbid_cpu_steps(fixture.network, monkeypatch, 'CPU timestep executed on MLX path')

    parameters = {"t_dly": .2*b.ms, "r_poi": 3700*b.Hz, "r_poi2": 7200*b.Hz}
    if policy == "strict_reset_zero_delay":
        # The state frontend owns the mask, the reset edits it, and zero-delay
        # input draws must observe its earlier synapses-phase value.
        parameters.update(t_dly=0*b.ms,
                          eq_rst="v=v_rst;w=0;g=0*mV;not_refractory=True")
    actual, _ = _run_pair(
        tmp_path, (0., .15, .15, .3), target="cython",
        parameters=parameters,
        initial_state={"v": np.array([-44., -49., -48., -50.])*b.mV,
                       "g": np.array([3., -2., 4., 1.])*b.mV},
        mutations={0: lambda fixture: trap_cpu_timesteps(fixture)
                   if type(b.get_device()).__name__ == "MLXDevice" else None},
    )
    assert actual.neurons.state_updater.codeobj.extra_compile_args == flags
    assert compiled_flags
