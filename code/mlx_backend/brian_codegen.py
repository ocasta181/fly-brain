"""Audited original-model arithmetic policy for Brian's Cython/Clang target.

The canonical generated state update is compiled with fast-math by default.
On macOS arm64 Clang reassociates its scalar conductance coefficient, uses two
fused multiply-adds for voltage, and rewrites timestep division to reciprocal
multiplication. These are observable, so preserving the unfused NumPy expression
does not preserve the default compiled program. Other compiled forms fail
explicitly until their arithmetic policy has an independent comparison.
"""

from dataclasses import replace
import math
import platform

import numpy as np


def _cython_lif_binary32_dynamics(model):
    """Audited compiled binary32 temporaries over binary64 LIF state."""
    from .binary64 import literal

    dt, tau, tm = np.float64(model.dt_s), np.float64(model.tau), np.float64(model.t_mbr)
    # The persistent arrays retain their construction dtype. Clang lowers
    # Brian's later float32 temporaries with expf, binary32 coefficient
    # operations, two binary64 FMAs, then a binary32 assignment round.
    import ctypes
    library = ctypes.CDLL(None)
    expf = library.expf
    expf.argtypes, expf.restype = [ctypes.c_float], ctypes.c_float
    fma = library.fma
    fma.argtypes, fma.restype = [ctypes.c_double]*3, ctypes.c_double
    decay_g = np.float32(expf(float(np.float32(-dt/tau))))
    decay_v = np.float32(expf(float(np.float32(-dt/tm))))
    grow_g = np.float32(expf(float(np.float32(dt/tau))))
    grow_v = np.float32(expf(float(np.float32(dt/tm))))
    ratio = np.float32(tau/(tm-tau))
    coefficient = np.float32(np.float32(decay_v*ratio)
                             *np.float32(decay_g*np.float32(grow_g-grow_v)))
    baseline = np.float32(fma(-model.v_0, float(decay_v), model.v_0))
    dynamics = f"""
        if(not_refractory) {{
            ulong eon_next_v = f_to_d(f_from_d(d_fma({literal(decay_v)}, v,
                d_fma({literal(coefficient)}, g, {literal(baseline)}))));
            g = f_to_d(f_from_d(d_mul({literal(decay_g)}, g)));
            v = eon_next_v;
        }}
    """
    return dynamics


def original_codegen_policy(prepared, group):
    from .binary64 import literal
    from .model import DEFAULT_EQUATIONS, DEFAULT_RESET, DEFAULT_THRESHOLD, _expression_key
    from .brian_device import UnsupportedBrianNetwork

    codeobj = group.state_updater.codeobj
    if codeobj.class_name == "numpy":
        return prepared
    if codeobj.class_name != "cython":
        raise UnsupportedBrianNetwork("Unsupported Brian CPU code-generation arithmetic target")
    if prepared.model.state_dtype == "float32":
        # The typed compiler already lowers the actual optimized Cython graph;
        # the canonical binary64 LIF replacement would erase its assignment rounds.
        return prepared
    if 'state' in prepared.program.compiled_integer_stages:
        # The complete actual Cython frontend/selected-compiler state block
        # already owns its casts, arithmetic, guards and error boundaries.
        return prepared
    flags = tuple(codeobj.extra_compile_args)
    model, program = prepared.model, prepared.program
    if program.refractory_policy == "cython_fast":
        # A compiler-owned audited lowering (e.g. a TimedArray threshold)
        # already preserves the actual compiled arithmetic and time policy.
        return prepared
    if "-ffast-math" not in flags:
        raise UnsupportedBrianNetwork("Nondefault Cython compiler flags need their own native arithmetic policy")
    if model.analysis is not None:
        from brian2.core.functions import DEFAULT_FUNCTIONS
        refractory_only='not_refractory = timestep(t - lastspike, dt) >= timestep(rfc, dt)'
        if (platform.system()=='Darwin' and platform.machine()=='arm64'
                and '-ffp-contract=off' not in flags
                and _expression_key(model.analysis.state_abstract)==_expression_key(refractory_only)
                and codeobj.variables.get('timestep')is DEFAULT_FUNCTIONS['timestep']
                and all(program.variables[name].dtype=='float64'for name in('lastspike','rfc'))):
            # This complete state block contains only the same stock timestep
            # calls already audited below. Threshold/reset arithmetic is
            # lowered separately from its actual target code objects.
            offset=np.float64(.001)*np.float64(model.dt_s)
            reciprocal=np.float64(1.)/np.float64(model.dt_s)
            refractory=("not_refractory = d_to_int(d_mul(d_sub(d_add(t, " + literal(offset)
                + "), lastspike), " + literal(reciprocal) + ")) >= d_to_int(d_mul(d_add(rfc, "
                + literal(offset) + "), " + literal(reciprocal) + ")); ")
            return replace(prepared,program=replace(program,refractory_code=refractory,
                refractory_policy='cython_fast'))
    if (platform.system() != "Darwin" or platform.machine() != "arm64"
            or "-ffp-contract=off" in flags
            or _expression_key(model.eqs) != _expression_key(DEFAULT_EQUATIONS)
            or _expression_key(model.eq_rst) != _expression_key(DEFAULT_RESET)
            or _expression_key(model.eq_th) != _expression_key(DEFAULT_THRESHOLD)):
        raise UnsupportedBrianNetwork("The compiled fast-math arithmetic policy is audited for the original Eon model on macOS arm64")
    dt, tau, tm = np.float64(model.dt_s), np.float64(model.tau), np.float64(model.t_mbr)
    if tau <= 0 or tm <= 0 or tau == tm:
        raise UnsupportedBrianNetwork("Compiled Eon time constants need distinct positive values")
    decay_g = np.float64(math.exp(float(-dt/tau)))
    decay_v = np.float64(math.exp(float(-dt/tm)))
    grow_g = np.float64(math.exp(float(dt/tau)))
    grow_v = np.float64(math.exp(float(dt/tm)))
    coefficient = ((decay_g*decay_v)*(np.float64(1.)/(tm-tau)))*((grow_g-grow_v)*tau)
    offset = np.float64(.001)*dt
    reciprocal = np.float64(1.)/dt
    refractory = ("not_refractory = d_to_int(d_mul(d_sub(d_add(t, " + literal(offset)
        + "), lastspike), " + literal(reciprocal) + ")) >= d_to_int(d_mul(d_add(rfc, "
        + literal(offset) + "), " + literal(reciprocal) + ")); ")
    dynamics = f"""
        if(not_refractory) {{
            ulong eon_next_v = d_fma(d_sub(v, v_0), {literal(decay_v)},
                                    d_fma({literal(coefficient)}, g, v_0));
            g = d_mul({literal(decay_g)}, g);
            v = eon_next_v;
        }}
    """
    if model.analysis.temporary_dtype == "float32":
        dynamics = _cython_lif_binary32_dynamics(model)
    program = replace(program, refractory_code=refractory, dynamics_code=dynamics,
                      refractory_policy="cython_fast")
    return replace(prepared, program=program)
