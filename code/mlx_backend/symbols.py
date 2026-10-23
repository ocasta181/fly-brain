"""Injective backend identifiers for Brian data symbols, independent of types.

Public state/namespace names remain dictionary keys. Function identifiers use
their existing registries; only data references and declarations use this map.
Hex encoding also distinguishes user namespace names beginning with '_'.
"""

# These names are the existing runtime/compiler ABI. Brian owns the clock and
# index names; the canonical Eon states and parameters have dedicated paths.
_ABI_NAMES = frozenset({
    "v", "g", "rfc", "lastspike", "not_refractory", "v_0", "v_rst", "v_th",
    "tau", "t_mbr", "t", "dt", "t_in_timesteps", "i", "N", "_cond",
})


def variable_code_name(name):
    if name in _ABI_NAMES:
        return name
    return "_eon_var_"+name.encode("utf8").hex()


def variable_code_names(names):
    return {name: variable_code_name(name) for name in names}
