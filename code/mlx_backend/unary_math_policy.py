"""Qualified unary helper names; numerical implementations remain in providers."""

UFUNC_HELPERS = {
    'exp': ('d_exponential', '_eon_basic32_exp'),
    'log': ('d_logarithm', '_eon_basic32_log'),
    'expm1': ('_eon_stable_expm1', '_eon_basic32_expm1'),
    'log1p': ('_eon_stable_log1p', '_eon_basic32_log1p'),
    'arcsin': ('d_inverse_arcsin', 'f_inverse_arcsin'),
    'arccos': ('d_inverse_arccos', 'f_inverse_arccos'),
    'log10': ('_eon_hyper_log10', '_eon_hyper32_log10'),
    'sinh': ('_eon_hyper_sinh', '_eon_hyper32_sinh'),
    'cosh': ('_eon_hyper_cosh', '_eon_hyper32_cosh'),
    'tanh': ('_eon_hyper_tanh_numpy', '_eon_hyper32_tanh_numpy'),
    'sin': ('d_sine', '_eon_trig32_sine_numpy'),
    'cos': ('d_cosine_numpy', '_eon_trig32_cosine_numpy'),
    'tan': ('d_tangent', '_eon_trig32_tangent_numpy'),
}


def unary_helper(name: str, single: bool, target: str) -> str:
    """Select one qualified symbol without rendering or converting operands."""
    if name == 'arctan':
        return '_eon_arctan32' if single else '_eon_arctan64'
    if name == 'exprel':
        return '_eon_stable_exprel'
    function = UFUNC_HELPERS[name][1 if single else 0]
    if target != 'numpy':
        if name == 'tanh' or single and name in ('sin', 'cos', 'tan'):
            return function.removesuffix('_numpy') + '_scalar'
        if name == 'cos' and not single:
            return 'd_cosine'
    return function
