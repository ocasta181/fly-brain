"""Exact native reduction in the small scalar32 sine acceleration path."""
import numpy as np
import pytest

@pytest.mark.metal
def test_native_integer_reduction_preserves_nearest_even_bits():
    """Signed 49-bit remainders include every binade and tie direction."""
    import mlx.core as mx
    from mlx_backend import binary64,binary32,math64
    from mlx_backend.trigonometry32 import metal_trigonometry32_header
    from test_mlx_sine32_certificate import _decomposition_witnesses
    values=np.unique(np.r_[np.array(_decomposition_witnesses(),np.int64),
        np.random.default_rng(20261019).integers(-(1<<49),1<<49,32768,dtype=np.int64)])
    expected=values.astype(np.float32).view(np.uint32)
    kernel=mx.fast.metal_kernel(name='sine32_floor_integer_conversion',input_names=['values'],output_names=['out'],
        header=binary64.MSL_HEADER+binary32.MSL_HEADER+math64.MSL_MATH_HEADER+metal_trigonometry32_header({'sin'}),
        source='uint i=thread_position_in_grid.x;float high=float(values[i]);long residual=values[i]-long(high);high=_eon_sin32_integer_nearest(values[i],high,residual);out[2*i]=as_type<uint>(high);out[2*i+1]=f_from_int(values[i]);',
        compile_options={'math_mode':'safe'})
    output,=kernel(inputs=[mx.array(values)],grid=(len(values),1,1),threadgroup=(128,1,1),
        output_shapes=[(len(values),2)],output_dtypes=[mx.uint32])
    mx.eval(output)
    np.testing.assert_array_equal(np.asarray(output)[:,0],expected)
    np.testing.assert_array_equal(np.asarray(output)[:,1],expected)

@pytest.mark.metal
def test_product_fast_normalization_matches_general_sum():
    """The dominance guard retains cancellation/FTZ fallback behavior."""
    import mlx.core as mx
    from mlx_backend import binary64,binary32,math64
    from mlx_backend.trigonometry32 import metal_trigonometry32_header
    rng=np.random.default_rng(20261020)
    # Normalized pairs cover certificate-scale values and extreme low tails.
    high=rng.uniform(-2,2,(8192,2)).astype(np.float32)
    low=(high.astype(np.float64)*rng.uniform(-2**-24,2**-24,(8192,2))).astype(np.float32)
    pairs=np.column_stack((high[:,0],low[:,0],high[:,1],low[:,1]))
    # Deliberately unnormalized values exercise the retained general path.
    extra=np.array([[1,-1,1,0],[1,2**24,1,0],[-0.,0.,0.,-0.],
        [2**-120,2**-121,2**-30,2**-31],[2**-64,2**-89,2**-64,2**-89],
        [1,0,-1,0],[0,0,1,0]],dtype=np.float32)
    pairs=np.concatenate((pairs,extra));raw=pairs.view(np.uint32)
    header=binary64.MSL_HEADER+binary32.MSL_HEADER+math64.MSL_MATH_HEADER+metal_trigonometry32_header({'sin'})
    source='''uint i=thread_position_in_grid.x;
    _eon_sin32_dd a={as_type<float>(raw[4*i]),as_type<float>(raw[4*i+1])};
    _eon_sin32_dd b={as_type<float>(raw[4*i+2]),as_type<float>(raw[4*i+3])};
    float p=a.hi*b.hi,t=fma(a.hi,b.hi,-p);t=fma(a.hi,b.lo,t);t=fma(a.lo,b.hi,t);
    auto original=_eon_sin32_dd_sum(p,t),candidate=_eon_sin32_dd_mul(a,b);
    out[4*i]=as_type<uint>(original.hi);out[4*i+1]=as_type<uint>(original.lo);
    out[4*i+2]=as_type<uint>(candidate.hi);out[4*i+3]=as_type<uint>(candidate.lo);'''
    kernel=mx.fast.metal_kernel(name='sine32_floor_product_normalization',input_names=['raw'],output_names=['out'],header=header,source=source,compile_options={'math_mode':'safe'})
    output,=kernel(inputs=[mx.array(raw)],grid=(len(raw),1,1),threadgroup=(128,1,1),output_shapes=[(len(raw),4)],output_dtypes=[mx.uint32]);mx.eval(output)
    words=np.asarray(output).copy()
    # A zero low component has the same exact enclosure for either zero sign.
    for column in (1,3):words[(words[:,column]&0x7fffffff)==0,column]=0
    np.testing.assert_array_equal(words[:,:2],words[:,2:])


def test_relative_sine_enclosure_against_pinned_cpu_graph():
    """Rational error polynomials prove the guard for every accepted reduction."""
    from fractions import Fraction as F
    from test_mlx_sine32_certificate import _coefficient_errors,_PINNED_COEFFICIENTS
    import struct
    u,d,tiny=F(1,2**24),F(1,2**53),F(1,2**120)
    class P(tuple):
        def __new__(cls,terms=(0,)):return tuple.__new__(cls,map(F,terms))
        def __add__(a,b):
            b=b if isinstance(b,P)else P((b,))
            return P(tuple((a[k]if k<len(a)else 0)+(b[k]if k<len(b)else 0)for k in range(max(len(a),len(b)))))
        __radd__=__add__
        def __mul__(a,b):
            b=b if isinstance(b,P)else P((b,));terms=[F(0)]*(len(a)+len(b)-1)
            for i,x in enumerate(a):
                for j,y in enumerate(b):terms[i+j]+=x*y
            return P(terms)
        __rmul__=__mul__
    errors=_coefficient_errors()
    def coefficient(raw,native):
        return P((abs(F(struct.unpack('d',struct.pack('Q',raw))[0])),)),P((errors[raw]if native else 0,))
    def operation(a,b,c=None,*,native):
        am,ae=a;bm,be=b;product=am*bm;error=am*be+bm*ae+ae*be
        if native:
            local=16*u*u*(am+ae)*(bm+be)+tiny;error+=local
            if c is None:return product,error
            cm,ce=c
            return product+cm,error+ce+16*u*u*((am+ae)*(bm+be)+local+cm+ce)+tiny
        magnitude=product+(c[0]if c else P())
        propagation=error+(c[1]if c else P())
        return magnitude,propagation+d*(magnitude+propagation)
    r=(P((0,1)),P())
    def evaluate(native,cosine=False):
        mul=lambda a,b:operation(a,b,native=native)
        fma=lambda a,b,c:operation(a,b,c,native=native)
        coef=lambda raw:coefficient(raw,native)
        q=mul(r,r);cube=mul(r,q)
        if native:
            p=fma(q,coef(_PINNED_COEFFICIENTS[3])if cosine else(P(),P()),coef(_PINNED_COEFFICIENTS[4 if cosine else 0]))
            p=fma(q,p,coef(_PINNED_COEFFICIENTS[5 if cosine else 1]))
            p=fma(q,p,coef(_PINNED_COEFFICIENTS[6 if cosine else 2]))
            return fma(q if cosine else cube,p,(P((1,)),P())if cosine else r)
        if cosine:
            q2=mul(q,q)
            p=fma(q,coef(_PINNED_COEFFICIENTS[3]),coef(_PINNED_COEFFICIENTS[4]))
            leading=fma(q,coef(_PINNED_COEFFICIENTS[5]),coef(_PINNED_COEFFICIENTS[6]))
            return fma(q,fma(q2,p,leading),(P((1,)),P()))
        fifth=mul(q,cube)
        p=fma(q,coef(_PINNED_COEFFICIENTS[0]),coef(_PINNED_COEFFICIENTS[1]))
        return fma(fifth,p,fma(cube,coef(_PINNED_COEFFICIENTS[2]),r))
    native,cpu=evaluate(True),evaluate(False);error=native[1]+cpu[1]
    relative_error=error[0]*2**49+sum(error[1:])
    assert relative_error<F(1,2**43)
    # The high component is within u of the exact dyadic reduction. The
    # guard's addition error is <2^-46 |r|; the selected 2^-42*|hi| covers both.
    assert relative_error+F(1,2**46)<(1-u)*F(1,2**42)
    cosine_error=evaluate(True,True)[1]+evaluate(False,True)[1]
    global_error=max(sum(error),sum(cosine_error))
    assert global_error+F(1,2**46)<F(1,2**42)
    from mlx_backend.trigonometry32_certificate import MSL_HEADER
    assert 'fabs(reduced)*0x1p-42f' in MSL_HEADER
    assert '_eon_sin32_dd_certify<true>(value,result,reduced.hi)' in MSL_HEADER



@pytest.mark.metal
@pytest.mark.parametrize("case", ["small", "large", "numpy", "float64", "no-sine", "paired"])
def test_actual_brian_scalar_sine_group_policy(tmp_path, monkeypatch, case):
    """Observe real neuron dispatches and compare the upstream CPU contract."""
    from mlx_backend import configured
    from trigonometry_checks import OriginalComparison

    dtype = "float64" if case == "float64" else "float32"
    target = "numpy" if case == "numpy" else "cython"
    count = 16385 if case == "large" else 257
    values = np.random.default_rng(20261017).uniform(-8*np.pi, 8*np.pi, count).astype(dtype)
    symbol = "occupancy_phase"
    expression = (f"{symbol}+1/64" if case == "no-sine" else
                  f"sin({symbol})+cos({symbol})" if case == "paired" else f"sin({symbol})")
    options = dict(state_inputs=True, dtype=dtype, phase_symbol=symbol,
                   result_expression=expression+f"; {symbol}+=1/32",
                   run_steps=(1, 2, 0, 3), cache_suffix="-sine-occupancy")
    comparison = OriginalComparison(target,"sin",values,tmp_path,**options)
    observed = []
    factory = configured._neuron_kernel

    def traced_factory(*args, **kwargs):
        kernel = factory(*args, **kwargs)

        def launch(**launch_kwargs):
            observed.append(launch_kwargs["threadgroup"][0])
            return kernel(**launch_kwargs)

        return launch

    monkeypatch.setattr(configured, "_neuron_kernel", traced_factory)
    with comparison.native() as (actual,native,device):
        assert device.native_steps == 6 and device.native_runs == 4
    assert observed and set(observed) == ({64} if case == "small" else {256})


@pytest.mark.metal
@pytest.mark.parametrize("mode", ["nearest", "toward-zero"])
def test_integer_decomposition_corrects_both_cast_modes(mode):
    """The cast correction covers either mode for every signed |value|<2^49.

    Below 2^24 an integer cast is exact. Otherwise, a cast toward zero is
    adjacent to the exact value and the spacing away from zero is 2^(E-23).
    Increment its magnitude word iff the residual exceeds half that spacing,
    or equals half and the lower word is odd. A nearest-even cast needs no
    correction; a cast rounded away from zero is unchanged.
    The final residual has magnitude <= 2^24 and is exactly representable in
    either cast mode. Independent quotient/remainder witnesses cover ties,
    both signs, binade boundaries and the full supported reduction range.
    """
    import mlx.core as mx
    from mlx_backend import binary64, binary32, math64
    from mlx_backend.trigonometry32 import metal_trigonometry32_header
    from test_mlx_sine32_certificate import _decomposition_witnesses, _integer_rne32

    values = np.unique(np.r_[np.array(_decomposition_witnesses(), np.int64),
                            np.random.default_rng(20261019).integers(-(1 << 49), 1 << 49,
                                                                   32768, dtype=np.int64)])
    expected = np.array([_integer_rne32(int(value)) for value in values], np.uint32)
    if mode == "toward-zero":
        truncated = []
        for raw in values:
            value = int(raw)
            shift = max(0, abs(value).bit_length()-24)
            magnitude = (abs(value) >> shift) << shift
            truncated.append(-magnitude if value < 0 else magnitude)
        supplied = np.asarray(truncated, np.int64).astype(np.float32).view(np.uint32)
    else:
        supplied = expected
    header = binary64.MSL_HEADER+binary32.MSL_HEADER+math64.MSL_MATH_HEADER
    header += metal_trigonometry32_header({"sin"})
    kernel = mx.fast.metal_kernel(name="sine32_integer_cast_mode", input_names=["values", "supplied"],
        output_names=["out"], header=header, source='''
        uint i=thread_position_in_grid.x;float high=as_type<float>(supplied[i]);
        long residual=values[i]-long(high);
        high=_eon_sin32_integer_nearest(values[i],high,residual);
        out[2*i]=as_type<uint>(high);out[2*i+1]=as_type<uint>(float(residual));
        ''', compile_options={"math_mode": "safe"})
    output, = kernel(inputs=[mx.array(values), mx.array(supplied)], grid=(len(values), 1, 1),
                     threadgroup=(128, 1, 1), output_shapes=[(len(values), 2)], output_dtypes=[mx.uint32])
    mx.eval(output)
    actual = np.asarray(output)
    np.testing.assert_array_equal(actual[:, 0], expected)
    residual = values-expected.view(np.float32).astype(np.int64)
    assert np.max(np.abs(residual)) <= 1 << 24
    np.testing.assert_array_equal(actual[:, 1], residual.astype(np.float32).view(np.uint32))
