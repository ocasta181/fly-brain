"""CPython3.10 complex arithmetic for dynamic pure-Python scalar provenance.

Adapted from CPython3.10.14 Objects/complexobject.c (_Py_c_prod/_Py_c_quot,
_Py_c_pow, c_powu/c_powi, complex_pow) and Include/pyport.h's ERANGE adjustment.
Changes: raw64 Metal carriers, existing target arithmetic/libm providers, and
explicit fault transport instead of host errno/Python object allocation. The
installed ARM64 instruction graph, including fused operations and paired
sincos, takes precedence over an uncontracted rendering of the C expressions.

Include binary64/math64 and the complex base before the returned header.
When power=True, also include complex basic/log, real exp/log, trigonometry64,
and operator64's MSL_POWER_HEADER.
These helpers never execute host arithmetic or participate in NumPy seterr.
The complete original CPython license is retained below.
"""

__all__ = ['metal_python_complex_header']

CPYTHON_SOURCE = 'https://github.com/python/cpython/blob/v3.10.14/Objects/complexobject.c'
CPYTHON_SOURCE_SHA256 = 'ece01bb8fa178413f23c2105aa23cee52af66ae921a610427a5d1807353afc3f'
CPYTHON_PYPORT_SHA256 = '7f8668d75ea83e4cbb3c187383bc032abb99a585eed5f67791a418b9c7065377'
CPYTHON_LIBPYTHON_SHA256 = '7c03e4c06b3f70fe1573ec5d9ce3fbc3210e9c4a533b570d900d140bd1105957'
ARM_FNMUL_REFERENCE = 'https://documentation-service.arm.com/static/6245c734b059dc5ff9a8bdab'
ARM_NAN_REFERENCE = 'https://documentation-service.arm.com/static/5fad51cbca04df4095c1c905'
# Source-only traversal of the installed macOS libm instruction graphs, including
# every direct branch/call target. No graph writes errno: no non-stack stores;
# only sincos calls four local helpers. This is not numerical qualification.
LIBM_ERRNO_GRAPH_INSTRUCTIONS = {
    'hypot': 41, 'pow': 268, 'atan2': 188,
    'exp': 69, 'log': 84, '__sincos_stret': 201,
}


_BASE_HEADER = r'''
#ifndef EON_PYTHON_COMPLEX_MATH64_INCLUDED
#define EON_PYTHON_COMPLEX_MATH64_INCLUDED
// ARM FMADD supplies the addend first to FPProcessNaNs3. Preserve that NaN
// selection around the existing finite arithmetic provider, whose API takes
// its multiplicands first. Default FPCR: signaling NaNs precede quiet NaNs.
// Arm DDI0568A.b H1.3 (FPProcessNaNs3), DDI0487 (FPMulAdd/FMADD).
inline ulong _eon_python_complex_fma(ulong a,ulong b,ulong c) {
    const ulong quiet=0x0008000000000000ul;
    bool an=d_is_nan(a),bn=d_is_nan(b),cn=d_is_nan(c);
    if(cn&&(c&quiet)&&
       ((d_is_zero(a)&&d_abs(b)==D_INF)||(d_abs(a)==D_INF&&d_is_zero(b))))
        return D_QNAN;
    if(cn&&!(c&quiet))return c|quiet;
    if(an&&!(a&quiet))return a|quiet;
    if(bn&&!(b&quiet))return b|quiet;
    if(cn)return c|quiet;
    if(an)return a|quiet;
    if(bn)return b|quiet;
    return d_fma(a,b,c);
}

// CPython3.10.14, installed ARM64 _Py_c_prod 0x89ac20 and _complex_mul 0x21f02c.
// Keep operand order: NumPy's scalar four-product graph is a different target.
inline _eon_complex _eon_python_complex_mul(_eon_complex a,_eon_complex b) {
    // FNMUL negates the rounded product, including the selected NaN sign.
    ulong cross_real=d_neg(d_mul(a.imag,b.imag));
    ulong cross_imag=d_mul(a.imag,b.real);
    return {_eon_python_complex_fma(a.real,b.real,cross_real),
            _eon_python_complex_fma(a.real,b.imag,cross_imag)};
}

// The C comparison-based absolute value preserves negative zero and NaN signs.
inline ulong _eon_python_complex_quot_abs(ulong value) {
    return d_lt(value,0ul)?d_neg(value):value;
}

// Fault1: complex division by zero. Source _Py_c_quot 0x89ac38 uses an actual
// divide for each numerator, unlike NumPy's reciprocal followed by multiply.
inline _eon_complex _eon_python_complex_div(_eon_complex a,_eon_complex b,
                                          thread uint& fault) {
    ulong ar=_eon_python_complex_quot_abs(b.real);
    ulong ai=_eon_python_complex_quot_abs(b.imag);
    if(d_ge(ar,ai)) {
        if(d_is_zero(ar)) {
            if(!fault)fault=1u;
            return {0ul,0ul};
        }
        ulong ratio=d_div(b.imag,b.real);
        ulong denom=_eon_python_complex_fma(b.imag,ratio,b.real);
        ulong real=d_div(_eon_python_complex_fma(a.imag,ratio,a.real),denom);
        ulong imag=d_div(_eon_python_complex_fma(d_neg(a.real),ratio,a.imag),denom);
        return {real,imag};
    }
    if(d_ge(ai,ar)) {
        ulong ratio=d_div(b.real,b.imag);
        ulong denom=_eon_python_complex_fma(b.real,ratio,b.imag);
        ulong real=d_div(_eon_python_complex_fma(a.real,ratio,a.imag),denom);
        ulong imag=d_div(_eon_python_complex_fma(a.imag,ratio,d_neg(a.real)),denom);
        return {real,imag};
    }
    return {D_QNAN,D_QNAN};
}
#endif // EON_PYTHON_COMPLEX_MATH64_INCLUDED
'''


_POWER_HEADER = r'''
#ifndef EON_PYTHON_COMPLEX_POWER64_INCLUDED
#define EON_PYTHON_COMPLEX_POWER64_INCLUDED
inline _eon_complex _eon_python_complex_powu(_eon_complex value,ulong exponent) {
    const _eon_complex one={0x3ff0000000000000ul,0ul};
    if(exponent==0ul)return one;
    // complex_pow's inlined first identity product is folded into these two
    // adds/multiplies. Returning value for exponent1 loses its signed-zero and
    // nonfinite behavior; a generic FMA identity product also changes this graph.
    _eon_complex first={d_add(value.real,d_mul(value.imag,0x8000000000000000ul)),
                        d_add(value.imag,d_mul(value.real,0ul))};
    _eon_complex result=(exponent&1ul)?first:one;
    if(exponent==1ul)return result;
    _eon_complex power=_eon_python_complex_mul(value,value);
    for(ulong mask=2ul;mask<=exponent;mask<<=1) {
        if(exponent&mask)result=_eon_python_complex_mul(result,power);
        // The installed wrapper drops the unused final square; it has no
        // errno or observable Python floating-policy action.
        if((mask<<1)<=exponent)power=_eon_python_complex_mul(power,power);
    }
    return result;
}

// The inlined c_powi inverse has folded quotient1 numerators. Preserve them
// rather than replacing this path with the general dynamic division helper.
inline _eon_complex _eon_python_complex_power_reciprocal(_eon_complex b,
                                                       thread uint& fault) {
    ulong ar=_eon_python_complex_quot_abs(b.real);
    ulong ai=_eon_python_complex_quot_abs(b.imag);
    if(d_ge(ar,ai)) {
        if(d_is_zero(ar)) {
            if(!fault)fault=2u;
            return {0ul,0ul};
        }
        ulong ratio=d_div(b.imag,b.real);
        ulong denom=_eon_python_complex_fma(b.imag,ratio,b.real);
        return {d_div(_eon_python_complex_fma(ratio,0ul,0x3ff0000000000000ul),denom),
                d_div(d_sub(0ul,ratio),denom)};
    }
    if(d_ge(ai,ar)) {
        ulong ratio=d_div(b.real,b.imag);
        ulong denom=_eon_python_complex_fma(b.real,ratio,b.imag);
        return {d_div(d_add(ratio,0ul),denom),
                d_div(_eon_python_complex_fma(ratio,0ul,0xbff0000000000000ul),denom)};
    }
    return {D_QNAN,D_QNAN};
}

// Fault2: zero to a negative or complex power. Fault3: complex exponentiation
// overflow. Pure Python arithmetic ignores NumPy floating actions.
inline _eon_complex _eon_python_complex_power(_eon_complex a,_eon_complex b,
                                            thread uint& fault) {
    _eon_complex result;
    if(d_is_zero(b.imag)&&d_eq(b.real,d_floor(b.real))&&
       d_le(d_abs(b.real),0x4059000000000000ul)) {
        // CPython's small-integer optimization is inclusive at both +/-100.
        long exponent=d_to_int(b.real);
        ulong magnitude=exponent<0?ulong(-exponent):ulong(exponent);
        result=_eon_python_complex_powu(a,magnitude);
        if(exponent<=0)result=_eon_python_complex_power_reciprocal(result,fault);
    } else if(d_is_zero(b.real)&&d_is_zero(b.imag)) {
        result={0x3ff0000000000000ul,0ul};
    } else if(d_is_zero(a.real)&&d_is_zero(a.imag)) {
        if((!d_is_zero(b.imag)||d_lt(b.real,0ul))&&!fault)fault=2u;
        result={0ul,0ul};
    } else {
        ulong magnitude=_eon_complex_hypot(a.real,a.imag);
        ulong length=d_operator_pow(magnitude,b.real);
        ulong angle=_eon_complex_atan2(a.imag,a.real);
        ulong phase=d_mul(b.real,angle);
        if(!d_is_zero(b.imag)) {
            length=d_div(length,d_exponential(d_mul(b.imag,angle)));
            phase=_eon_python_complex_fma(b.imag,d_logarithm(magnitude),phase);
        }
        // _Py_c_pow 0x89ad68 and complex_pow 0x21f43c call __sincos_stret.
        _eon_trig64_pair pair=_eon_trig64_sincos_scalar(phase);
        result={d_mul(length,pair.cosine),d_mul(length,pair.sine)};
    }
    // The pinned installed libm graphs never write errno. Together with the
    // explicit zero-domain fault above, this therefore implements the complete
    // target errno contract: Py_ADJUST_ERANGE2 forces ERANGE for final infinity,
    // including nonfinite input operands; finite/NaN pairs leave errno zero.
    if(!fault&&(d_abs(result.real)==D_INF||d_abs(result.imag)==D_INF))fault=3u;
    return result;
}
#endif // EON_PYTHON_COMPLEX_POWER64_INCLUDED
'''


def metal_python_complex_header(power=False):
    """Return source-bound Python-scalar helpers; caller owns error delivery."""
    return _BASE_HEADER + (_POWER_HEADER if power else '')


_CPYTHON_LICENSE = r"""
A. HISTORY OF THE SOFTWARE
==========================

Python was created in the early 1990s by Guido van Rossum at Stichting
Mathematisch Centrum (CWI, see https://www.cwi.nl) in the Netherlands
as a successor of a language called ABC.  Guido remains Python's
principal author, although it includes many contributions from others.

In 1995, Guido continued his work on Python at the Corporation for
National Research Initiatives (CNRI, see https://www.cnri.reston.va.us)
in Reston, Virginia where he released several versions of the
software.

In May 2000, Guido and the Python core development team moved to
BeOpen.com to form the BeOpen PythonLabs team.  In October of the same
year, the PythonLabs team moved to Digital Creations, which became
Zope Corporation.  In 2001, the Python Software Foundation (PSF, see
https://www.python.org/psf/) was formed, a non-profit organization
created specifically to own Python-related Intellectual Property.
Zope Corporation was a sponsoring member of the PSF.

All Python releases are Open Source (see https://opensource.org for
the Open Source Definition).  Historically, most, but not all, Python
releases have also been GPL-compatible; the table below summarizes
the various releases.

    Release         Derived     Year        Owner       GPL-
                    from                                compatible? (1)

    0.9.0 thru 1.2              1991-1995   CWI         yes
    1.3 thru 1.5.2  1.2         1995-1999   CNRI        yes
    1.6             1.5.2       2000        CNRI        no
    2.0             1.6         2000        BeOpen.com  no
    1.6.1           1.6         2001        CNRI        yes (2)
    2.1             2.0+1.6.1   2001        PSF         no
    2.0.1           2.0+1.6.1   2001        PSF         yes
    2.1.1           2.1+2.0.1   2001        PSF         yes
    2.1.2           2.1.1       2002        PSF         yes
    2.1.3           2.1.2       2002        PSF         yes
    2.2 and above   2.1.1       2001-now    PSF         yes

Footnotes:

(1) GPL-compatible doesn't mean that we're distributing Python under
    the GPL.  All Python licenses, unlike the GPL, let you distribute
    a modified version without making your changes open source.  The
    GPL-compatible licenses make it possible to combine Python with
    other software that is released under the GPL; the others don't.

(2) According to Richard Stallman, 1.6.1 is not GPL-compatible,
    because its license has a choice of law clause.  According to
    CNRI, however, Stallman's lawyer has told CNRI's lawyer that 1.6.1
    is "not incompatible" with the GPL.

Thanks to the many outside volunteers who have worked under Guido's
direction to make these releases possible.


B. TERMS AND CONDITIONS FOR ACCESSING OR OTHERWISE USING PYTHON
===============================================================

Python software and documentation are licensed under the
Python Software Foundation License Version 2.

Starting with Python 3.8.6, examples, recipes, and other code in
the documentation are dual licensed under the PSF License Version 2
and the Zero-Clause BSD license.

Some software incorporated into Python is under different licenses.
The licenses are listed with code falling under that license.


PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2
--------------------------------------------

1. This LICENSE AGREEMENT is between the Python Software Foundation
("PSF"), and the Individual or Organization ("Licensee") accessing and
otherwise using this software ("Python") in source or binary form and
its associated documentation.

2. Subject to the terms and conditions of this License Agreement, PSF hereby
grants Licensee a nonexclusive, royalty-free, world-wide license to reproduce,
analyze, test, perform and/or display publicly, prepare derivative works,
distribute, and otherwise use Python alone or in any derivative version,
provided, however, that PSF's License Agreement and PSF's notice of copyright,
i.e., "Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023 Python Software Foundation;
All Rights Reserved" are retained in Python alone or in any derivative version
prepared by Licensee.

3. In the event Licensee prepares a derivative work that is based on
or incorporates Python or any part thereof, and wants to make
the derivative work available to others as provided herein, then
Licensee hereby agrees to include in any such work a brief summary of
the changes made to Python.

4. PSF is making Python available to Licensee on an "AS IS"
basis.  PSF MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR
IMPLIED.  BY WAY OF EXAMPLE, BUT NOT LIMITATION, PSF MAKES NO AND
DISCLAIMS ANY REPRESENTATION OR WARRANTY OF MERCHANTABILITY OR FITNESS
FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF PYTHON WILL NOT
INFRINGE ANY THIRD PARTY RIGHTS.

5. PSF SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF PYTHON
FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR LOSS AS
A RESULT OF MODIFYING, DISTRIBUTING, OR OTHERWISE USING PYTHON,
OR ANY DERIVATIVE THEREOF, EVEN IF ADVISED OF THE POSSIBILITY THEREOF.

6. This License Agreement will automatically terminate upon a material
breach of its terms and conditions.

7. Nothing in this License Agreement shall be deemed to create any
relationship of agency, partnership, or joint venture between PSF and
Licensee.  This License Agreement does not grant permission to use PSF
trademarks or trade name in a trademark sense to endorse or promote
products or services of Licensee, or any third party.

8. By copying, installing or otherwise using Python, Licensee
agrees to be bound by the terms and conditions of this License
Agreement.


BEOPEN.COM LICENSE AGREEMENT FOR PYTHON 2.0
-------------------------------------------

BEOPEN PYTHON OPEN SOURCE LICENSE AGREEMENT VERSION 1

1. This LICENSE AGREEMENT is between BeOpen.com ("BeOpen"), having an
office at 160 Saratoga Avenue, Santa Clara, CA 95051, and the
Individual or Organization ("Licensee") accessing and otherwise using
this software in source or binary form and its associated
documentation ("the Software").

2. Subject to the terms and conditions of this BeOpen Python License
Agreement, BeOpen hereby grants Licensee a non-exclusive,
royalty-free, world-wide license to reproduce, analyze, test, perform
and/or display publicly, prepare derivative works, distribute, and
otherwise use the Software alone or in any derivative version,
provided, however, that the BeOpen Python License is retained in the
Software, alone or in any derivative version prepared by Licensee.

3. BeOpen is making the Software available to Licensee on an "AS IS"
basis.  BEOPEN MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR
IMPLIED.  BY WAY OF EXAMPLE, BUT NOT LIMITATION, BEOPEN MAKES NO AND
DISCLAIMS ANY REPRESENTATION OR WARRANTY OF MERCHANTABILITY OR FITNESS
FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF THE SOFTWARE WILL NOT
INFRINGE ANY THIRD PARTY RIGHTS.

4. BEOPEN SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF THE
SOFTWARE FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR LOSS
AS A RESULT OF USING, MODIFYING OR DISTRIBUTING THE SOFTWARE, OR ANY
DERIVATIVE THEREOF, EVEN IF ADVISED OF THE POSSIBILITY THEREOF.

5. This License Agreement will automatically terminate upon a material
breach of its terms and conditions.

6. This License Agreement shall be governed by and interpreted in all
respects by the law of the State of California, excluding conflict of
law provisions.  Nothing in this License Agreement shall be deemed to
create any relationship of agency, partnership, or joint venture
between BeOpen and Licensee.  This License Agreement does not grant
permission to use BeOpen trademarks or trade names in a trademark
sense to endorse or promote products or services of Licensee, or any
third party.  As an exception, the "BeOpen Python" logos available at
http://www.pythonlabs.com/logos.html may be used according to the
permissions granted on that web page.

7. By copying, installing or otherwise using the software, Licensee
agrees to be bound by the terms and conditions of this License
Agreement.


CNRI LICENSE AGREEMENT FOR PYTHON 1.6.1
---------------------------------------

1. This LICENSE AGREEMENT is between the Corporation for National
Research Initiatives, having an office at 1895 Preston White Drive,
Reston, VA 20191 ("CNRI"), and the Individual or Organization
("Licensee") accessing and otherwise using Python 1.6.1 software in
source or binary form and its associated documentation.

2. Subject to the terms and conditions of this License Agreement, CNRI
hereby grants Licensee a nonexclusive, royalty-free, world-wide
license to reproduce, analyze, test, perform and/or display publicly,
prepare derivative works, distribute, and otherwise use Python 1.6.1
alone or in any derivative version, provided, however, that CNRI's
License Agreement and CNRI's notice of copyright, i.e., "Copyright (c)
1995-2001 Corporation for National Research Initiatives; All Rights
Reserved" are retained in Python 1.6.1 alone or in any derivative
version prepared by Licensee.  Alternately, in lieu of CNRI's License
Agreement, Licensee may substitute the following text (omitting the
quotes): "Python 1.6.1 is made available subject to the terms and
conditions in CNRI's License Agreement.  This Agreement together with
Python 1.6.1 may be located on the internet using the following
unique, persistent identifier (known as a handle): 1895.22/1013.  This
Agreement may also be obtained from a proxy server on the internet
using the following URL: http://hdl.handle.net/1895.22/1013".

3. In the event Licensee prepares a derivative work that is based on
or incorporates Python 1.6.1 or any part thereof, and wants to make
the derivative work available to others as provided herein, then
Licensee hereby agrees to include in any such work a brief summary of
the changes made to Python 1.6.1.

4. CNRI is making Python 1.6.1 available to Licensee on an "AS IS"
basis.  CNRI MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR
IMPLIED.  BY WAY OF EXAMPLE, BUT NOT LIMITATION, CNRI MAKES NO AND
DISCLAIMS ANY REPRESENTATION OR WARRANTY OF MERCHANTABILITY OR FITNESS
FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF PYTHON 1.6.1 WILL NOT
INFRINGE ANY THIRD PARTY RIGHTS.

5. CNRI SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF PYTHON
1.6.1 FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR LOSS AS
A RESULT OF MODIFYING, DISTRIBUTING, OR OTHERWISE USING PYTHON 1.6.1,
OR ANY DERIVATIVE THEREOF, EVEN IF ADVISED OF THE POSSIBILITY THEREOF.

6. This License Agreement will automatically terminate upon a material
breach of its terms and conditions.

7. This License Agreement shall be governed by the federal
intellectual property law of the United States, including without
limitation the federal copyright law, and, to the extent such
U.S. federal law does not apply, by the law of the Commonwealth of
Virginia, excluding Virginia's conflict of law provisions.
Notwithstanding the foregoing, with regard to derivative works based
on Python 1.6.1 that incorporate non-separable material that was
previously distributed under the GNU General Public License (GPL), the
law of the Commonwealth of Virginia shall govern this License
Agreement only as to issues arising under or with respect to
Paragraphs 4, 5, and 7 of this License Agreement.  Nothing in this
License Agreement shall be deemed to create any relationship of
agency, partnership, or joint venture between CNRI and Licensee.  This
License Agreement does not grant permission to use CNRI trademarks or
trade name in a trademark sense to endorse or promote products or
services of Licensee, or any third party.

8. By clicking on the "ACCEPT" button where indicated, or by copying,
installing or otherwise using Python 1.6.1, Licensee agrees to be
bound by the terms and conditions of this License Agreement.

        ACCEPT


CWI LICENSE AGREEMENT FOR PYTHON 0.9.0 THROUGH 1.2
--------------------------------------------------

Copyright (c) 1991 - 1995, Stichting Mathematisch Centrum Amsterdam,
The Netherlands.  All rights reserved.

Permission to use, copy, modify, and distribute this software and its
documentation for any purpose and without fee is hereby granted,
provided that the above copyright notice appear in all copies and that
both that copyright notice and this permission notice appear in
supporting documentation, and that the name of Stichting Mathematisch
Centrum or CWI not be used in advertising or publicity pertaining to
distribution of the software without specific, written prior
permission.

STICHTING MATHEMATISCH CENTRUM DISCLAIMS ALL WARRANTIES WITH REGARD TO
THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS, IN NO EVENT SHALL STICHTING MATHEMATISCH CENTRUM BE LIABLE
FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

ZERO-CLAUSE BSD LICENSE FOR CODE IN THE PYTHON DOCUMENTATION
----------------------------------------------------------------------

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
"""
