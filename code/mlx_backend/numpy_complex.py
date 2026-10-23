"""NumPy complex temporaries without complex persistent Brian state.

Complex expressions carry both binary64 words. Destination stores still use
the original real/integer/Boolean state types. Immutable Python objects retain
the existing folding and scalar-store error path.
"""
import ast
from dataclasses import replace

import numpy as np

from .compiler import UnsupportedModelError, _Expression, _literal
from .numpy32 import Expression, Renderer as RealRenderer, _choose, _and, _scalar_boundary
from .numpy_half import Renderer as HalfRenderer


def is_complex(value):
    return value.kind == 'complex' or getattr(value,'value',None) is not None and np.iscomplexobj(value.value)


class ComplexRenderer:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.expression_fault_bits={}

    def _complex(self, value):
        self.used_math.add('_complex')
        if value.kind == 'complex':
            return value
        if value.value is not None and np.iscomplexobj(value.value):
            scalar = np.asarray(value.value)[()]
            code = ('_eon_complex{' + _literal(float(scalar.real), 'float') + ','
                    + _literal(float(scalar.imag), 'float') + '}')
            return replace(value, code=code, kind='complex',
                           narrow=np.asarray(scalar).dtype == np.dtype('complex64'))
        return Expression('_eon_complex{' + self.convert(value, 'float').code + ',0ul}',
                          'complex', value.scalar, getattr(value, 'narrow', False))

    def _complex_narrow(self, values):
        samples = []
        predicates = []
        has_array = any(not value.scalar or value.array_object for value in values)
        for value in values:
            if value.value is not None:
                samples.append(value.value)
            elif value.kind == 'complex':
                samples.append(np.zeros(() if value.scalar else 1,
                    np.complex64 if value.narrow is not False else np.complex128))
            elif value.kind == 'float':
                samples.append(np.zeros(() if value.scalar else 1,
                    np.float32 if value.narrow is not False else np.float64))
            else:
                dtype = np.dtype('bool') if value.kind == 'bool' else np.dtype(
                    ('uint' if value.integer_unsigned else 'int') + str(value.integer_bits))
                samples.append(np.zeros(() if value.scalar else 1, dtype))
            if value.value is not None:
                continue
            if not value.scalar and isinstance(value.narrow,str):
                predicates.append(value.narrow)
            if has_array and value.scalar:
                if value.kind in ('complex','float') and value.narrow is not True:
                    parts = ([f'({value.code}).real',f'({value.code}).imag'] if value.kind == 'complex'
                             else [value.code])
                    fits = _and([f'(d_le(d_abs({part}),{_scalar_boundary()})||d_exp({part})==2047ul)'
                                 for part in parts])
                    predicates.append(_choose(value.narrow,'true',fits))
                elif value.kind == 'int' and value.integer_bits>16:
                    predicates.append(f'(({value.code})>=-32768L&&({value.code})<=65535L)')
        return _and([np.result_type(*samples) == np.dtype('complex64'), *predicates])

    def _complex_round(self, code, narrow):
        return _choose(narrow, f'_eon_complex_round32({code})', code)

    def _complex_int32_arguments(self,value):
        rank,count=self._array_indices(scalar=value.scalar)
        return rank,count,_choose(value.narrow,'32u','64u')

    def _binary(self, left, right, operation):
        if not any(is_complex(value) for value in (left, right)):
            return super()._binary(left, right, operation)
        runtime_numpy=(getattr(self,'numpy_operation_boundaries',False) or
            any(action!='ignore' for action in getattr(self,'numpy_error_actions',{}).values()))
        if (all(value.scalar and value.value is not None for value in (left, right))
                and (not runtime_numpy or all(value.python_scalar for value in (left,right)))):
            return super()._binary(left, right, operation)
        # NumPy float64 inherits Python float. A Python complex left operand
        # accepts it directly; the reverse order dispatches through NumPy.
        python_scalar=(left.scalar and right.scalar and (
            left.python_scalar and right.python_scalar or
            left.python_scalar and is_complex(left) and not right.array_object
            and right.kind=='float' and not is_complex(right) and right.narrow is False
            and getattr(right,'half',False) is False))
        if python_scalar and isinstance(operation,ast.Mult):
            a,b=(self._complex(value) for value in (left,right))
            self.used_math.add('_python_complex')
            return Expression(f'_eon_python_complex_mul({a.code},{b.code})',
                              'complex',True,False,python_scalar=True)
        if python_scalar and isinstance(operation,(ast.Div,ast.Pow)):
            a,b=(self._complex(value) for value in (left,right))
            name='div' if isinstance(operation,ast.Div) else 'power'
            result=f'_eon_python_complex_{self.temporary_index}'
            self.temporary_index+=1
            status=result+'_fault'
            self.used_math.add('_python_complex')
            if name=='power':
                self.used_math.update(('_python_complex_power','_complex_power','_complex_log',
                    '_complex_basic','_complex_trig','operator_pow_float','log','log1p',
                    'log_f32','log1p_f32','sin','cos','exp','exp_f32','expm1','_half_f32'))
            self.prelude.extend((f'uint {status}=0u;',
                f'const _eon_complex {result}=_eon_python_complex_{name}({a.code},{b.code},{status});'))
            from .expression_errors import PYTHON_COMPLEX_ERRORS
            choices=(1,) if name=='div' else (2,3)
            for fault in choices:
                bits,error=PYTHON_COMPLEX_ERRORS[fault]
                self.runtime_errors.append(error)
                kind=len(self.runtime_errors)
                self.expression_fault_bits[kind]=bits
                flag=self.index_error_name
                self.prelude.append(f'if(!{flag}&&{status}=={fault}u)'
                    f'{{{flag}_kind={kind};{flag}=true;}}')
            return Expression(result,'complex',True,False,python_scalar=True)
        if isinstance(operation, ast.Pow) and right.value is not None:
            exponent = np.asarray(right.value)[()]
            if np.isreal(exponent) and float(np.real(exponent)).is_integer() and abs(exponent) < 100:
                narrow = self._complex_narrow((left, right))
                value = self._complex(left)
                arguments = f'({value.code},{int(np.real(exponent))}L)'
                code = _choose(narrow,'_eon_complex_powi32'+arguments,'_eon_complex_powi'+arguments)
                return Expression(code, 'complex', left.scalar and right.scalar, narrow)
        if isinstance(operation,ast.Pow):
            narrow=self._complex_narrow((left,right))
            a,b=(self._complex(value) for value in (left,right))
            self.used_math.update(('_complex_power','_complex_log','_complex_basic','_complex_trig',
                'log','log1p','log_f32','log1p_f32','sin','cos','exp','exp_f32','expm1','_half_f32'))
            code=_choose(narrow,f'_eon_complex_power32(_eon_complex_round32({a.code}),_eon_complex_round32({b.code}))',
                         f'_eon_complex_power({a.code},{b.code})')
            return Expression(self._complex_round(code,narrow),'complex',left.scalar and right.scalar,narrow)
        names = {ast.Add: 'add', ast.Sub: 'sub', ast.Mult: 'mul', ast.Div: 'div'}
        if type(operation) not in names:
            raise UnsupportedModelError('Native complex operation needs ' + type(operation).__name__)
        narrow = self._complex_narrow((left, right))
        a, b = (self._complex(value) for value in (left, right))
        helper = '_eon_complex_' + names[type(operation)]
        if isinstance(operation, ast.Mult) and any(
                not value.scalar or value.array_object for value in (left, right)):
            helper = '_eon_complex_multiply_numpy'
        narrow_code = (helper+'32(_eon_complex_round32('+a.code+'),'
                       '_eon_complex_round32('+b.code+'))')
        code = _choose(narrow,narrow_code,f'{helper}({a.code},{b.code})')
        return Expression(self._complex_round(code, narrow), 'complex',
                          left.scalar and right.scalar, narrow,python_scalar=python_scalar)

    def convert(self, value, kind):
        if value.kind == 'complex':
            if kind == 'complex':
                return value
            if kind == 'bool':
                return _Expression(f'_eon_complex_truth({value.code})', 'bool')
            value = replace(value, code=f'({value.code}).real', kind='float', value=None)
        return super().convert(value, kind)

    def _storage_type(self, value):
        return '_eon_complex' if value.kind == 'complex' else super()._storage_type(value)

    def _storage_zero(self, value):
        return '_eon_complex{0ul,0ul}' if value.kind == 'complex' else super()._storage_zero(value)

    def _storage_encode(self, value):
        return value.code if value.kind == 'complex' else super()._storage_encode(value)

    def _storage_expression(self, local, value):
        if value.kind == 'complex':
            return replace(value, code=local, value=None)
        return super()._storage_expression(local, value)

    def _temporary_type(self, value):
        return '_eon_complex' if value.kind == 'complex' else super()._temporary_type(value)

    def _requires_typed_storage(self,value):
        return value.kind=='complex' or super()._requires_typed_storage(value)

    def _inplace_value(self, previous, value):
        if previous.kind == 'complex':
            value = self._complex(value)
            return replace(value, code=self._complex_round(value.code, previous.narrow),
                           scalar=previous.scalar, narrow=previous.narrow,
                           array_object=previous.array_object, value=None)
        return super()._inplace_value(previous, value)

    def finish(self):
        originals = {}
        casts={}
        self.complex_store_sources={}
        python_casts={}
        for name, value in self.temporaries.items():
            if name not in self.persistent or value.kind != 'complex':
                continue
            originals[name] = value
            self.complex_store_sources[name]=value
            if np.dtype(self.types[name]).kind == 'b':
                self.temporaries[name] = replace(value,
                    code=f'_eon_complex_truth({value.code})', kind='bool', value=None)
            else:
                if value.python_scalar:python_casts[name]=np.dtype(self.types[name]).kind
                else:casts[name]=value
                self.temporaries[name] = replace(value,
                    code=f'({value.code}).real', kind='float', value=None,python_scalar=False)
        try:
            lines=super().finish()
            from .compiler import _metal_type
            for index,(name,expression,dtype) in self.finish_stores.items():
                if name not in originals or dtype.kind not in 'iu':continue
                source=originals[name]
                rank,count,width=self._complex_int32_arguments(source)
                unsigned=str(dtype.kind=='u').lower()
                code=(f'_eon_complex_integer_cast(({source.code}).real,{rank},{count},'
                      f'{width},{dtype.itemsize*8}u,{unsigned})')
                lines[index]=f'{self.code_name(name)} = ({_metal_type(dtype)})({code});'
            self.complex_cast_stores={index:name for index,line in enumerate(lines)
                for name in casts if f'{self.code_name(name)} = ' in line}
            self.python_complex_cast_stores={index:(name,python_casts[name])
                for index,line in enumerate(lines) for name in python_casts
                if f'{self.code_name(name)} = ' in line}
            if self.complex_cast_stores or self.python_complex_cast_stores:self.used_math.add('_complex_cast')
            return lines
        finally:
            self.temporaries.update(originals)

    def _select(self, condition, left, right):
        if not any(is_complex(value) for value in (left, right)):
            return super()._select(condition, left, right)
        narrow = self._complex_narrow((left, right))
        a, b = (self._complex(value) for value in (left, right))
        code = f'({self.convert(condition,"bool").code}?{a.code}:{b.code})'
        return Expression(self._complex_round(code, narrow), 'complex',
                          condition.scalar and left.scalar and right.scalar, narrow)

    def expression(self, source):
        if isinstance(source, Expression):
            return source
        node = ast.parse(source, mode='eval').body if isinstance(source, str) else source
        # Preserve complete immutable folding, including Python-vs-NumPy
        # scalar provenance and its destination-specific TypeError behavior.
        try:
            invariant = self.evaluate(node)
        except KeyError:
            pass
        else:
            if np.iscomplexobj(invariant):
                return self._constant_expression(invariant)
            return super().expression(node)
        if isinstance(node,ast.BinOp):
            left,right=(self.expression(part) for part in (node.left,node.right))
            if is_complex(left) or is_complex(right):
                # Half's registered real-power adapter must not project a
                # dynamic complex operand before complex lowering sees it.
                return self._binary(left,right,node.op)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self.expression(node.operand)
            if is_complex(value):
                value = self._complex(value)
                return value if isinstance(node.op, ast.UAdd) else replace(
                    value, code=f'_eon_complex_neg({value.code})', value=None)
        if isinstance(node, ast.Compare):
            values = [self.expression(part) for part in (node.left, *node.comparators)]
            if any(is_complex(value) for value in values):
                values = [self._complex(value) for value in values]
                predicates = []
                names = {ast.Eq: 'eq', ast.NotEq: 'ne', ast.Lt: 'lt',
                         ast.LtE: 'le', ast.Gt: 'gt', ast.GtE: 'ge'}
                for operation, a, b in zip(node.ops, values, values[1:]):
                    predicates.append(f'_eon_complex_{names[type(operation)]}({a.code},{b.code})')
                return Expression('(' + '&&'.join(predicates) + ')', 'bool',
                                  all(value.scalar for value in values))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            arguments = [self.expression(part) for part in node.args]
            if any(is_complex(value) for value in arguments):
                name = node.func.id
                if name == 'clip' and len(arguments) == 3:
                    narrow = self._complex_narrow(arguments)
                    values = [self._complex(value) for value in arguments]
                    code = '_eon_complex_clip(' + ','.join(value.code for value in values) + ')'
                    return Expression(self._complex_round(code,narrow), 'complex',
                                      all(value.scalar for value in values), narrow)
                if name == 'timestep' and len(arguments) == 2:
                    self.used_math.add('_complex_cast')
                    offset = self._binary(self.expression('.001'), arguments[1], ast.Mult())
                    result = self._binary(self._binary(arguments[0],offset,ast.Add()),arguments[1],ast.Div())
                    scalar=all(value.scalar for value in arguments)
                    return Expression(self.convert(result,'int').code,'int',scalar,python_scalar=scalar)
                if len(arguments) == 1:
                    value = self._complex(arguments[0])
                    if name == 'int':
                        self.used_math.add('_complex_cast')
                        rank,count,width=self._complex_int32_arguments(value)
                        code=f'_eon_complex_int32(({value.code}).real,{rank},{count},{width})'
                        return Expression(code,'int',value.scalar,integer_bits=32)
                    if name in ('floor','ceil'):
                        self.used_math.add('_complex_cast')
                        from .expression_errors import COMPLEX_UFUNC_ERRORS
                        bits,error=COMPLEX_UFUNC_ERRORS[name]
                        self.runtime_errors.append(error)
                        self.expression_fault_bits[len(self.runtime_errors)]=bits
                        flag = self.index_error_name
                        self.prelude.append(f'if(!{flag}){{{flag}_kind={len(self.runtime_errors)};{flag}=true;}}')
                        return Expression('0L','int',value.scalar,integer_bits=32)
                    if name == 'sign':
                        return Expression(f'_eon_complex_sign({value.code})','complex',value.scalar,value.narrow)
                    if name == 'exprel':
                        numerator=self.expression(ast.Call(ast.Name('expm1',ast.Load()),[node.args[0]],[]))
                        ratio=self._binary(numerator,value,ast.Div())
                        self.used_math.update(('_complex_basic','_complex_trig','sin','cos','expm1','exp','exp_f32','_half_f32'))
                        absolute=_choose(value.narrow,f'_eon_complex_abs_numpy32({value.code})',
                                         f'_eon_complex_abs_numpy({value.code})')
                        small=f'd_lt({absolute},{_literal(1e-16,"float")})'
                        big=f'_eon_complex_gt({value.code},_eon_complex{{{_literal(717.,"float")},0ul}})'
                        code=f'({small}?_eon_complex{{0x3ff0000000000000ul,0ul}}:({big}?_eon_complex{{D_INF,0ul}}:{ratio.code}))'
                        return replace(ratio,code=code,value=None,
                                       array_object=self.wrapper_arrays.get('exprel',False))
                    if name in ('sin','cos','sinh','cosh'):
                        self.used_math.update(('_complex_trig', 'sin', 'cos', 'expm1'))
                    elif name in ('exp','sqrt','abs'):
                        self.used_math.update(('_complex_basic','_complex_trig','sin','cos','expm1','exp','exp_f32','_half_f32'))
                    elif name in ('expm1','tanh','tan'):
                        self.used_math.update(('_complex_extended','_complex_trig','sin','cos','tan',
                            'sin_f32','cos_f32','tan_f32','expm1','expm1_f32','exp','exp_f32','_half_f32'))
                    elif name in ('log','log1p','log10'):
                        self.used_math.update(('_complex_log','log','log1p','log_f32','log1p_f32'))
                    elif name in ('arctan','arcsin','arccos'):
                        self.used_math.update(('_complex_inverse','_complex_log','_complex_trig',
                            'log','log1p','log_f32','log1p_f32','arctan','sin','cos','expm1'))
                    else:
                        raise UnsupportedModelError('Native complex function needs ' + name)
                    if name == 'abs':
                        code = _choose(value.narrow,f'_eon_complex_abs_numpy32({value.code})',
                                       f'_eon_complex_abs_numpy({value.code})')
                        return Expression(code,
                                          'float',value.scalar,value.narrow)
                    code = _choose(value.narrow,f'_eon_complex_{name}32({value.code})',
                                   f'_eon_complex_{name}({value.code})')
                    return Expression(self._complex_round(code, value.narrow),
                                      'complex', value.scalar, value.narrow)
                raise UnsupportedModelError('Native complex function needs ' + name)
        return super().expression(node)


class Renderer(ComplexRenderer, HalfRenderer):
    pass


class BasicRenderer(ComplexRenderer, RealRenderer):
    pass
