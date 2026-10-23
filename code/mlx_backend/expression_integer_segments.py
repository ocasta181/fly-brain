"""Selected Cython frontend lowering for integer RNG expression subtrees."""
import ast
import copy
import numpy as np


def render(renderer,node,*,ordered=False,force=False):
    if renderer.reference_target!='cython':return None
    integer_call=(isinstance(node,ast.Call)and isinstance(node.func,ast.Name)
        and node.func.id=='int'and len(node.args)==1 and not node.keywords)
    if not integer_call:
        if not isinstance(node,ast.BinOp)or not isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Mod,ast.FloorDiv)):return None
        if not force and not any(isinstance(part,ast.BinOp)and isinstance(part.op,(ast.Mod,ast.FloorDiv))for part in ast.walk(node)):return None
    inference_types=dict(renderer.types)
    from .operator_policy import compiled_integer_operand_dtype
    # Both operators retain the usual arithmetic operand type in Cython.
    # The actual operations and checked error edges come from its frontend.
    class OperandTypes(ast.NodeTransformer):
        def visit_Call(self,node):
            if isinstance(node.func,ast.Name)and node.func.id=='poisson':
                name='_integer_poisson_type'
                while name in inference_types:name+='_'
                inference_types[name]=np.dtype('int32')
                return ast.Name(id=name,ctx=ast.Load())
            return node
        def visit_BinOp(self,node):
            node=self.generic_visit(node)
            if isinstance(node.op,(ast.Mod,ast.FloorDiv)):node.op=ast.Add()
            return node
    typed=OperandTypes().visit(copy.deepcopy(node))
    try:dtype=compiled_integer_operand_dtype(typed,inference_types,target='cython',
        unsigned_char=renderer.compiled_unsigned_char)
    except KeyError:return None
    if dtype is None:return None
    from brian2.codegen.statements import Statement
    from .compiled_integer import lower
    from .integer_policy import snapshot_integer_policy
    from .compiler import _Expression,_metal_type,_TimedArray,UnsupportedModelError
    from .expression_errors import INTEGER_ERRORS
    if integer_call:
        class StatefulLeaves(ast.NodeTransformer):
            def capture(self,part):
                expression=renderer.expression(part)
                dtype={'float':np.dtype('float64'),'float32':np.dtype('float32'),
                    'bool':np.dtype('bool'),'int':np.dtype('int32')}[expression.kind]
                if expression.kind=='int':
                    inferred=compiled_integer_operand_dtype(part,inference_types,target='cython',
                        unsigned_char=renderer.compiled_unsigned_char)
                    if inferred is not None:dtype=inferred
                temporary=f'_rng_integer_operand_{renderer.temporary_index}'
                renderer.temporary_index+=1
                while temporary in renderer.types:temporary+='_'
                renderer.types[temporary]=dtype
                destination=renderer.lines if ordered else renderer.prelude
                destination.append(f'const {_metal_type(dtype)} {temporary}={expression.code};')
                return ast.copy_location(ast.Name(id=temporary,ctx=ast.Load()),part)
            def visit_Call(self,part):
                if isinstance(part.func,ast.Name)and(part.func.id in('rand','randn','poisson')
                        or isinstance(renderer.values.get(part.func.id),_TimedArray)):
                    return self.capture(part)
                return self.generic_visit(part)
            def visit_BinOp(self,part):
                if ordered and isinstance(part.op,ast.Div):
                    # A checked division must execute before a later draw.
                    return self.capture(part)
                return self.generic_visit(part)
        node=StatefulLeaves().visit(copy.deepcopy(node))
    elif any(isinstance(part,ast.Call)for part in ast.walk(node)):
        # Evaluate each side in CPU order. A checked integer prefix must
        # precede a later draw, and each side retains its original C width.
        operands=[]
        for operand in (node.left,node.right):
            try:literal=ast.literal_eval(operand)
            except (ValueError,TypeError):literal=None
            if type(literal)in(bool,int):
                # Keep literal divisors visible to the actual frontend.
                # Replacing them with inputs invents unreachable error paths.
                operands.append(operand)
                continue
            operand_typed=OperandTypes().visit(copy.deepcopy(operand))
            operand_dtype=compiled_integer_operand_dtype(operand_typed,inference_types,
                target='cython',unsigned_char=renderer.compiled_unsigned_char)
            expression=render(renderer,operand,ordered=ordered,force=True)
            if expression is None:expression=renderer.expression(operand)
            temporary=f'_rng_integer_operand_{renderer.temporary_index}'
            renderer.temporary_index+=1
            while temporary in renderer.types:temporary+='_'
            renderer.types[temporary]=operand_dtype
            destination=renderer.lines if ordered else renderer.prelude
            destination.append(f'const {_metal_type(operand_dtype)} {temporary}={expression.code};')
            operands.append(ast.Name(id=temporary,ctx=ast.Load()))
        node=ast.BinOp(left=operands[0],op=node.op,right=operands[1])
    name=f'_rng_integer_{renderer.temporary_index}'
    renderer.temporary_index+=1
    while name in renderer.types:name+='_'
    result_dtype=dtype
    types=dict(renderer.types);types[name]=result_dtype
    errors=[]
    statement=Statement(name,':=',ast.unparse(node),'',result_dtype,constant=False,scalar=False)
    code=lower(snapshot_integer_policy('cython'),[statement],types,renderer.values,{name},
        errors=errors,used_math=renderer.used_math,operator_policies=renderer.operator_policies,
        error_name=name+'_error',code_names=renderer.code_names)
    if not isinstance(code,str):raise UnsupportedModelError('Cython integer expression: '+code.diagnostics)
    lines=[f'{_metal_type(result_dtype)} {name}=0;',
        f'bool {name}_error=false;int {name}_error_kind=0;',code]
    if ordered:
        renderer.integer_may_raise=getattr(renderer,'integer_may_raise',False) or bool(errors)
        for index,error in enumerate(errors,1):
            if error not in INTEGER_ERRORS:raise UnsupportedModelError('Unregistered Cython RNG integer error: '+repr(error))
            lines.append(f'if({name}_error_kind=={index} && !_compiled_rng.error)_compiled_rng.error={INTEGER_ERRORS[error]|1}u;')
        renderer.lines.extend(lines)
    else:renderer.prelude.extend(lines)
    return _Expression(name,'int',integer_unsigned=result_dtype.kind=='u')
