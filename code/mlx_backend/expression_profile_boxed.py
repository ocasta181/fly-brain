"""Typed summaries of stock Cython boxed constant comparisons.

The stock floating producer remains the arithmetic oracle. Only the proved
Python comparison and reference-count region is replaced with its value.
"""
import math
import re
from .expression_profile_mir import Word,ProfileError,_constant
from .binary64 import literal


def float_integer_compare(real,integer,op):
    """Compare binary64 with an exact integer, including inexact boundaries."""
    try:boundary=float(integer)
    except OverflowError:boundary=math.inf if integer>0 else -math.inf
    if math.isinf(boundary):
        equal='false'
        less=('!d_eq('+real+','+literal(math.inf)+')')if integer>0 else('d_eq('+real+','+literal(-math.inf)+')')
        greater=('d_eq('+real+','+literal(math.inf)+')')if integer>0 else('!d_eq('+real+','+literal(-math.inf)+')')
        less='(!d_is_nan('+real+')&&('+less+'))';greater='(!d_is_nan('+real+')&&('+greater+'))'
    else:
        raw=literal(boundary);exact=boundary==integer
        equal='d_eq('+real+','+raw+')'if exact else'false'
        less='d_lt('+real+','+raw+')'
        greater='d_lt('+raw+','+real+')'
        if boundary<integer:less='('+less+'||d_eq('+real+','+raw+'))'
        elif boundary>integer:greater='('+greater+'||d_eq('+real+','+raw+'))'
    return {0:less,1:'('+less+'||'+equal+')',2:equal,3:'!('+equal+')',4:greater,5:'('+greater+'||'+equal+')'}[op]


def lower_large_literal(renderer,node):
    """Exact preliminary lowering; actual compiled caller binding remains required."""
    import ast
    from .compiler import _Expression
    if not isinstance(node,ast.Compare)or len(node.ops)!=1:return None
    def integer(part):
        if isinstance(part,ast.Constant)and type(part.value)is int:return part.value
        if isinstance(part,ast.UnaryOp)and isinstance(part.op,ast.USub):
            value=integer(part.operand)
            return -value if value is not None else None
        return None
    limit=integer(node.comparators[0]);reverse=False;producer=node.left
    if limit is None:
        limit=integer(node.left);reverse=True;producer=node.comparators[0]
    if limit is None or -(2**63)<=limit<2**63:return None
    real=renderer.expression(producer)
    if real.kind not in ('float','float32'):raise ProfileError('Large literal requires audited floating comparison')
    op=(ast.Lt,ast.LtE,ast.Eq,ast.NotEq,ast.Gt,ast.GtE).index(type(node.ops[0]))
    if reverse:op=(4,5,2,3,0,1)[op]
    return _Expression(float_integer_compare(renderer.convert(real,'float').code,limit,op),'bool')


def bind(g,text):
    if g.metadata['target']!='cython':return
    # LLVM memory operands retain the identical global address in both the
    # initialization store and immutable model load, independent of offsets.
    literals={}
    for match in re.finditer(r'(%\d+) = (?:tail )?call ptr @PyLong_From(?:LongLong|Long)\(i64 noundef (-?\d+)\)[^\n]*\n\s*store ptr \1, ptr (.+?), align ',text):
        literals[match[3]]=int(match[2])
    strings={match[1]:match[2]for match in re.finditer(
        r'(@[\w.]+) = private unnamed_addr constant \[\d+ x i8\] c"([-+0-9a-zA-Z]+)\\00"',text)}
    for match in re.finditer(r'(%\d+) = (?:tail )?call ptr @PyLong_FromString\(ptr noundef nonnull (@[\w.]+), ptr noundef nonnull %\d+, i32 noundef (\d+)\)[^\n]*\n\s*store ptr \1, ptr (.+?), align ',text):
        digits=strings.get(match[2]);base=int(match[3])
        if digits is not None and 2<=base<=36:
            try:literals[match[4]]=int(digits,base)
            except ValueError:raise ProfileError('Invalid boxed integer initializer')
    integers={}
    for reg,ins in g.nodes.items():
        if ins.op!='LDRXui':continue
        memory=re.search(r'from `ptr (.*?)`',ins.raw)
        if memory and memory[1]in literals:integers[reg]=literals[memory[1]]
    def uncopy(reg):
        seen=set()
        while reg not in seen and reg in g.nodes and g.nodes[reg].op=='COPY':
            seen.add(reg)
            following=g.nodes[reg].refs[0]
            if following.startswith('unknown:'):break
            reg=following
        return reg
    calls=[];boxes={};truths=[]
    for block,items in g.blocks.items():
        physical={}
        for index,ins in enumerate(items):
            if ins.op=='COPY'and ins.defs and ins.defs[0].startswith('$'):physical[ins.defs[0]]=ins.refs[0]
            if ins.op!='BL':continue
            name=re.match(r'@([^, ]+)',ins.args)
            name=name[1]if name else''
            if name not in('PyFloat_FromDouble','PyObject_RichCompare','PyObject_IsTrue'):continue
            abi='unknown:$w0'if name=='PyObject_IsTrue'else'unknown:$x0'
            result=next((x for x in items[index+1:]if x.op=='COPY'and x.refs==(abi,)),None)
            if result is None:raise ProfileError('Missing boxed comparison return')
            if name=='PyFloat_FromDouble':boxes[result.defs[0]]=(physical.get('$d0'),block)
            elif name=='PyObject_IsTrue':truths.append((result.defs[0],physical.get('$x0')))
            else:calls.append((result.defs[0],physical.copy(),ins))
    scopes=dict(re.findall(r'!(\d+) = distinct !DISubprogram\(name: "([^"\n]+)"',text))
    lexical=dict(re.findall(r'!(\d+) = (?:distinct )?!DILexicalBlock(?:File)?\(scope: !(\d+)',text))
    locations={identifier:re.search(r'\bscope: !(\d+)',body)[1]
        for identifier,body in re.findall(r'!(\d+) = (?:distinct )?!DILocation\(([^\n]+)\)',text)}
    g.caller_boxed_regions=set()
    for result,args,call in calls:
        left=uncopy(args.get('$x0',''));right=uncopy(args.get('$x1',''))
        reverse=left in integers and right in boxes
        if reverse:left,right=right,left
        if left not in boxes or right not in integers:continue
        operation=g.constant(args.get('$w2',''))
        if operation is None or operation.raw not in range(6):raise ProfileError('Unknown boxed comparison operation')
        op=operation.raw
        location=re.search(r'debug-location !(\d+)',call.raw)
        scope=locations.get(location[1])if location else None
        while scope in lexical:scope=lexical[scope]
        helper=scopes.get(scope,'')
        expected='__Pyx_PyObject_Compare'+('Lt','Le','Eq','Ne','Gt','Ge')[op]+('_int_float'if reverse else'_float_int')
        if helper!=expected:raise ProfileError('Unproved boxed comparison helper '+helper)
        producer,entry=boxes[left]
        value=g.symbolic(producer)
        if value is None or value.kind!='fp':raise ProfileError('Unknown boxed floating producer')
        predicate=float_integer_compare(value.code,integers[right],(4,5,2,3,0,1)[op]if reverse else op)
        def family(reg,seen=None):
            seen=set()if seen is None else seen
            if reg in seen:return None
            if reg==result:return {'result'}
            ins=g.nodes.get(reg)
            if ins is None:return None
            if ins.op=='LOADgot':
                if '@_Py_TrueStruct'in ins.raw:return {'true'}
                if '@_Py_FalseStruct'in ins.raw:return {'false'}
            if ins.op not in('COPY','PHI'):return None
            parts=[family(r,seen|{reg})for r in ins.refs]
            return set().union(*parts)if parts and all(p is not None for p in parts)else None
        candidates=[(reg,ins)for reg,ins in g.nodes.items()if ins.op=='PHI'and (family(reg)or set())=={'result','true','false'}]
        if len(candidates)!=1:raise ProfileError('Cannot prove one boxed comparison join')
        join,join_ins=candidates[0]
        predecessors={b:{p for p in g.region if b in g.successors[p]}for b in g.region}
        region=set();pending=list(predecessors[join_ins.block])
        while pending:
            block=pending.pop()
            if block in region:continue
            if block not in g.region:raise ProfileError('Boxed comparison escapes model loop')
            region.add(block)
            if block!=entry:pending.extend(predecessors[block])
        if entry not in region:raise ProfileError('Missing boxed comparison entry')
        for block in region:
            for ins in g.blocks[block]:
                if g.store_name(ins)in g.writes:raise ProfileError('Model store inside boxed comparison')
                if ins.op=='BL'and not re.match(r'@(?:PyFloat_FromDouble|PyObject_RichCompare),',ins.args):
                    raise ProfileError('Side effect inside boxed comparison '+ins.raw)
        g.caller_boxed_regions.update(region)
        g.bindings[join]=Word('int','(('+predicate+')?17L:16L)')
        # The native value is owned and non-null; refcounts only decide whether
        # CPython frees temporary storage, and cannot change this model value.
        objects={left,join}
        for register,operand in truths:
            if uncopy(operand)==join:g.bindings[register]=Word('int','long('+predicate+')',width=32)
        for reg,ins in g.nodes.items():
            if ins.op=='LOADgot':
                for name,number in(('_Py_TrueStruct',17),('_Py_FalseStruct',16),('_Py_NoneStruct',18)):
                    if '@'+name in ins.raw:g.bindings[reg]=_constant(number)
            if ins.op=='LDRXui'and ins.refs and uncopy(ins.refs[0])in objects and re.match(r'[^,]+, 0(?:,|$)',ins.args):
                g.bindings[reg]=_constant(2)
        g._constant_cache.clear();g._symbolic_cache.clear()
