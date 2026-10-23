"""Lower the pointwise arithmetic of an actual stock-Clang Brian state loop.

MIR is consumed as compiler data. Python binding/setup code is not executed on
Metal; typed model inputs replace its validated arrays and immutable scalars.
The native operation order, fused instructions and per-cell branches remain.
"""
from dataclasses import dataclass
import ctypes,re
import numpy as np

_MASK=(1<<64)-1
_LIB=ctypes.CDLL(None)


class ProfileError(ValueError):pass


@dataclass(frozen=True)
class Word:
    kind:str
    code:str
    raw:int|None=None
    width:int=64


def _constant(raw,kind='int',width=64):
    raw=int(raw)&((1<<width)-1)
    return Word(kind,(str(raw)+('u'if width==32 else'ul')if kind=='fp'else str(raw if raw<1<<(width-1)else raw-(1<<width))+'L'),raw,width)


def _number(word):return float(np.asarray(word.raw,np.uint32 if word.width==32 else np.uint64).view(np.float32 if word.width==32 else np.float64))
def _bits(value):return int(np.asarray(value,np.float64).view(np.uint64))


def _integer_float32_bits(number):
    """Round an integer directly to binary32, without a binary64 midpoint."""
    sign=int(number<0)<<31
    magnitude=abs(number)
    if not magnitude:return sign
    exponent=magnitude.bit_length()-1
    shift=exponent-23
    if shift>0:
        significand,remainder=divmod(magnitude,1<<shift)
        halfway=1<<(shift-1)
        significand+=remainder>halfway or (remainder==halfway and significand&1)
        if significand==1<<24:
            significand>>=1;exponent+=1
    else:significand=magnitude<<(-shift)
    return sign|((exponent+127)<<23)|(significand&0x7fffff)


def _call(name,values):
    if all(v.raw is not None for v in values):
        function=getattr(_LIB,name);function.argtypes=[ctypes.c_double]*len(values);function.restype=ctypes.c_double
        return _constant(_bits(function(*(_number(v)for v in values))),'fp')
    helpers={'sqrt':'d_sqrt','fabs':'d_abs','log':'expr_apple_log','exp':'expr_apple_exp','fma':'expr_fma'}
    if name not in helpers:raise ProfileError('Dynamic library operation '+name)
    return Word('fp',helpers[name]+'('+','.join(v.code for v in values)+')')


def _fp(op,values):
    if op=='FNMULDrr':return _fp('FNEGDr',[_fp('FMULDrr',values)])
    if op=='FNEGDr':
        a=values[0];return _constant(a.raw^(1<<63),'fp')if a.raw is not None else Word('fp','d_neg('+a.code+')')
    if all(v.raw is not None for v in values):
        a=[np.float64(_number(v))for v in values]
        with np.errstate(all='ignore'):
            result={'FMULDrr':lambda:a[0]*a[1],'FADDDrr':lambda:a[0]+a[1],
                'FSUBDrr':lambda:a[0]-a[1],'FDIVDrr':lambda:a[0]/a[1]}.get(op)
            if result is not None:return _constant(_bits(result()),'fp')
    if op in('FMADDDrrr','FMSUBDrrr','FNMADDDrrr','FNMSUBDrrr'):
        a,b,c=values
        if op in('FMSUBDrrr','FNMADDDrrr'):a=_fp('FNEGDr',[a])
        if op in('FNMSUBDrrr','FNMADDDrrr'):c=_fp('FNEGDr',[c])
        return _call('fma',[a,b,c])
    function={'FMULDrr':'d_mul','FADDDrr':'d_add','FSUBDrr':'d_sub','FDIVDrr':'d_div'}[op]
    return Word('fp',function+'('+','.join(v.code for v in values)+')')


@dataclass
class Instruction:
    op:str
    defs:tuple
    refs:tuple
    args:str
    block:int
    raw:str
    types:tuple=()


class Graph:
    def __init__(self,text,metadata,fixed_values=None,region=None):
        self.metadata=metadata;self.arrays=metadata['arrays'];self.constants=metadata['constants']
        self.namespace_arrays=metadata.get('namespace_arrays',{})
        self.namespace_buffers={v['name'] for v in self.namespace_arrays.values()}
        # Clang can merge the immutable namespace pointers into one global.
        # Debug expressions retain the exact named slot and byte offset.
        globals_by_id=dict(re.findall(r'!(\d+) = distinct !DIGlobalVariable\(name: "([^"\n]+)"',text))
        expressions={key:(variable,int(offset or 0)) for key,variable,offset in re.findall(
            r'!(\d+) = !DIGlobalVariableExpression\(var: !(\d+), expr: !DIExpression\((?:DW_OP_plus_uconst, (\d+))?\)\)',text)}
        self.namespace_slots={}
        for line in text.splitlines():
            match=re.match(r'\s*@([^ ]+) =.*\bglobal\b',line)
            if match is None:continue
            for identifier in re.findall(r'!dbg !(\d+)',line):
                variable,offset=expressions.get(identifier,(None,0))
                original=globals_by_id.get(variable,'')
                for suffix,info in self.namespace_arrays.items():
                    if original.endswith(suffix):self.namespace_slots[(match[1],offset)]=info['name']
        self.fixed_values=fixed_values or{}
        self.writes=set(metadata['writes']);self.nodes={};self.blocks={};self.bindings={};self.physical={};self.calls={}
        main_scopes=set(re.findall(r'!(\d+) = distinct !DISubprogram\(name: "__pyx_pf_[^"\n]*_main"',text))
        debug={identifier:name for identifier,name,scope in re.findall(
            r'!(\d+) = !DILocalVariable\(name: "([^"\n]+)"[^\n]*?scope: !(\d+)',text)if scope in main_scopes}
        documents=[doc for doc in text.split('\n---\n')if'\nbody:'in doc]
        if metadata['target']=='cpp':documents=[doc for doc in documents if metadata.get('function','stateupdater_codeobject')in doc.split('body:',1)[0]]
        else:
            # Larger callers can remain outlined behind Cython's Python
            # argument wrapper. Debug linkage identifies the actual body.
            bodies=set(re.findall(r'!DISubprogram\(name: "__pyx_pf_[^"\n]*_main", linkageName: "([^"\n]+)"',text))
            outlined=[doc for doc in documents if re.search(r'^name:\s+(\S+)',doc)[1]in bodies]
            documents=outlined or[doc for doc in documents if'1main'in doc.split('body:',1)[0]and'.cold.'not in doc.split('body:',1)[0]]
        if len(documents)!=1:raise ProfileError('Cannot identify one state loop')
        block=None
        for line in documents[0].split('body:',1)[1].splitlines():
            match=re.match(r'  bb\.(\d+) ',line)
            if match:block=int(match[1]);self.blocks[block]=[];continue
            s=line.strip()
            if block is None or not s or s=='...'or s.startswith(('successors:','liveins:','DBG_LABEL','DBG_VALUE_LIST')):continue
            dbg=re.match(r'DBG_VALUE (%\d+), \$noreg, !(\d+), !DIExpression\(\)',s)
            if dbg:
                reg,identifier=dbg.groups();name=debug.get(identifier,'')
                if name.startswith('__pyx_v_'):
                    name=name[len('__pyx_v_'):]
                    if name in self.arrays:self.bindings[reg]=('ptr',self.arrays[name]['name'])
                    elif name in self.constants:
                        c=self.constants[name];self.bindings[reg]=_constant(c['raw'],c['kind'])
                continue
            if s.startswith('DBG_'):continue
            main=s.split(' :: ',1)[0].split(',  debug-location',1)[0]
            if' = 'in main:
                left,right=main.split(' = ',1)
                defs=tuple(re.findall(r'%\d+|\$[a-z0-9]+',left));types=tuple(re.findall(r':(\w+)',left))
            else:defs=();types=();right=main
            right=re.sub(r'\b(?:nsz|arcp|contract|afn|reassoc|nofpexcept|nuw|nsw)\b\s*','',right)
            op,_,args=right.partition(' ');refs=tuple(re.findall(r'%\d+|\$[a-z0-9]+',args.split('implicit')[0]))
            if op=='COPY'and refs and refs[0].startswith('$'):
                refs=(self.physical.get(refs[0],refs[0]),)
            ins=Instruction(op,defs,refs,args,block,s,types);self.blocks[block].append(ins)
            if op=='BL':
                function=re.match(r'[&@]([^, ]+)',args)
                name=function[1]if function else''
                if name in('exp','expm1','log','log1p','sqrt','fabs','sin','cos','tan','fma','pow'):
                    count=3 if name=='fma'else 2 if name=='pow'else 1
                    call=Instruction('call:'+name,(),tuple(self.physical.get('$d'+str(i),'$d'+str(i))for i in range(count)),'',block,s)
                    key='call'+str(len(self.nodes));self.nodes[key]=call;self.physical['$d0']=key;self.calls[id(ins)]=key
                elif name=='PyErr_Occurred':
                    key='call'+str(len(self.nodes));self.nodes[key]=Instruction('constant-zero',(),(),' ',block,s)
                    self.physical['$x0']=key;self.physical['$w0']=key
                else:
                    for reg in('$d0','$x0','$w0'):self.physical[reg]='unknown:'+reg
                continue
            for index,dest in enumerate(defs):
                if dest.startswith('$'):
                    if op=='COPY':self.physical[dest]=refs[0]
                    continue
                instruction=ins
                if op.startswith('LDR')and len(defs)==2:
                    instruction=Instruction('post-address'if index==0 else'load-value',(dest,),refs,args,block,s,(types[index],))
                self.nodes[dest]=instruction
        self.array_info={v['name']:v for v in self.arrays.values()}
        self.initial={n:('_profile_input_'+n)for n in self.array_info}
        self._roots={};self._constant_cache={};self._constant_active=set()
        self._symbolic_cache={};self._symbolic_active=set()
        self.successors={block:self.edges(block)for block in self.blocks}
        self.reachable=set();pending=[min(self.blocks)]
        while pending:
            block=pending.pop()
            if block in self.reachable:continue
            self.reachable.add(block);pending.extend(self.successors[block])
        store_blocks={b for b,items in self.blocks.items()if any(self.store_name(ins)in self.writes for ins in items)}
        regions=[part for part in self.components()if len(part)>1 or any(b in self.successors[b]for b in part)]
        regions=[part for part in regions if part&store_blocks]
        self.other_region=None;self.bulk_count=None
        if len(regions)>1:
            vector=[part for part in regions if any(ins.op.startswith(('FMULv','FMLAv','FADDv','FSUBv'))
                for b in part for ins in self.blocks[b])]
            if len(regions)!=2 or len(vector)!=1:raise ProfileError('Cannot separate one pointwise SIMD loop and scalar tail')
            bulk=vector[0];tail=next(part for part in regions if part!=bulk)
            starts=[]
            for reg,ins in self.nodes.items():
                if ins.op!='PHI'or ins.block in bulk|tail or self.root(reg)is not None:continue
                incoming=[r for r,b in re.findall(r'(%\d+), %bb\.(\d+)',ins.args)if int(b)in bulk]
                if not incoming:continue
                values=[self.constant(r)for r in incoming]
                if not all(v is not None and v.kind=='int'and v.raw is not None and 0<v.raw<=metadata['neurons']for v in values):continue
                if len({v.raw for v in values})!=1:continue
                starts.append(values[0].raw);self.bindings[reg]=values[0]
            if not starts:
                # With an exact SIMD multiple, a separate scalar loop can be
                # only the overlap fallback. Prove there is no bulk-to-scalar
                # path before discarding it for Brian's disjoint arrays.
                pending=[t for b in bulk for t in self.successors[b]if t not in bulk];seen=set()
                while pending:
                    b=pending.pop()
                    if b in seen:continue
                    seen.add(b);pending.extend(self.successors[b]-bulk-seen)
                if seen&tail:raise ProfileError('Cannot prove the scalar-tail start from the SIMD exit')
                self.bulk_count=metadata['neurons']
                if region is None:region=bulk
            else:
                if len(set(starts))!=1:raise ProfileError('Scalar-tail starts disagree')
                self._constant_cache.clear();self.bulk_count=starts[0]
                if region is None:self.other_region=tail;region=bulk
        self.unrolled=not regions;self.region=self.reachable if self.unrolled else(region if region is not None else regions[0])
        headers=[b for b in self.region if any(b in edges and p in self.reachable and p not in self.region for p,edges in self.successors.items())]
        self.header=min(self.region)if self.unrolled else headers[0]if len(headers)==1 else None
        if self.header is None:raise ProfileError('Multiple arithmetic loop entries')
        self.latches={b for b in self.region if self.header in self.successors[b]}
        self.normal_exits={t for b in self.latches for t in self.successors[b]if t not in self.region}
        self.values={};self.declarations={};self.prelude=[];self.may_raise=False;self.flags=('false',)*4

    def root(self,reg,visiting=None):
        if reg in self.bindings and isinstance(self.bindings[reg],tuple):return self.bindings[reg][1]
        if reg in self._roots:return self._roots[reg]
        visiting=set()if visiting is None else visiting
        if reg in visiting:return None
        visiting=visiting|{reg};ins=self.nodes.get(reg)
        if ins is None:return None
        if ins.op in('LOADgot','LDRXui'):
            for suffix,info in self.namespace_arrays.items():
                if re.search(re.escape(suffix)+r'\b',ins.raw):return info['name']
            symbol=re.search(r'@([^ ,]+)(?: \+ (\d+))?',ins.raw)
            if symbol:
                namespace=self.namespace_slots.get((symbol[1],int(symbol[2]or 0)))
                if namespace is not None:return namespace
            for array,v in self.arrays.items():
                if array+'E'in ins.raw or array+'\b'in ins.raw:return v['name']
        if ins.op in('COPY','PHI','post-address')or ins.op.startswith(('ADDX','SUBX')):
            roots={self.root(r,visiting)for r in ins.refs if isinstance(r,str)and r.startswith('%')}
            roots.discard(None)
            if len(roots)==1:
                result=roots.pop();self._roots[reg]=result;return result
        if ins.op.startswith('STR')and ins.refs:return self.root(ins.refs[1],visiting)
        return None

    def store_name(self,ins):
        return self.root(ins.refs[1])if ins.op.startswith('STR')and len(ins.refs)>1 else None

    def field(self,name):
        if name in self.fixed_values:return _constant(self.fixed_values[name],'fp')
        if name in self.constants:
            c=self.constants[name];return _constant(c['raw'],c['kind'])
        v=self.array_info[name];kind='fp'if v['dtype']in('float32','float64')else'int'
        return Word(kind,self.initial[name],width=32 if v['dtype']in('float32','int32','bool')else 64)

    def constant(self,reg):
        if isinstance(reg,Word):return reg
        if reg in('$xzr','$wzr'):return _constant(0)
        value=self.bindings.get(reg)
        if isinstance(value,Word):return value
        if reg in self._constant_cache:return self._constant_cache[reg]
        if reg in self._constant_active:return None
        self._constant_active.add(reg)
        try:
            ins=self.nodes.get(reg)
            if ins is None:return None
            value=self.evaluate(ins,self.constant,constant=True)
            if value is not None and value.raw is not None:self._constant_cache[reg]=value;return value
            return None
        finally:self._constant_active.remove(reg)

    def evaluate(self,ins,get,constant=False):
        op=ins.op
        if op=='INSERT_SUBREG':return get(ins.refs[1])
        if op.startswith(('DUPv','SUBREG_TO_REG')):return get(ins.refs[0])
        if op=='FMOVv2f64_ns':
            return self.evaluate(Instruction('FMOVDi',(),(),ins.args,ins.block,ins.raw),get,constant)
        if op=='MOVIv8b_ns':return _constant(int(ins.args.split(',')[0]),width=8)
        if op in('MOVi64imm','MOVi32imm'):return _constant(int(ins.args.split(',')[0]),width=32 if op=='MOVi32imm'else 64)
        if op=='constant-zero':return _constant(0)
        if op=='FMOVD0':return _constant(0,'fp')
        if op=='FMOVDi':
            imm=int(ins.args.split(',')[0]);b=(imm>>6)&1
            return _constant(((imm>>7)<<63)|(((1-b)<<10|255*b<<2|((imm>>4)&3))<<52)|((imm&15)<<48),'fp')
        if op.startswith('LDR')or op=='load-value':
            if not ins.refs:return None if constant else self.fail(ins)
            name=self.root(ins.refs[0])
            if name is None:return None if constant else self.fail(ins)
            if name in self.namespace_buffers:
                if ins.op=='LDRDroW'and len(ins.refs)==2:
                    index=get(ins.refs[1])
                    if index is None:return None if constant else self.fail(ins)
                    if ins.args.split(',')[-2:]!=[' 0',' 1']:return None if constant else self.fail(ins)
                    code='uint('+index.code+')'
                elif ins.op=='LDRDui':code=str(int(ins.args.split(',')[1]))+'L'
                else:return None if constant else self.fail(ins)
                return None if constant else Word('fp',name+'['+code+']')
            value=self.field(name);return value if not constant or value.raw is not None else None
        if op=='PHI':
            values=[get(r)for r in ins.refs]
            if all(v is not None and v.raw is not None for v in values)and len({(v.raw,v.kind)for v in values})==1:return values[0]
            if all(v is not None for v in values)and len({(v.code,v.kind)for v in values})==1:return values[0]
            if not constant:
                alias=self.phi_alias(ins,values)
                if alias is not None:return alias
            return None
        if op=='BSPv16i8':
            # An unknown predicate cannot change two identical raw operands.
            # Visit the selected operand lazily when its mask is immutable.
            mask=get(ins.refs[0])
            if mask is not None and mask.raw is not None and (mask.raw&_MASK)in(0,_MASK):
                return get(ins.refs[1]if mask.raw&_MASK else ins.refs[2])
            a,b=(get(r)for r in ins.refs[1:])
            if a is not None and b is not None and a.raw is not None and a.raw==b.raw and a.kind==b.kind:return a
            if mask is None or a is None or b is None:return None if constant else self.fail(ins)
            return Word(a.kind,f'((ulong({mask.code})&ulong({a.code}))|(~ulong({mask.code})&ulong({b.code})))',width=a.width)
        values=[get(r)for r in ins.refs]
        if any(v is None for v in values):return None if constant else self.fail(ins)
        if op in('CSELWr','CSELXr'):
            flags=None
            for previous in self.blocks[ins.block]:
                if previous is ins:break
                if previous.op.startswith(('FCMP','SUBS','ADDS','CCMP','CCMN','FCCMP')):
                    flags=self.flag_values(previous,get,flags)
            if flags is None:return None if constant else self.fail(ins)
            condition=self.condition(int(ins.args.split(',')[2]),flags)
            if isinstance(condition,(bool,np.bool_)):return values[0 if condition else 1]
            return Word('int',f'(({condition})?{values[0].code}:{values[1].code})',width=32 if op=='CSELWr'else 64)
        if op=='BICWrs':
            shift=int(ins.args.split(',')[2]);amount=shift&63;kind=shift>>6
            if kind!=2 or amount>=32:return None if constant else self.fail(ins)
            a,b=values
            if a.raw is not None and b.raw is not None:
                signed=(b.raw&0xffffffff)-(1<<32)if b.raw&(1<<31)else b.raw&0xffffffff
                return _constant((a.raw&0xffffffff)&~(signed>>amount),width=32)
            return Word('int',f'as_type<int>(uint({a.code})&~uint(as_type<int>(uint({b.code}))>>{amount}))',width=32)
        if op=='FCVTZSUWDr':
            value=values[0]
            if value.raw is not None:
                x=_number(value)
                result=0 if np.isnan(x)else (2147483647 if x>=2147483647 else
                    -2147483648 if x<=-2147483648 else int(x))
                return _constant(result,width=32)
            return Word('int','d_to_int32_sat('+value.code+')',width=32)
        if op in('FCVTSDr','FCVTDSr'):
            v=values[0];width=32 if op=='FCVTSDr'else 64
            if v.raw is not None:
                with np.errstate(all='ignore'):
                    raw=int(np.asarray(_number(v),np.float32 if width==32 else np.float64).view(np.uint32 if width==32 else np.uint64))
                return _constant(raw,'fp',width)
            return Word('fp',('f_from_d'if width==32 else'f_to_d')+'('+v.code+')',width=width)
        if op=='COPY':
            v=values[0]
            if ins.types and ins.types[0]=='fpr64'and v.raw is not None:return _constant(v.raw,'fp')
            return v
        vector_fp={'FMULv2f64':'FMULDrr','FMULv2i64_indexed':'FMULDrr','FADDv2f64':'FADDDrr',
            'FSUBv2f64':'FSUBDrr','FDIVv2f64':'FDIVDrr','FNEGv2f64':'FNEGDr'}
        if op in vector_fp:return _fp(vector_fp[op],values)
        if op.startswith(('FMLAv2','FMLSv2')):
            return _call('fma',[_fp('FNEGDr',[values[1]])if op.startswith('FMLS')else values[1],values[2],values[0]])
        if op.startswith('SSHLL'):
            v=values[0]
            if v.raw is not None:
                raw=v.raw&0xffffffff;return _constant(raw if raw<1<<31 else raw-(1<<32))
            return Word('int','long(as_type<int>(uint('+v.code+')))')
        if op in('ORNWrr','ORNXrr'):
            a,b=values;width=32 if op=='ORNWrr'else 64
            if a.raw is not None and b.raw is not None:return _constant(a.raw|~b.raw,width=width)
            cast='uint'if width==32 else'ulong';signed='int'if width==32 else'long'
            return Word('int',f'as_type<{signed}>({cast}({a.code})|~{cast}({b.code}))',width=width)
        if op=='SBFMXri'and re.search(r', 0, 31(?:,|$)',ins.args):
            value=values[0]
            if value.raw is not None:
                raw=value.raw&0xffffffff
                return _constant(raw if raw<2**31 else raw-2**32)
            return Word('int','long(as_type<int>(uint('+value.code+')))')
        if op=='NEGv4i32':
            v=values[0]
            return(_constant(-v.raw,width=32)if v.raw is not None else
                   Word('int','as_type<int>(0u-uint('+v.code+'))',width=32))
        if op=='CMGEv2i64':
            a,b=values
            if a.raw is not None and b.raw is not None:
                signed=lambda v:v.raw if v.raw<1<<63 else v.raw-(1<<64)
                return _constant(-int(signed(a)>=signed(b)),'mask')
            return Word('mask',f'(({a.code})>=({b.code})?-1L:0L)')
        if op.startswith(('UZP1v','XTNv')):
            width=32 if op=='UZP1v4i32'else 16 if op=='UZP1v8i16'else 8 if op=='XTNv8i8'else None
            if width is None or any(v.kind!='mask'for v in values):return None if constant else self.fail(ins)
            canonical=[self.symbolic(r)for r in ins.refs]
            if any(v is None for v in canonical)or len({v.code for v in canonical})!=1:
                return None if constant else self.fail(ins)
            return Word('mask',values[0].code,values[0].raw,width)
        if op=='ANDv8i8':
            a,b=values
            if a.raw is not None and b.raw is not None:return _constant(a.raw&b.raw,width=8)
            return Word('int',f'(uchar({a.code})&uchar({b.code}))',width=8)
        if op in('FADDSrr','FSUBSrr','FMULSrr','FDIVSrr'):
            function={'FADDSrr':'f_add','FSUBSrr':'f_sub','FMULSrr':'f_mul','FDIVSrr':'f_div'}[op]
            return Word('fp',function+'('+','.join(v.code for v in values)+')',width=32)
        if op in('FMULDrr','FNMULDrr','FADDDrr','FSUBDrr','FDIVDrr','FMADDDrrr','FMSUBDrrr','FNMADDDrrr','FNMSUBDrrr','FNEGDr'):return _fp(op,values)
        if op.startswith('call:'):return _call(op.split(':',1)[1],values)
        if op in('FCVTZSUXDr','FCVTZSv2f64'):
            if values[0].raw is not None:
                x=_number(values[0]);result=(0 if np.isnan(x)else (1<<63)-1 if x>=float(1<<63)
                    else -(1<<63)if x<=float(-(1<<63))else int(x))
                return _constant(result)
            return Word('int','d_to_int('+values[0].code+')')
        if op.startswith(('SCVTF','UCVTF')):
            v=values[0];unsigned=op.startswith('UCVTF')
            width=32 if 'UW'in op else v.width
            output_width=32 if ins.types and ins.types[0]=='fpr32'else 64
            if v.raw is not None:
                raw=v.raw&((1<<width)-1)
                number=raw if unsigned or raw<1<<(width-1)else raw-(1<<width)
                raw=_integer_float32_bits(number)if output_width==32 else _bits(number)
                return _constant(raw,'fp',width=output_width)
            code=('uint('+v.code+')'if unsigned else'as_type<int>(uint('+v.code+'))')if width==32 else v.code
            function=('f_operator_from_uint'if output_width==32 else'd_from_uint')if unsigned else('f_from_int'if output_width==32 else'd_from_int')
            cast='ulong'if unsigned else'long'
            return Word('fp',function+'('+cast+'('+code+'))',width=output_width)
        if op.startswith(('SUBSX','SUBSW','ADDSX','ADDSW','ADDX','ADDW','SUBX','SUBW')):
            if not values:return None if constant else self.fail(ins)
            width=32 if'W'in op else 64;a=values[0]
            b=values[1]if len(values)>1 else _constant(int(ins.args.split(',')[1].strip()),width=width)
            subtract=op.startswith('SUB')
            if a.raw is not None and b.raw is not None:return _constant(a.raw+(-b.raw if subtract else b.raw),width=width)
            symbol='-'if subtract else'+';cast='as_type<int>(uint'if width==32 else'as_type<long>(ulong'
            return Word('int',cast+'('+a.code+')'+symbol+('uint'if width==32 else'ulong')+'('+b.code+'))',width=width)
        if constant:return None
        return self.fail(ins)

    def phi_alias(self,ins,values):
        """Prove Cython's checked-call sentinel merge preserves its value.

        A successful call can return -1. Clang can materialize that same
        sentinel on the PyErr_Occurred()==0 edge rather than carrying the
        returned SSA register. The edge must prove equality; matching the
        constant alone would incorrectly erase user conditional expressions.
        """
        if self.metadata['target']!='cython'or not hasattr(self,'successors'):return None
        pairs=re.findall(r'(%\d+), %bb\.(\d+)',ins.args)
        candidates={v.code:v for v in values if v is not None and v.raw is None}
        if len(candidates)!=1 or any(v is None for v in values):return None
        value=next(iter(candidates.values()))
        predecessors={b:{p for p in self.reachable if b in self.successors[p]}for b in self.reachable}

        def equal_edge(source,target,sentinel):
            compare=None;yes=None;no=None
            for item in self.blocks[source]:
                if item.op=='FCMPDrr'and len(item.refs)==2:
                    try:a,b=(self.symbolic(r)for r in item.refs)
                    except ProfileError:return False
                    if a and b:
                        compare=bool((a.code==value.code and b.raw==sentinel.raw)
                                     or(b.code==value.code and a.raw==sentinel.raw))
                elif item.op.startswith(('ADDSWri','SUBSWri'))and len(item.refs)==1:
                    try:operand=self.symbolic(item.refs[0])
                    except ProfileError:return False
                    immediate=int(item.args.split(',')[1])
                    expected=(-immediate if item.op.startswith('ADD')else immediate)&0xffffffff
                    compare=bool(operand and operand.code==value.code and sentinel.kind=='int'
                                 and (sentinel.raw&0xffffffff)==expected)
                elif item.op=='Bcc':
                    cc,destination=re.match(r'(\d+), %bb\.(\d+)',item.args).groups()
                    if int(cc)==0:yes=int(destination)
                    elif int(cc)==1:no=int(destination)
                    else:return False
                elif item.op=='B':
                    destination=int(re.search(r'%bb\.(\d+)',item.args)[1])
                    if yes is None:yes=destination
                    else:no=destination
            return bool(compare and yes==target and no is not None)

        def guarded(block,sentinel,seen):
            if block in seen or block not in self.region or block==self.header:return False
            zero=getattr(self,'caller_zero_return_words',{}).get((block,ins.block))
            if sentinel.raw==0 and zero is not None and zero.code==value.code:return True
            primitive=getattr(self,'caller_error_words',{}).get(block)
            if primitive is not None and primitive.code==value.code:
                # The function registry has identified an inlined primitive's
                # error return, not a user branch. The existing native sampler
                # supplies this value and owns its error flag/commit prefix.
                return sentinel.raw in(0xffffffff,0xbff0000000000000)
            incoming=predecessors.get(block,set())
            return bool(incoming)and all(equal_edge(p,block,sentinel)or guarded(p,sentinel,seen|{block})for p in incoming)

        for (_,block),other in zip(pairs,values):
            if other.code==value.code:continue
            if other.raw is None or not guarded(int(block),other,{ins.block}):return None
        return value

    def fail(self,ins):raise ProfileError('Unlowered '+ins.raw)

    def flag_values(self,ins,get,previous=None):
        op=ins.op;values=[get(r)for r in ins.refs]
        if not values or any(v is None for v in values):return None
        if op.startswith(('CCMP','CCMN','FCCMP')):
            fields=ins.args.split(',');fallback=int(fields[2]);cc=int(fields[3])
            condition=self.condition(cc,previous)
            if condition is None:return None
            operation=('FCMPDrr'if op.startswith('FCCMP')else('ADDS'if op.startswith('CCMN')else'SUBS')
                +('W'if'W'in op else'X')+('rr'if len(values)>1 else'ri'))
            ordinary=self.flag_values(Instruction(operation,ins.defs,ins.refs,ins.args,ins.block,ins.raw,ins.types),get)
            if ordinary is None:return None
            alternative=tuple(bool(fallback&bit)for bit in(8,4,2,1))
            if condition is True:return ordinary
            if condition is False:return alternative
            code=lambda value:('true'if value else'false')if isinstance(value,(bool,np.bool_))else value
            return tuple('('+condition+')?('+code(a)+'):'+code(b)for a,b in zip(ordinary,alternative))
        if op.startswith('FCMP'):
            a=values[0];b=values[1]if len(values)>1 else _constant(0,'fp')
            if a.raw is not None and b.raw is not None:
                x,y=_number(a),_number(b);return(x<y,x==y,not(x<y),bool(np.isnan(x)or np.isnan(y)))
            return('d_lt('+a.code+','+b.code+')','d_eq('+a.code+','+b.code+')',
                '!d_lt('+a.code+','+b.code+')','(d_is_nan('+a.code+')||d_is_nan('+b.code+'))')
        width=32 if'W'in op else 64;a=values[0]
        b=values[1]if len(values)>1 else _constant(int(ins.args.split(',')[1]),width=width)
        if a.raw is not None and b.raw is not None:
            mask=(1<<width)-1;x=a.raw&mask;y=b.raw&mask;add=op.startswith('ADDS');r=(x+y if add else x-y)&mask
            overflow=((~(x^y)if add else x^y)&(x^r))>>(width-1)&1
            return(bool(r>>(width-1)),r==0,x+y>mask if add else x>=y,bool(overflow))
        cast='uint'if width==32 else'ulong';x=f'{cast}({a.code})';y=f'{cast}({b.code})';add=op.startswith('ADDS')
        r=f'{cast}({x}{"+"if add else"-"}{y})';shift=str(width-1)
        return(f'bool({r}>>{shift})',f'({r}==0)',f'({r}<{x})'if add else f'({x}>={y})',
            f'bool((({"~"if add else""}({x}^{y}))&({x}^{r}))>>{shift})')

    @staticmethod
    def condition(code,flags):
        if flags is None:return None
        n,z,c,v=flags
        if all(isinstance(x,(bool,np.bool_))for x in flags):
            return {0:z,1:not z,2:c,3:not c,4:n,5:not n,6:v,7:not v,8:c and not z,
                9:not c or z,10:n==v,11:n!=v,12:not z and n==v,13:z or n!=v,14:True}[code]
        return{0:z,1:f'!({z})',2:c,3:f'!({c})',4:n,5:f'!({n})',6:v,7:f'!({v})',
            8:f'({c})&&!({z})',9:f'!({c})||({z})',10:f'({n})==({v})',11:f'({n})!=({v})',
            12:f'!({z})&&(({n})==({v}))',13:f'({z})||(({n})!=({v}))',14:'true'}[code]

    def branch(self,ins,flags,get):
        op=ins.op
        if op=='B':return True,int(re.search(r'%bb\.(\d+)',ins.args)[1])
        if op=='Bcc':return self.condition(int(ins.args.split(',')[0]),flags),int(re.search(r'%bb\.(\d+)',ins.args)[1])
        if op.startswith(('CBZ','CBNZ','TBZ','TBNZ')):
            target=int(re.search(r'%bb\.(\d+)',ins.args)[1])
            v=get(ins.refs[0])if ins.refs else None;nonzero=op.startswith(('CBNZ','TBNZ'))
            bit=int(ins.args.split(',')[1])if op.startswith('TB')else None
            if v is None:return None,target
            if v.raw is not None:
                truth=bool(v.raw&(1<<bit))if bit is not None else bool(v.raw)
                return truth==nonzero,target
            expr=f'(({v.code}&{1<<bit}L)!=0)'if bit is not None else f'({v.code}!=0)'
            return expr if nonzero else'!'+expr,target
        return None

    def edges(self,block):
        flags=None;edges=[];terminated=False
        for ins in self.blocks[block]:
            if ins.op.startswith('RET_'):terminated=True;break
            if ins.op.startswith(('FCMP','SUBS','ADDS','CCMP','CCMN','FCCMP')):flags=self.flag_values(ins,self.constant,flags)
            branch=self.branch(ins,flags,self.constant)
            if branch:
                condition,target=branch
                if condition is not False:edges.append(target)
                if condition is True:terminated=True;break
        order=list(self.blocks);index=order.index(block)
        if not terminated and index+1<len(order):edges.append(order[index+1])
        return set(edges)

    def components(self):
        index={};low={};stack=[];on=set();result=[]
        def visit(v):
            index[v]=low[v]=len(index);stack.append(v);on.add(v)
            for w in self.successors[v]:
                if w not in index:visit(w);low[v]=min(low[v],low[w])
                elif w in on:low[v]=min(low[v],index[w])
            if low[v]==index[v]:
                part=set()
                while True:
                    w=stack.pop();on.remove(w);part.add(w)
                    if w==v:break
                result.append(part)
        for v in self.reachable:
            if v not in index:visit(v)
        return result

    def declare(self,reg,kind,width=64):
        name='_profile_r'+reg.replace('%','').replace(':','_');self.declarations[name]=('uint'if width==32 else'ulong')if kind=='fp'else'long'
        return Word(kind,name,width=width)

    def descriptor(self,reg,active=None):
        active=set()if active is None else active
        if reg in active:return'int',64
        active=active|{reg};ins=self.nodes.get(reg)
        if ins is None:return'int',64
        op=ins.op
        if op.startswith('LDR')or op=='load-value':
            name=self.root(ins.refs[0])if ins.refs else None
            if name in self.array_info:
                value=self.field(name);return value.kind,value.width
        if op=='CMGEv2i64':return'mask',64
        if op in('UZP1v4i32','UZP1v8i16','XTNv8i8'):
            return'mask',{'UZP1v4i32':32,'UZP1v8i16':16,'XTNv8i8':8}[op]
        if op=='ANDv8i8':return'int',8
        if op in('NEGv4i32','ORNWrr'):return'int',32
        if op=='FCVTSDr':return'fp',32
        if op=='FCVTDSr':return'fp',64
        if op.startswith(('FCVTZS','SSHLL')):return'int',64
        if op.startswith(('SCVTF','UCVTF')):return'fp',32 if ins.types and ins.types[0]=='fpr32'else 64
        if op=='INSERT_SUBREG':return self.descriptor(ins.refs[1],active)
        if op.startswith(('DUPv','SUBREG_TO_REG'))or op=='COPY':
            return self.descriptor(ins.refs[0],active)
        return('fp'if ins.types and ins.types[0].startswith('fpr')or op.startswith('call:')else'int'),32 if ins.types and ins.types[0]=='fpr32'else 64

    def symbolic(self,reg):
        value=self.constant(reg)
        if value is not None:return value
        if reg in self._symbolic_cache:return self._symbolic_cache[reg]
        if reg in self._symbolic_active:return None
        self._symbolic_active.add(reg)
        try:
            ins=self.nodes.get(reg)
            if ins is None:return None
            value=self.evaluate(ins,self.symbolic)
            self._symbolic_cache[reg]=value;return value
        finally:self._symbolic_active.remove(reg)

    def value(self,reg):
        constant=self.constant(reg)
        if constant is not None:return constant
        if reg in self.values:return self.values[reg]
        ins=self.nodes.get(reg)
        if ins is None:raise ProfileError('No arithmetic binding for '+str(reg))
        if self.root(reg)is not None:raise ProfileError('Array pointer used numerically '+reg)
        kind,width=self.descriptor(reg)
        v=self.declare(reg,kind,width);self.values[reg]=v
        if ins.block not in self.region:
            result=self.evaluate(ins,self.value)
            if result is None:self.fail(ins)
            self.prelude.append(v.code+'='+result.code+';')
        return v

    def destination(self,target,block):
        if target==self.header or target in self.normal_exits:
            if block in self.latches:return '_profile_pc=-1;break;'
        if target not in self.region:
            self.may_raise=True;return '_profile_error=2u;_profile_pc=-1;break;'
        return f'_profile_previous={block};_profile_pc={target};break;'

    def render(self):
        output={};flag_names=tuple('_profile_'+s for s in('n','z','c','overflow'))
        for block in sorted(self.region):
            lines=[];flags=None
            for ins in self.blocks[block]:
                op=ins.op
                if op.startswith(('FCMP','SUBS','ADDS','CCMP','CCMN','FCCMP')):
                    flags=self.flag_values(ins,self.constant,flags)
                    runtime=self.flag_values(ins,self.value,flag_names)
                    if runtime is None:self.fail(ins)
                    updates=tuple('_profile_next_'+str(i)for i in range(4))
                    code=lambda v:str(v).lower()if isinstance(v,(bool,np.bool_))else v
                    lines.append('{'+''.join('bool '+name+'='+code(v)+';'for name,v in zip(updates,runtime))
                        +''.join(name+'='+value+';'for name,value in zip(flag_names,updates))+'}')
                branch=self.branch(ins,flags,self.constant)
                if branch:
                    condition,target=branch
                    if condition is False:continue
                    destination=self.destination(target,block)
                    if condition is True:lines.append(destination);break
                    runtime=self.branch(ins,flag_names,self.value)[0]
                    lines.append(f'if({runtime}){{{destination}}}');continue
                if op.startswith(('ADJCALL','DBG_','LOADgot')):continue
                if op=='BL':
                    if'PyErr_Occurred'in ins.args:continue
                    if id(ins)in self.calls:
                        key=self.calls[id(ins)]
                        if self.constant(key)is None:
                            result=self.value(key);value=self.evaluate(self.nodes[key],self.value)
                            lines.append(result.code+'='+value.code+';')
                        continue
                    self.fail(ins)
                if op.startswith('RET_'):lines.append('_profile_pc=-1;break;');break
                name=self.store_name(ins)
                if name in self.writes:
                    value=self.value(ins.refs[0]);lines.append(name+'='+('bool('+value.code+')'if self.array_info[name]['dtype']=='bool'else value.code)+';')
                    continue
                for dest in ins.defs:
                    if dest.startswith('$')or self.root(dest)is not None or self.constant(dest)is not None:continue
                    result=self.value(dest)
                    if op=='PHI':
                        if block==self.header:
                            lines.append(result.code+'=long(i);');continue
                        pairs=[(r,b)for r,b in re.findall(r'(%\d+), %bb\.(\d+)',ins.args)
                               if int(b)in self.region and block in self.successors[int(b)]]
                        if not pairs:self.fail(ins)
                        expr=''.join(f'_profile_previous=={b}?{self.value(r).code}:'for r,b in pairs[:-1])+self.value(pairs[-1][0]).code
                    elif op in('CSINCWr','FCSELDrrr'):
                        cc=int(ins.args.split(',')[2]);condition=self.condition(cc,flag_names)
                        if op=='CSINCWr':expr=f'({condition})?{self.value(ins.refs[0]).code}:({self.value(ins.refs[1]).code}+1L)'
                        else:expr=f'({condition})?{self.value(ins.refs[0]).code}:{self.value(ins.refs[1]).code}'
                    else:
                        value=self.evaluate(self.nodes[dest],self.value)
                        if value is None:self.fail(ins)
                        expr=value.code
                    lines.append(result.code+'='+expr+';')
            else:
                order=list(self.blocks);index=order.index(block)
                if index+1<len(order):lines.append(self.destination(order[index+1],block))
            output[block]='\n'.join(lines)
        captures=[]
        for name,code in self.initial.items():
            if name in self.constants:continue
            dtype=self.array_info[name]['dtype'];captures.append(('ulong'if dtype=='float64'else'long')+' '+code+'='+name+';')
        declarations='\n'.join(dtype+' '+name+'=0;'for name,dtype in self.declarations.items())
        source='\n'.join(captures)+'\n'+declarations+'\n'+'\n'.join(self.prelude)
        source+='\nbool '+','.join(n+'=false'for n in flag_names)+';\nuint _profile_error=0;\n'
        source+=f'int _profile_pc={self.header},_profile_previous=-1;while(_profile_pc>=0){{switch(_profile_pc){{\n'
        source+='\n'.join(f'case {b}:{{\n{code}\n}}'for b,code in output.items())+'\n}}'
        return dict(source=source,writes=list(self.metadata['writes']),may_raise=self.may_raise)


def lower(text,metadata,*,fixed_values=None):
    full=Graph(text,metadata);profile=full.render()
    simd=any(ins.op.startswith(('FMULv','FMLAv','FADDv','FSUBv'))for b in full.region for ins in full.blocks[b])
    if simd:
        # Brian's independently allocated state arrays satisfy Clang's
        # non-overlap check. Preserve its distinct SIMD and scalar-tail DAGs.
        # Projection to one cell is valid only if packed mask lanes and every
        # unrolled store prove identical symbolic expressions after replacing
        # each pointwise input lane by that cell's actual input.
        expressions={}
        for block in full.region:
            for ins in full.blocks[block]:
                name=full.store_name(ins)
                if name not in full.writes:continue
                value=full.symbolic(ins.refs[0])
                if value is None:raise ProfileError('Cannot prove pointwise SIMD store')
                expressions.setdefault(name,set()).add(value.code)
        if any(len(values)!=1 for values in expressions.values()):
            raise ProfileError('SIMD unroll positions require distinct cell projections')
    if full.other_region is not None:
        tail=Graph(text,metadata,region=full.other_region).render()
        profile['source']=f'if(i<{full.bulk_count}){{\n{profile["source"]}\n}}else{{\n{tail["source"]}\n}}'
        profile['may_raise']|=tail['may_raise']
        profile['simd_cells']=full.bulk_count
    # A bit-exact fixed point is a preparation proof over actual compiled
    # operations. Unknown inputs remain symbolic; both sides of a conditional
    # store must prove the same bits. Refractory bookkeeping still executes.
    if fixed_values and full.other_region is None and not profile['may_raise']and set(profile['writes'])=={'v','g','not_refractory'}:
        reduced=Graph(text,metadata,fixed_values);seen=set();proved=True
        for block in reduced.region:
            for ins in reduced.blocks[block]:
                name=reduced.store_name(ins)
                if name not in fixed_values:continue
                seen.add(name);value=reduced.constant(ins.refs[0])
                if value is None or value.raw!=fixed_values[name]:proved=False
        if proved and seen==set(fixed_values)&set(profile['writes']):
            rest=reduced.render()
            if not rest['may_raise']:
                guard='&&'.join(f'{name}=={word}ul'for name,word in sorted(fixed_values.items()))
                profile['source']=f'if({guard}){{\n{rest["source"]}\n}}else{{\n{profile["source"]}\n}}'
                profile['fixed_point_words']={name:fixed_values[name]for name in seen}
                profile['specialization_words']=fixed_values
    return profile
