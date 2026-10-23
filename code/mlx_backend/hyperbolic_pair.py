"""Exact reuse for existing binary64 sinh/cosh calls with one SSA operand.

Only their common middle-range reduction is shared. Each output retains the
original helper's operation order, including its final rounding and NaN policy.
The guarded fallback calls the existing helpers. This adds no numerical engine.
"""

MSL_HEADER = r'''
struct _eon_hyper_pair_value { ulong sinh; ulong cosh; };
inline _eon_hyper_pair_value _eon_hyper_pair(ulong x) {
    ulong a=d_abs(x);
    if(a<0x3fd62e4200000000ul || a>=0x4035000000000000ul)
        return {d_hyper_sinh(x),d_hyper_cosh(x)};
    long k;ulong delta;
    ulong r=hyper_exp_reduced(a,&k,&delta);
    uint i=uint(k)&127u,j=127u-i;
    ulong negative_r=d_sub(0x3f80000000000000ul,r);
    ulong positive_factor=hyper_exp_factor(r,delta,hyper_exp_table[2*i+1]);
    ulong negative_factor=hyper_exp_factor(negative_r,d_neg(delta),hyper_exp_table[2*j+1]);
    long scale=k>>7;
    ulong positive=hyper_exp_table[2*i]+(ulong(scale-1)<<52);
    ulong negative=hyper_exp_table[2*j]-(ulong(scale+2)<<52);
    ulong shigh=d_sub(positive,negative);
    ulong scorrection=d_add(d_sub(shigh,positive),negative);
    scorrection=hyper_fma_normal(negative,negative_factor,scorrection);
    scorrection=hyper_fma_normal(d_neg(positive),positive_factor,scorrection);
    ulong chigh=d_add(positive,negative);
    ulong ccorrection=d_sub(d_sub(chigh,positive),negative);
    ccorrection=hyper_fma_normal(negative,negative_factor,d_neg(ccorrection));
    ccorrection=hyper_fma_normal(positive,positive_factor,ccorrection);
    return {d_sub(shigh,scorrection)|(x&0x8000000000000000ul),d_add(chigh,ccorrection)};
}
'''


class _PairValues:
    def __init__(self, used_math, prefix="_eon_hyper_pair_"):
        self.used_math = used_math
        self.prefix = prefix
        self.names = {}

    def _expression(self, argument, function, emit, lines):
        if argument not in self.names:
            value = emit(argument)
            name = self.prefix + str(len(self.names))
            self.names[argument] = name
            lines.append(f"const _eon_hyper_pair_value {name} = _eon_hyper_pair({value});")
            self.used_math.update(("sinh", "cosh", "_hyperbolic_pair"))
        return self.names[argument] + "." + function


class SsaPairs(_PairValues):
    """Find complementary reachable calls, retaining typed SSA identity.

    The existing mixed-width graph keeps this struct in one emitted scope.
    A changed operand or branch produces a different SSA node and cannot hit
    this cache. Calls with only one live output retain their original helper.
    """

    def __init__(self, outputs, used_math):
        super().__init__(used_math)
        groups, visited = {}, set()

        def visit(node):
            if node in visited:
                return
            visited.add(node)
            for argument in node.args:
                visit(argument)
            if node.dtype != "float64":
                return
            function = node.op.removeprefix("call:").removeprefix("llvm.").split(".")[0]
            if function in ("sinh", "cosh") and len(node.args) == 1:
                groups.setdefault(node.args[0], {})[function] = node

        for node in outputs.values():
            visit(node)
        self.nodes = {node: (argument, function)
                      for argument, calls in groups.items() if len(calls) == 2
                      for function, node in calls.items()}

    def expression(self, node, emit, lines):
        found = self.nodes.get(node)
        if found is None:
            return None
        argument, function = found
        return self._expression(argument, function, emit, lines)


class IrPairs(_PairValues):
    """The same typed reuse within one existing mixed integer CFG block.

    Calls in distinct blocks cannot share a local declaration or execution
    context. Only a literal selected libm double call and one actual LLVM SSA
    operand qualify; float calls and expressions resembling operands do not.
    """
    def __init__(self, instructions, used_math):
        import re
        super().__init__(used_math)
        calls = {}
        for instruction in instructions:
            call = re.search(r"\bcall\s+(?:\w+\s+)*double\s+@(?:llvm\.)?(sinh|cosh)(?:\.f64)?"
                             r"\(double\s+(?:noundef\s+)?(%[\w.]+)\)", instruction)
            if call:
                calls.setdefault(call[2], set()).add(call[1])
        self.arguments = {argument for argument, functions in calls.items()
                          if functions == {"sinh", "cosh"}}

    def expression(self, function, dtype, arguments, emit, lines):
        if (function not in ("sinh", "cosh", "llvm.sinh.f64", "llvm.cosh.f64") or dtype != "double"
                or len(arguments) != 1 or arguments[0][0] != "double"
                or arguments[0][1] not in self.arguments):
            return None
        return self._expression(arguments[0][1], function.removeprefix("llvm.").split(".")[0],
                                lambda value: emit(value, "double"), lines)


class StatementPairs(_PairValues):
    """Reuse pure scalar-local arguments across straight-line statements.

    Only a named operand is eligible, so calls, random draws and TimedArray
    accesses are never evaluated fewer times. Every write versions that name;
    a conditional write/Boolean branch ends the reuse context. This planner is
    for the Cython/CPP scalar-local renderer, not NumPy's aliased array objects.
    """
    def __init__(self, statements, used_math):
        import ast
        super().__init__(used_math, "_eon_hyper_statement_pair_")
        versions, groups, records, epoch = {}, {}, {}, 0
        for statement, conditional in statements:
            if (conditional or statement.var=="not_refractory"
                    or getattr(statement, "boolean_simplified_expressions", None)):
                epoch += 1
            else:
                root = ast.parse(statement.expr, mode="eval").body
                for node in ast.walk(root):
                    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                            and node.func.id in ("sinh", "cosh") and len(node.args)==1
                            and isinstance(node.args[0], ast.Name)):
                        name = node.args[0].id
                        key = (epoch, name, versions.get(name, 0))
                        groups.setdefault(key, set()).add(node.func.id)
                        records[(id(statement), node.func.id, name)] = key
            versions[statement.var] = versions.get(statement.var, 0) + 1
        self.calls = {call:key for call,key in records.items()
                      if groups[key]=={"sinh", "cosh"}}
        self.statement = None

    def expression(self, function, argument, code, lines):
        import ast
        if not isinstance(argument, ast.Name):
            return None
        key = self.calls.get((id(self.statement), function, argument.id))
        if key is None:
            return None
        return self._expression(key, function, lambda _:code, lines)
