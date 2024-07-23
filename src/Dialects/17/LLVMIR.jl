module llvm

import ...IR:
    IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes

"""
`ashr`

"""
function ashr(
    lhs::Value, rhs::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.ashr",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`add`

"""
function add(
    lhs::Value, rhs::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.add",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`addrspacecast`

"""
function addrspacecast(arg::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[arg,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.addrspacecast",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mlir_addressof`

Creates an SSA value containing a pointer to a global variable or constant
defined by `llvm.mlir.global`. The global value can be defined after its
first referenced. If the global value is a constant, storing into it is not
allowed.

Examples:

```mlir
func @foo() {
  // Get the address of a global variable.
  %0 = llvm.mlir.addressof @const : !llvm.ptr<i32>

  // Use it as a regular pointer.
  %1 = llvm.load %0 : !llvm.ptr<i32>

  // Get the address of a function.
  %2 = llvm.mlir.addressof @foo : !llvm.ptr<func<void ()>>

  // The function address can be used for indirect calls.
  llvm.call %2() : () -> ()
}

// Define the global.
llvm.mlir.global @const(42 : i32) : i32
```
"""
function mlir_addressof(; res::IR.Type, global_name, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("global_name", global_name),]

    return IR.create_operation(
        "llvm.mlir.addressof",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`alloca`

"""
function alloca(
    arraySize::Value;
    res::IR.Type,
    alignment=nothing,
    elem_type=nothing,
    inalloca=nothing,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[arraySize,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(alignment) && push!(attributes, namedattribute("alignment", alignment))
    !isnothing(elem_type) && push!(attributes, namedattribute("elem_type", elem_type))
    !isnothing(inalloca) && push!(attributes, namedattribute("inalloca", inalloca))

    return IR.create_operation(
        "llvm.alloca",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`and`

"""
function and(
    lhs::Value, rhs::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.and",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`cmpxchg`

"""
function cmpxchg(
    ptr::Value,
    cmp::Value,
    val::Value;
    res::IR.Type,
    success_ordering,
    failure_ordering,
    syncscope=nothing,
    alignment=nothing,
    weak=nothing,
    volatile_=nothing,
    access_groups=nothing,
    alias_scopes=nothing,
    noalias_scopes=nothing,
    tbaa=nothing,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[ptr, cmp, val]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("success_ordering", success_ordering),
        namedattribute("failure_ordering", failure_ordering),
    ]
    !isnothing(syncscope) && push!(attributes, namedattribute("syncscope", syncscope))
    !isnothing(alignment) && push!(attributes, namedattribute("alignment", alignment))
    !isnothing(weak) && push!(attributes, namedattribute("weak", weak))
    !isnothing(volatile_) && push!(attributes, namedattribute("volatile_", volatile_))
    !isnothing(access_groups) &&
        push!(attributes, namedattribute("access_groups", access_groups))
    !isnothing(alias_scopes) &&
        push!(attributes, namedattribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) &&
        push!(attributes, namedattribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, namedattribute("tbaa", tbaa))

    return IR.create_operation(
        "llvm.cmpxchg",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`atomicrmw`

"""
function atomicrmw(
    ptr::Value,
    val::Value;
    res::IR.Type,
    bin_op,
    ordering,
    syncscope=nothing,
    alignment=nothing,
    volatile_=nothing,
    access_groups=nothing,
    alias_scopes=nothing,
    noalias_scopes=nothing,
    tbaa=nothing,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[ptr, val]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("bin_op", bin_op), namedattribute("ordering", ordering)
    ]
    !isnothing(syncscope) && push!(attributes, namedattribute("syncscope", syncscope))
    !isnothing(alignment) && push!(attributes, namedattribute("alignment", alignment))
    !isnothing(volatile_) && push!(attributes, namedattribute("volatile_", volatile_))
    !isnothing(access_groups) &&
        push!(attributes, namedattribute("access_groups", access_groups))
    !isnothing(alias_scopes) &&
        push!(attributes, namedattribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) &&
        push!(attributes, namedattribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, namedattribute("tbaa", tbaa))

    return IR.create_operation(
        "llvm.atomicrmw",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`bitcast`

"""
function bitcast(arg::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[arg,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.bitcast",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`br`

"""
function br(
    destOperands::Vector{Value}; loop_annotation=nothing, dest::Block, location=Location()
)
    _results = IR.Type[]
    _operands = Value[destOperands...,]
    _owned_regions = Region[]
    _successors = Block[dest,]
    _attributes = NamedAttribute[]
    !isnothing(loop_annotation) &&
        push!(attributes, namedattribute("loop_annotation", loop_annotation))

    return IR.create_operation(
        "llvm.br",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`call`

In LLVM IR, functions may return either 0 or 1 value. LLVM IR dialect
implements this behavior by providing a variadic `call` operation for 0- and
1-result functions. Even though MLIR supports multi-result functions, LLVM
IR dialect disallows them.

The `call` instruction supports both direct and indirect calls. Direct calls
start with a function name (`@`-prefixed) and indirect calls start with an
SSA value (`%`-prefixed). The direct callee, if present, is stored as a
function attribute `callee`. The trailing type list contains the optional
indirect callee type and the MLIR function type, which differs from the
LLVM function type that uses a explicit void type to model functions that do
not return a value.

Examples:

```mlir
// Direct call without arguments and with one result.
%0 = llvm.call @foo() : () -> (f32)

// Direct call with arguments and without a result.
llvm.call @bar(%0) : (f32) -> ()

// Indirect call with an argument and without a result.
llvm.call %1(%0) : !llvm.ptr, (f32) -> ()
```
"""
function call(
    operand_0::Vector{Value};
    result=nothing::Union{Nothing,IR.Type},
    callee=nothing,
    fastmathFlags=nothing,
    branch_weights=nothing,
    access_groups=nothing,
    alias_scopes=nothing,
    noalias_scopes=nothing,
    tbaa=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[operand_0...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(result) && push!(_results, result)
    !isnothing(callee) && push!(attributes, namedattribute("callee", callee))
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))
    !isnothing(branch_weights) &&
        push!(attributes, namedattribute("branch_weights", branch_weights))
    !isnothing(access_groups) &&
        push!(attributes, namedattribute("access_groups", access_groups))
    !isnothing(alias_scopes) &&
        push!(attributes, namedattribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) &&
        push!(attributes, namedattribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, namedattribute("tbaa", tbaa))

    return IR.create_operation(
        "llvm.call",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`comdat`

Provides access to object file COMDAT section/group functionality.

Examples:
```mlir
llvm.comdat @__llvm_comdat {
  llvm.comdat_selector @any any
}
llvm.mlir.global internal constant @has_any_comdat(1 : i64) comdat(@__llvm_comdat::@any) : i64
```
"""
function comdat(; sym_name, body::Region, location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[body,]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("sym_name", sym_name),]

    return IR.create_operation(
        "llvm.comdat",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`comdat_selector`

Provides access to object file COMDAT section/group functionality.

Examples:
```mlir
llvm.comdat @__llvm_comdat {
  llvm.comdat_selector @any any
}
llvm.mlir.global internal constant @has_any_comdat(1 : i64) comdat(@__llvm_comdat::@any) : i64
```
"""
function comdat_selector(; sym_name, comdat, location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("sym_name", sym_name), namedattribute("comdat", comdat)
    ]

    return IR.create_operation(
        "llvm.comdat_selector",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`cond_br`

"""
function cond_br(
    condition::Value,
    trueDestOperands::Vector{Value},
    falseDestOperands::Vector{Value};
    branch_weights=nothing,
    loop_annotation=nothing,
    trueDest::Block,
    falseDest::Block,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[condition, trueDestOperands..., falseDestOperands...]
    _owned_regions = Region[]
    _successors = Block[trueDest, falseDest]
    _attributes = NamedAttribute[]
    push!(
        attributes,
        operandsegmentsizes([1, length(trueDestOperands), length(falseDestOperands)]),
    )
    !isnothing(branch_weights) &&
        push!(attributes, namedattribute("branch_weights", branch_weights))
    !isnothing(loop_annotation) &&
        push!(attributes, namedattribute("loop_annotation", loop_annotation))

    return IR.create_operation(
        "llvm.cond_br",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mlir_constant`

Unlike LLVM IR, MLIR does not have first-class constant values. Therefore,
all constants must be created as SSA values before being used in other
operations. `llvm.mlir.constant` creates such values for scalars and
vectors. It has a mandatory `value` attribute, which may be an integer,
floating point attribute; dense or sparse attribute containing integers or
floats. The type of the attribute is one of the corresponding MLIR builtin
types. It may be omitted for `i64` and `f64` types that are implied. The
operation produces a new SSA value of the specified LLVM IR dialect type.
The type of that value _must_ correspond to the attribute type converted to
LLVM IR.

Examples:

```mlir
// Integer constant, internal i32 is mandatory
%0 = llvm.mlir.constant(42 : i32) : i32

// It\'s okay to omit i64.
%1 = llvm.mlir.constant(42) : i64

// Floating point constant.
%2 = llvm.mlir.constant(42.0 : f32) : f32

// Splat dense vector constant.
%3 = llvm.mlir.constant(dense<1.0> : vector<4xf32>) : vector<4xf32>
```
"""
function mlir_constant(; res::IR.Type, value, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("value", value),]

    return IR.create_operation(
        "llvm.mlir.constant",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`extractelement`

"""
function extractelement(vector::Value, position::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[vector, position]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.extractelement",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`extractvalue`

"""
function extractvalue(container::Value; res::IR.Type, position, location=Location())
    _results = IR.Type[res,]
    _operands = Value[container,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("position", position),]

    return IR.create_operation(
        "llvm.extractvalue",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`fadd`

"""
function fadd(
    lhs::Value,
    rhs::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.fadd",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`fcmp`

"""
function fcmp(
    lhs::Value,
    rhs::Value;
    res::IR.Type,
    predicate,
    fastmathFlags=nothing,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("predicate", predicate),]
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.fcmp",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`fdiv`

"""
function fdiv(
    lhs::Value,
    rhs::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.fdiv",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`fmul`

"""
function fmul(
    lhs::Value,
    rhs::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.fmul",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`fneg`

"""
function fneg(
    operand::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.fneg",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`fpext`

"""
function fpext(arg::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[arg,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.fpext",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`fptosi`

"""
function fptosi(arg::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[arg,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.fptosi",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`fptoui`

"""
function fptoui(arg::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[arg,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.fptoui",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`fptrunc`

"""
function fptrunc(arg::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[arg,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.fptrunc",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`frem`

"""
function frem(
    lhs::Value,
    rhs::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.frem",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`fsub`

"""
function fsub(
    lhs::Value,
    rhs::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.fsub",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`fence`

"""
function fence(; ordering, syncscope=nothing, location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("ordering", ordering),]
    !isnothing(syncscope) && push!(attributes, namedattribute("syncscope", syncscope))

    return IR.create_operation(
        "llvm.fence",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`freeze`

"""
function freeze(val::Value; res=nothing::Union{Nothing,IR.Type}, location=Location())
    _results = IR.Type[]
    _operands = Value[val,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.freeze",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`getelementptr`

This operation mirrors LLVM IRs \'getelementptr\' operation that is used to
perform pointer arithmetic.

Like in LLVM IR, it is possible to use both constants as well as SSA values
as indices. In the case of indexing within a structure, it is required to
either use constant indices directly, or supply a constant SSA value.

An optional \'inbounds\' attribute specifies the low-level pointer arithmetic
overflow behavior that LLVM uses after lowering the operation to LLVM IR.

Examples:

```mlir
// GEP with an SSA value offset
%0 = llvm.getelementptr %1[%2] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>

// GEP with a constant offset and the inbounds attribute set
%0 = llvm.getelementptr inbounds %1[3] : (!llvm.ptr<f32>) -> !llvm.ptr<f32>

// GEP with constant offsets into a structure
%0 = llvm.getelementptr %1[0, 1]
   : (!llvm.ptr<struct(i32, f32)>) -> !llvm.ptr<f32>
```
"""
function getelementptr(
    base::Value,
    dynamicIndices::Vector{Value};
    res::IR.Type,
    rawConstantIndices,
    elem_type=nothing,
    inbounds=nothing,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[base, dynamicIndices...]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("rawConstantIndices", rawConstantIndices),]
    !isnothing(elem_type) && push!(attributes, namedattribute("elem_type", elem_type))
    !isnothing(inbounds) && push!(attributes, namedattribute("inbounds", inbounds))

    return IR.create_operation(
        "llvm.getelementptr",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mlir_global_ctors`

Specifies a list of constructor functions and priorities. The functions
referenced by this array will be called in ascending order of priority (i.e.
lowest first) when the module is loaded. The order of functions with the
same priority is not defined. This operation is translated to LLVM\'s
global_ctors global variable. The initializer functions are run at load
time. The `data` field present in LLVM\'s global_ctors variable is not
modeled here.

Examples:

```mlir
llvm.mlir.global_ctors {@ctor}

llvm.func @ctor() {
  ...
  llvm.return
}
```
"""
function mlir_global_ctors(; ctors, priorities, location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("ctors", ctors), namedattribute("priorities", priorities)
    ]

    return IR.create_operation(
        "llvm.mlir.global_ctors",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mlir_global_dtors`

Specifies a list of destructor functions and priorities. The functions
referenced by this array will be called in descending order of priority (i.e.
highest first) when the module is unloaded. The order of functions with the
same priority is not defined. This operation is translated to LLVM\'s
global_dtors global variable. The `data` field present in LLVM\'s
global_dtors variable is not modeled here.

Examples:

```mlir
llvm.func @dtor() {
  llvm.return
}
llvm.mlir.global_dtors {@dtor}
```
"""
function mlir_global_dtors(; dtors, priorities, location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("dtors", dtors), namedattribute("priorities", priorities)
    ]

    return IR.create_operation(
        "llvm.mlir.global_dtors",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mlir_global`

Since MLIR allows for arbitrary operations to be present at the top level,
global variables are defined using the `llvm.mlir.global` operation. Both
global constants and variables can be defined, and the value may also be
initialized in both cases.

There are two forms of initialization syntax. Simple constants that can be
represented as MLIR attributes can be given in-line:

```mlir
llvm.mlir.global @variable(32.0 : f32) : f32
```

This initialization and type syntax is similar to `llvm.mlir.constant` and
may use two types: one for MLIR attribute and another for the LLVM value.
These types must be compatible.

More complex constants that cannot be represented as MLIR attributes can be
given in an initializer region:

```mlir
// This global is initialized with the equivalent of:
//   i32* getelementptr (i32* @g2, i32 2)
llvm.mlir.global constant @int_gep() : !llvm.ptr<i32> {
  %0 = llvm.mlir.addressof @g2 : !llvm.ptr<i32>
  %1 = llvm.mlir.constant(2 : i32) : i32
  %2 = llvm.getelementptr %0[%1]
     : (!llvm.ptr<i32>, i32) -> !llvm.ptr<i32>
  // The initializer region must end with `llvm.return`.
  llvm.return %2 : !llvm.ptr<i32>
}
```

Only one of the initializer attribute or initializer region may be provided.

`llvm.mlir.global` must appear at top-level of the enclosing module. It uses
an @-identifier for its value, which will be uniqued by the module with
respect to other @-identifiers in it.

Examples:

```mlir
// Global values use @-identifiers.
llvm.mlir.global constant @cst(42 : i32) : i32

// Non-constant values must also be initialized.
llvm.mlir.global @variable(32.0 : f32) : f32

// Strings are expected to be of wrapped LLVM i8 array type and do not
// automatically include the trailing zero.
llvm.mlir.global @string(\"abc\") : !llvm.array<3 x i8>

// For strings globals, the trailing type may be omitted.
llvm.mlir.global constant @no_trailing_type(\"foo bar\")

// A complex initializer is constructed with an initializer region.
llvm.mlir.global constant @int_gep() : !llvm.ptr<i32> {
  %0 = llvm.mlir.addressof @g2 : !llvm.ptr<i32>
  %1 = llvm.mlir.constant(2 : i32) : i32
  %2 = llvm.getelementptr %0[%1]
     : (!llvm.ptr<i32>, i32) -> !llvm.ptr<i32>
  llvm.return %2 : !llvm.ptr<i32>
}
```

Similarly to functions, globals have a linkage attribute. In the custom
syntax, this attribute is placed between `llvm.mlir.global` and the optional
`constant` keyword. If the attribute is omitted, `external` linkage is
assumed by default.

Examples:

```mlir
// A constant with internal linkage will not participate in linking.
llvm.mlir.global internal constant @cst(42 : i32) : i32

// By default, \"external\" linkage is assumed and the global participates in
// symbol resolution at link-time.
llvm.mlir.global @glob(0 : f32) : f32

// Alignment is optional
llvm.mlir.global private constant @y(dense<1.0> : tensor<8xf32>) : !llvm.array<8 x f32>
```

Like global variables in LLVM IR, globals can have an (optional)
alignment attribute using keyword `alignment`. The integer value of the
alignment must be a positive integer that is a power of 2.

Examples:

```mlir
// Alignment is optional
llvm.mlir.global private constant @y(dense<1.0> : tensor<8xf32>) { alignment = 32 : i64 } : !llvm.array<8 x f32>
```
"""
function mlir_global(;
    global_type,
    constant=nothing,
    sym_name,
    linkage,
    dso_local=nothing,
    thread_local_=nothing,
    value=nothing,
    alignment=nothing,
    addr_space=nothing,
    unnamed_addr=nothing,
    section=nothing,
    comdat=nothing,
    visibility_=nothing,
    initializer::Region,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[initializer,]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("global_type", global_type),
        namedattribute("sym_name", sym_name),
        namedattribute("linkage", linkage),
    ]
    !isnothing(constant) && push!(attributes, namedattribute("constant", constant))
    !isnothing(dso_local) && push!(attributes, namedattribute("dso_local", dso_local))
    !isnothing(thread_local_) &&
        push!(attributes, namedattribute("thread_local_", thread_local_))
    !isnothing(value) && push!(attributes, namedattribute("value", value))
    !isnothing(alignment) && push!(attributes, namedattribute("alignment", alignment))
    !isnothing(addr_space) && push!(attributes, namedattribute("addr_space", addr_space))
    !isnothing(unnamed_addr) &&
        push!(attributes, namedattribute("unnamed_addr", unnamed_addr))
    !isnothing(section) && push!(attributes, namedattribute("section", section))
    !isnothing(comdat) && push!(attributes, namedattribute("comdat", comdat))
    !isnothing(visibility_) && push!(attributes, namedattribute("visibility_", visibility_))

    return IR.create_operation(
        "llvm.mlir.global",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`icmp`

"""
function icmp(lhs::Value, rhs::Value; res::IR.Type, predicate, location=Location())
    _results = IR.Type[res,]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("predicate", predicate),]

    return IR.create_operation(
        "llvm.icmp",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`inline_asm`

The InlineAsmOp mirrors the underlying LLVM semantics with a notable
exception: the embedded `asm_string` is not allowed to define or reference
any symbol or any global variable: only the operands of the op may be read,
written, or referenced.
Attempting to define or reference any symbol or any global behavior is
considered undefined behavior at this time.
"""
function inline_asm(
    operands::Vector{Value};
    res=nothing::Union{Nothing,IR.Type},
    asm_string,
    constraints,
    has_side_effects=nothing,
    is_align_stack=nothing,
    asm_dialect=nothing,
    operand_attrs=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[operands...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("asm_string", asm_string), namedattribute("constraints", constraints)
    ]
    !isnothing(res) && push!(_results, res)
    !isnothing(has_side_effects) &&
        push!(attributes, namedattribute("has_side_effects", has_side_effects))
    !isnothing(is_align_stack) &&
        push!(attributes, namedattribute("is_align_stack", is_align_stack))
    !isnothing(asm_dialect) && push!(attributes, namedattribute("asm_dialect", asm_dialect))
    !isnothing(operand_attrs) &&
        push!(attributes, namedattribute("operand_attrs", operand_attrs))

    return IR.create_operation(
        "llvm.inline_asm",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`insertelement`

"""
function insertelement(
    vector::Value,
    value::Value,
    position::Value;
    res=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[vector, value, position]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.insertelement",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`insertvalue`

"""
function insertvalue(
    container::Value,
    value::Value;
    res=nothing::Union{Nothing,IR.Type},
    position,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[container, value]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("position", position),]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.insertvalue",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`inttoptr`

"""
function inttoptr(arg::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[arg,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.inttoptr",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`invoke`

"""
function invoke(
    callee_operands::Vector{Value},
    normalDestOperands::Vector{Value},
    unwindDestOperands::Vector{Value};
    result_0::Vector{IR.Type},
    callee=nothing,
    branch_weights=nothing,
    normalDest::Block,
    unwindDest::Block,
    location=Location(),
)
    _results = IR.Type[result_0...,]
    _operands = Value[callee_operands..., normalDestOperands..., unwindDestOperands...]
    _owned_regions = Region[]
    _successors = Block[normalDest, unwindDest]
    _attributes = NamedAttribute[]
    push!(
        attributes,
        operandsegmentsizes([
            length(callee_operands), length(normalDestOperands), length(unwindDestOperands)
        ]),
    )
    !isnothing(callee) && push!(attributes, namedattribute("callee", callee))
    !isnothing(branch_weights) &&
        push!(attributes, namedattribute("branch_weights", branch_weights))

    return IR.create_operation(
        "llvm.invoke",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`func`

MLIR functions are defined by an operation that is not built into the IR
itself. The LLVM dialect provides an `llvm.func` operation to define
functions compatible with LLVM IR. These functions have LLVM dialect
function type but use MLIR syntax to express it. They are required to have
exactly one result type. LLVM function operation is intended to capture
additional properties of LLVM functions, such as linkage and calling
convention, that may be modeled differently by the built-in MLIR function.

```mlir
// The type of @bar is !llvm<\"i64 (i64)\">
llvm.func @bar(%arg0: i64) -> i64 {
  llvm.return %arg0 : i64
}

// Type type of @foo is !llvm<\"void (i64)\">
// !llvm.void type is omitted
llvm.func @foo(%arg0: i64) {
  llvm.return
}

// A function with `internal` linkage.
llvm.func internal @internal_func() {
  llvm.return
}
```
"""
function func(;
    sym_name,
    function_type,
    linkage=nothing,
    dso_local=nothing,
    CConv=nothing,
    comdat=nothing,
    personality=nothing,
    garbageCollector=nothing,
    passthrough=nothing,
    arg_attrs=nothing,
    res_attrs=nothing,
    function_entry_count=nothing,
    memory=nothing,
    visibility_=nothing,
    arm_streaming=nothing,
    arm_locally_streaming=nothing,
    section=nothing,
    unnamed_addr=nothing,
    alignment=nothing,
    body::Region,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[body,]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("sym_name", sym_name), namedattribute("function_type", function_type)
    ]
    !isnothing(linkage) && push!(attributes, namedattribute("linkage", linkage))
    !isnothing(dso_local) && push!(attributes, namedattribute("dso_local", dso_local))
    !isnothing(CConv) && push!(attributes, namedattribute("CConv", CConv))
    !isnothing(comdat) && push!(attributes, namedattribute("comdat", comdat))
    !isnothing(personality) && push!(attributes, namedattribute("personality", personality))
    !isnothing(garbageCollector) &&
        push!(attributes, namedattribute("garbageCollector", garbageCollector))
    !isnothing(passthrough) && push!(attributes, namedattribute("passthrough", passthrough))
    !isnothing(arg_attrs) && push!(attributes, namedattribute("arg_attrs", arg_attrs))
    !isnothing(res_attrs) && push!(attributes, namedattribute("res_attrs", res_attrs))
    !isnothing(function_entry_count) &&
        push!(attributes, namedattribute("function_entry_count", function_entry_count))
    !isnothing(memory) && push!(attributes, namedattribute("memory", memory))
    !isnothing(visibility_) && push!(attributes, namedattribute("visibility_", visibility_))
    !isnothing(arm_streaming) &&
        push!(attributes, namedattribute("arm_streaming", arm_streaming))
    !isnothing(arm_locally_streaming) &&
        push!(attributes, namedattribute("arm_locally_streaming", arm_locally_streaming))
    !isnothing(section) && push!(attributes, namedattribute("section", section))
    !isnothing(unnamed_addr) &&
        push!(attributes, namedattribute("unnamed_addr", unnamed_addr))
    !isnothing(alignment) && push!(attributes, namedattribute("alignment", alignment))

    return IR.create_operation(
        "llvm.func",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`lshr`

"""
function lshr(
    lhs::Value, rhs::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.lshr",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`landingpad`

"""
function landingpad(
    operand_0::Vector{Value}; res::IR.Type, cleanup=nothing, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[operand_0...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(cleanup) && push!(attributes, namedattribute("cleanup", cleanup))

    return IR.create_operation(
        "llvm.landingpad",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`load`

The `load` operation is used to read from memory. A load may be marked as
atomic, volatile, and/or nontemporal, and takes a number of optional
attributes that specify aliasing information.

An atomic load only supports a limited set of pointer, integer, and
floating point types, and requires an explicit alignment.

Examples:
```mlir
// A volatile load of a float variable.
%0 = llvm.load volatile %ptr : !llvm.ptr -> f32

// A nontemporal load of a float variable.
%0 = llvm.load %ptr {nontemporal} : !llvm.ptr -> f32

// An atomic load of an integer variable.
%0 = llvm.load %ptr atomic monotonic {alignment = 8 : i64}
    : !llvm.ptr -> i64
```

See the following link for more details:
https://llvm.org/docs/LangRef.html#load-instruction
"""
function load(
    addr::Value;
    res::IR.Type,
    alignment=nothing,
    volatile_=nothing,
    nontemporal=nothing,
    ordering=nothing,
    syncscope=nothing,
    access_groups=nothing,
    alias_scopes=nothing,
    noalias_scopes=nothing,
    tbaa=nothing,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[addr,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(alignment) && push!(attributes, namedattribute("alignment", alignment))
    !isnothing(volatile_) && push!(attributes, namedattribute("volatile_", volatile_))
    !isnothing(nontemporal) && push!(attributes, namedattribute("nontemporal", nontemporal))
    !isnothing(ordering) && push!(attributes, namedattribute("ordering", ordering))
    !isnothing(syncscope) && push!(attributes, namedattribute("syncscope", syncscope))
    !isnothing(access_groups) &&
        push!(attributes, namedattribute("access_groups", access_groups))
    !isnothing(alias_scopes) &&
        push!(attributes, namedattribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) &&
        push!(attributes, namedattribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, namedattribute("tbaa", tbaa))

    return IR.create_operation(
        "llvm.load",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`metadata`

llvm.metadata op defines one or more metadata nodes.

# Example
```mlir
llvm.metadata @metadata {
  llvm.access_group @group1
  llvm.access_group @group2
}
```
"""
function metadata(; sym_name, body::Region, location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[body,]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("sym_name", sym_name),]

    return IR.create_operation(
        "llvm.metadata",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mul`

"""
function mul(
    lhs::Value, rhs::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.mul",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`mlir_null`

Unlike LLVM IR, MLIR does not have first-class null pointers. They must be
explicitly created as SSA values using `llvm.mlir.null`. This operation has
no operands or attributes, and returns a null value of a wrapped LLVM IR
pointer type.

Examples:

```mlir
// Null pointer to i8.
%0 = llvm.mlir.null : !llvm.ptr<i8>

// Null pointer to a function with signature void().
%1 = llvm.mlir.null : !llvm.ptr<func<void ()>>
```
"""
function mlir_null(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.mlir.null",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`or`

"""
function or(
    lhs::Value, rhs::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.or",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`mlir_poison`

Unlike LLVM IR, MLIR does not have first-class poison values. Such values
must be created as SSA values using `llvm.mlir.poison`. This operation has
no operands or attributes. It creates a poison value of the specified LLVM
IR dialect type.

# Example

```mlir
// Create a poison value for a structure with a 32-bit integer followed
// by a float.
%0 = llvm.mlir.poison : !llvm.struct<(i32, f32)>
```
"""
function mlir_poison(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.mlir.poison",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`ptrtoint`

"""
function ptrtoint(arg::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[arg,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.ptrtoint",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`resume`

"""
function resume(value::Value; location=Location())
    _results = IR.Type[]
    _operands = Value[value,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.resume",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`return_`

"""
function return_(arg=nothing::Union{Nothing,Value}; location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(arg) && push!(operands, arg)

    return IR.create_operation(
        "llvm.return",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`sdiv`

"""
function sdiv(
    lhs::Value, rhs::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.sdiv",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`sext`

"""
function sext(arg::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[arg,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.sext",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`sitofp`

"""
function sitofp(arg::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[arg,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.sitofp",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`srem`

"""
function srem(
    lhs::Value, rhs::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.srem",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`select`

"""
function select(
    condition::Value,
    trueValue::Value,
    falseValue::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[condition, trueValue, falseValue]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.select",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`shl`

"""
function shl(
    lhs::Value, rhs::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.shl",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`shufflevector`

"""
function shufflevector(v1::Value, v2::Value; res::IR.Type, mask, location=Location())
    _results = IR.Type[res,]
    _operands = Value[v1, v2]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("mask", mask),]

    return IR.create_operation(
        "llvm.shufflevector",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`store`

The `store` operation is used to write to memory. A store may be marked as
atomic, volatile, and/or nontemporal, and takes a number of optional
attributes that specify aliasing information.

An atomic store only supports a limited set of pointer, integer, and
floating point types, and requires an explicit alignment.

Examples:
```mlir
// A volatile store of a float variable.
llvm.store volatile %val, %ptr : f32, !llvm.ptr

// A nontemporal store of a float variable.
llvm.store %val, %ptr {nontemporal} : f32, !llvm.ptr

// An atomic store of an integer variable.
llvm.store %val, %ptr atomic monotonic {alignment = 8 : i64}
    : i64, !llvm.ptr
```

See the following link for more details:
https://llvm.org/docs/LangRef.html#store-instruction
"""
function store(
    value::Value,
    addr::Value;
    alignment=nothing,
    volatile_=nothing,
    nontemporal=nothing,
    ordering=nothing,
    syncscope=nothing,
    access_groups=nothing,
    alias_scopes=nothing,
    noalias_scopes=nothing,
    tbaa=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[value, addr]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(alignment) && push!(attributes, namedattribute("alignment", alignment))
    !isnothing(volatile_) && push!(attributes, namedattribute("volatile_", volatile_))
    !isnothing(nontemporal) && push!(attributes, namedattribute("nontemporal", nontemporal))
    !isnothing(ordering) && push!(attributes, namedattribute("ordering", ordering))
    !isnothing(syncscope) && push!(attributes, namedattribute("syncscope", syncscope))
    !isnothing(access_groups) &&
        push!(attributes, namedattribute("access_groups", access_groups))
    !isnothing(alias_scopes) &&
        push!(attributes, namedattribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) &&
        push!(attributes, namedattribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, namedattribute("tbaa", tbaa))

    return IR.create_operation(
        "llvm.store",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`sub`

"""
function sub(
    lhs::Value, rhs::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.sub",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`switch`

"""
function switch(
    value::Value,
    defaultOperands::Vector{Value},
    caseOperands::Vector{Value};
    case_values=nothing,
    case_operand_segments,
    branch_weights=nothing,
    defaultDestination::Block,
    caseDestinations::Vector{Block},
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[value, defaultOperands..., caseOperands...]
    _owned_regions = Region[]
    _successors = Block[defaultDestination, caseDestinations...]
    _attributes = NamedAttribute[namedattribute(
        "case_operand_segments", case_operand_segments
    ),]
    push!(
        attributes, operandsegmentsizes([1, length(defaultOperands), length(caseOperands)])
    )
    !isnothing(case_values) && push!(attributes, namedattribute("case_values", case_values))
    !isnothing(branch_weights) &&
        push!(attributes, namedattribute("branch_weights", branch_weights))

    return IR.create_operation(
        "llvm.switch",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`trunc`

"""
function trunc(arg::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[arg,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.trunc",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`udiv`

"""
function udiv(
    lhs::Value, rhs::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.udiv",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`uitofp`

"""
function uitofp(arg::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[arg,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.uitofp",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`urem`

"""
function urem(
    lhs::Value, rhs::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.urem",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`mlir_undef`

Unlike LLVM IR, MLIR does not have first-class undefined values. Such values
must be created as SSA values using `llvm.mlir.undef`. This operation has no
operands or attributes. It creates an undefined value of the specified LLVM
IR dialect type.

# Example

```mlir
// Create a structure with a 32-bit integer followed by a float.
%0 = llvm.mlir.undef : !llvm.struct<(i32, f32)>
```
"""
function mlir_undef(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.mlir.undef",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`unreachable`

"""
function unreachable(; location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.unreachable",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`xor`

"""
function xor(
    lhs::Value, rhs::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.xor",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`zext`

"""
function zext(arg::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[arg,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.zext",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

import ...IR:
    IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes

"""
`intr_abs`

"""
function intr_abs(in::Value; res::IR.Type, is_int_min_poison, location=Location())
    _results = IR.Type[res,]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("is_int_min_poison", is_int_min_poison),]

    return IR.create_operation(
        "llvm.intr.abs",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_annotation`

"""
function intr_annotation(
    integer::Value,
    annotation::Value,
    fileName::Value,
    line::Value;
    res=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[integer, annotation, fileName, line]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.intr.annotation",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_assume`

"""
function intr_assume(cond::Value; location=Location())
    _results = IR.Type[]
    _operands = Value[cond,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.assume",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_bitreverse`

"""
function intr_bitreverse(
    in::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.intr.bitreverse",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_bswap`

"""
function intr_bswap(in::Value; res=nothing::Union{Nothing,IR.Type}, location=Location())
    _results = IR.Type[]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.intr.bswap",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`call_intrinsic`

Call the specified llvm intrinsic. If the intrinsic is overloaded, use
the MLIR function type of this op to determine which intrinsic to call.
"""
function call_intrinsic(
    args::Vector{Value};
    results::Vector{IR.Type},
    intrin,
    fastmathFlags=nothing,
    location=Location(),
)
    _results = IR.Type[results...,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("intrin", intrin),]
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.call_intrinsic",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_copysign`

"""
function intr_copysign(
    a::Value,
    b::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[a, b]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.intr.copysign",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_coro_align`

"""
function intr_coro_align(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.coro.align",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_coro_begin`

"""
function intr_coro_begin(token::Value, mem::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[token, mem]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.coro.begin",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_coro_end`

"""
function intr_coro_end(handle::Value, unwind::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[handle, unwind]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.coro.end",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_coro_free`

"""
function intr_coro_free(id::Value, handle::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[id, handle]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.coro.free",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_coro_id`

"""
function intr_coro_id(
    align::Value,
    promise::Value,
    coroaddr::Value,
    fnaddrs::Value;
    res::IR.Type,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[align, promise, coroaddr, fnaddrs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.coro.id",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_coro_resume`

"""
function intr_coro_resume(handle::Value; location=Location())
    _results = IR.Type[]
    _operands = Value[handle,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.coro.resume",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_coro_save`

"""
function intr_coro_save(handle::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[handle,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.coro.save",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_coro_size`

"""
function intr_coro_size(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.coro.size",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_coro_suspend`

"""
function intr_coro_suspend(save::Value, final::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[save, final]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.coro.suspend",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_cos`

"""
function intr_cos(
    in::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.intr.cos",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_ctlz`

"""
function intr_ctlz(in::Value; res::IR.Type, is_zero_poison, location=Location())
    _results = IR.Type[res,]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("is_zero_poison", is_zero_poison),]

    return IR.create_operation(
        "llvm.intr.ctlz",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_cttz`

"""
function intr_cttz(in::Value; res::IR.Type, is_zero_poison, location=Location())
    _results = IR.Type[res,]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("is_zero_poison", is_zero_poison),]

    return IR.create_operation(
        "llvm.intr.cttz",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_ctpop`

"""
function intr_ctpop(in::Value; res=nothing::Union{Nothing,IR.Type}, location=Location())
    _results = IR.Type[]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.intr.ctpop",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_dbg_declare`

"""
function intr_dbg_declare(addr::Value; varInfo, location=Location())
    _results = IR.Type[]
    _operands = Value[addr,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("varInfo", varInfo),]

    return IR.create_operation(
        "llvm.intr.dbg.declare",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_dbg_label`

"""
function intr_dbg_label(; label, location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("label", label),]

    return IR.create_operation(
        "llvm.intr.dbg.label",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_dbg_value`

"""
function intr_dbg_value(value::Value; varInfo, location=Location())
    _results = IR.Type[]
    _operands = Value[value,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("varInfo", varInfo),]

    return IR.create_operation(
        "llvm.intr.dbg.value",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_debugtrap`

"""
function intr_debugtrap(; location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.debugtrap",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_eh_typeid_for`

"""
function intr_eh_typeid_for(type_info::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[type_info,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.eh.typeid.for",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_exp2`

"""
function intr_exp2(
    in::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.intr.exp2",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_exp`

"""
function intr_exp(
    in::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.intr.exp",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_expect`

"""
function intr_expect(
    val::Value, expected::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[val, expected]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.intr.expect",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_expect_with_probability`

"""
function intr_expect_with_probability(
    val::Value,
    expected::Value;
    res=nothing::Union{Nothing,IR.Type},
    prob,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[val, expected]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("prob", prob),]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.intr.expect.with.probability",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_fabs`

"""
function intr_fabs(
    in::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.intr.fabs",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_ceil`

"""
function intr_ceil(
    in::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.intr.ceil",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_floor`

"""
function intr_floor(
    in::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.intr.floor",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_fma`

"""
function intr_fma(
    a::Value,
    b::Value,
    c::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[a, b, c]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.intr.fma",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_fmuladd`

"""
function intr_fmuladd(
    a::Value,
    b::Value,
    c::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[a, b, c]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.intr.fmuladd",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_trunc`

"""
function intr_trunc(
    in::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.intr.trunc",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_fshl`

"""
function intr_fshl(
    a::Value, b::Value, c::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[a, b, c]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.intr.fshl",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_fshr`

"""
function intr_fshr(
    a::Value, b::Value, c::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[a, b, c]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.intr.fshr",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_get_active_lane_mask`

"""
function intr_get_active_lane_mask(base::Value, n::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[base, n]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.get.active.lane.mask",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_is_constant`

"""
function intr_is_constant(
    val::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[val,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.intr.is.constant",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_is_fpclass`

"""
function intr_is_fpclass(in::Value; res::IR.Type, bit, location=Location())
    _results = IR.Type[res,]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("bit", bit),]

    return IR.create_operation(
        "llvm.intr.is.fpclass",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_lifetime_end`

"""
function intr_lifetime_end(ptr::Value; size, location=Location())
    _results = IR.Type[]
    _operands = Value[ptr,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("size", size),]

    return IR.create_operation(
        "llvm.intr.lifetime.end",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_lifetime_start`

"""
function intr_lifetime_start(ptr::Value; size, location=Location())
    _results = IR.Type[]
    _operands = Value[ptr,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("size", size),]

    return IR.create_operation(
        "llvm.intr.lifetime.start",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_llrint`

"""
function intr_llrint(val::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[val,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.llrint",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_llround`

"""
function intr_llround(val::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[val,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.llround",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_log10`

"""
function intr_log10(
    in::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.intr.log10",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_log2`

"""
function intr_log2(
    in::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.intr.log2",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_log`

"""
function intr_log(
    in::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.intr.log",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_lrint`

"""
function intr_lrint(val::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[val,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.lrint",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_lround`

"""
function intr_lround(val::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[val,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.lround",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_masked_load`

"""
function intr_masked_load(
    data::Value,
    mask::Value,
    pass_thru::Vector{Value};
    res::IR.Type,
    alignment,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[data, mask, pass_thru...]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("alignment", alignment),]

    return IR.create_operation(
        "llvm.intr.masked.load",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_masked_store`

"""
function intr_masked_store(
    value::Value, data::Value, mask::Value; alignment, location=Location()
)
    _results = IR.Type[]
    _operands = Value[value, data, mask]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("alignment", alignment),]

    return IR.create_operation(
        "llvm.intr.masked.store",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_matrix_column_major_load`

"""
function intr_matrix_column_major_load(
    data::Value, stride::Value; res::IR.Type, isVolatile, rows, columns, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[data, stride]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("isVolatile", isVolatile),
        namedattribute("rows", rows),
        namedattribute("columns", columns),
    ]

    return IR.create_operation(
        "llvm.intr.matrix.column.major.load",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_matrix_column_major_store`

"""
function intr_matrix_column_major_store(
    matrix::Value,
    data::Value,
    stride::Value;
    isVolatile,
    rows,
    columns,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[matrix, data, stride]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("isVolatile", isVolatile),
        namedattribute("rows", rows),
        namedattribute("columns", columns),
    ]

    return IR.create_operation(
        "llvm.intr.matrix.column.major.store",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_matrix_multiply`

"""
function intr_matrix_multiply(
    lhs::Value,
    rhs::Value;
    res::IR.Type,
    lhs_rows,
    lhs_columns,
    rhs_columns,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[lhs, rhs]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("lhs_rows", lhs_rows),
        namedattribute("lhs_columns", lhs_columns),
        namedattribute("rhs_columns", rhs_columns),
    ]

    return IR.create_operation(
        "llvm.intr.matrix.multiply",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_matrix_transpose`

"""
function intr_matrix_transpose(
    matrix::Value; res::IR.Type, rows, columns, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[matrix,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("rows", rows), namedattribute("columns", columns)
    ]

    return IR.create_operation(
        "llvm.intr.matrix.transpose",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_maxnum`

"""
function intr_maxnum(
    a::Value,
    b::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[a, b]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.intr.maxnum",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_maximum`

"""
function intr_maximum(
    a::Value,
    b::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[a, b]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.intr.maximum",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_memcpy_inline`

"""
function intr_memcpy_inline(
    dst::Value,
    src::Value;
    len,
    isVolatile,
    access_groups=nothing,
    alias_scopes=nothing,
    noalias_scopes=nothing,
    tbaa=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[dst, src]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("len", len), namedattribute("isVolatile", isVolatile)
    ]
    !isnothing(access_groups) &&
        push!(attributes, namedattribute("access_groups", access_groups))
    !isnothing(alias_scopes) &&
        push!(attributes, namedattribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) &&
        push!(attributes, namedattribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, namedattribute("tbaa", tbaa))

    return IR.create_operation(
        "llvm.intr.memcpy.inline",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_memcpy`

"""
function intr_memcpy(
    dst::Value,
    src::Value,
    len::Value;
    isVolatile,
    access_groups=nothing,
    alias_scopes=nothing,
    noalias_scopes=nothing,
    tbaa=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[dst, src, len]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("isVolatile", isVolatile),]
    !isnothing(access_groups) &&
        push!(attributes, namedattribute("access_groups", access_groups))
    !isnothing(alias_scopes) &&
        push!(attributes, namedattribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) &&
        push!(attributes, namedattribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, namedattribute("tbaa", tbaa))

    return IR.create_operation(
        "llvm.intr.memcpy",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_memmove`

"""
function intr_memmove(
    dst::Value,
    src::Value,
    len::Value;
    isVolatile,
    access_groups=nothing,
    alias_scopes=nothing,
    noalias_scopes=nothing,
    tbaa=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[dst, src, len]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("isVolatile", isVolatile),]
    !isnothing(access_groups) &&
        push!(attributes, namedattribute("access_groups", access_groups))
    !isnothing(alias_scopes) &&
        push!(attributes, namedattribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) &&
        push!(attributes, namedattribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, namedattribute("tbaa", tbaa))

    return IR.create_operation(
        "llvm.intr.memmove",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_memset`

"""
function intr_memset(
    dst::Value,
    val::Value,
    len::Value;
    isVolatile,
    access_groups=nothing,
    alias_scopes=nothing,
    noalias_scopes=nothing,
    tbaa=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[dst, val, len]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("isVolatile", isVolatile),]
    !isnothing(access_groups) &&
        push!(attributes, namedattribute("access_groups", access_groups))
    !isnothing(alias_scopes) &&
        push!(attributes, namedattribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) &&
        push!(attributes, namedattribute("noalias_scopes", noalias_scopes))
    !isnothing(tbaa) && push!(attributes, namedattribute("tbaa", tbaa))

    return IR.create_operation(
        "llvm.intr.memset",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_minnum`

"""
function intr_minnum(
    a::Value,
    b::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[a, b]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.intr.minnum",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_minimum`

"""
function intr_minimum(
    a::Value,
    b::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[a, b]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.intr.minimum",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_nearbyint`

"""
function intr_nearbyint(
    in::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.intr.nearbyint",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_experimental_noalias_scope_decl`

"""
function intr_experimental_noalias_scope_decl(; scope, location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("scope", scope),]

    return IR.create_operation(
        "llvm.intr.experimental.noalias.scope.decl",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_powi`

"""
function intr_powi(
    val::Value, power::Value; res::IR.Type, fastmathFlags=nothing, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[val, power]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.intr.powi",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_pow`

"""
function intr_pow(
    a::Value,
    b::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[a, b]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.intr.pow",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_prefetch`

"""
function intr_prefetch(addr::Value; rw, hint, cache, location=Location())
    _results = IR.Type[]
    _operands = Value[addr,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("rw", rw),
        namedattribute("hint", hint),
        namedattribute("cache", cache),
    ]

    return IR.create_operation(
        "llvm.intr.prefetch",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_ptr_annotation`

"""
function intr_ptr_annotation(
    ptr::Value,
    annotation::Value,
    fileName::Value,
    line::Value,
    attr::Value;
    res=nothing::Union{Nothing,IR.Type},
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[ptr, annotation, fileName, line, attr]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.intr.ptr.annotation",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_rint`

"""
function intr_rint(
    in::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.intr.rint",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_roundeven`

"""
function intr_roundeven(
    in::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.intr.roundeven",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_round`

"""
function intr_round(
    in::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.intr.round",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_sadd_sat`

"""
function intr_sadd_sat(
    a::Value, b::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[a, b]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.intr.sadd.sat",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_sadd_with_overflow`

"""
function intr_sadd_with_overflow(
    operand_0::Value, operand_1::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[operand_0, operand_1]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.sadd.with.overflow",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_smax`

"""
function intr_smax(
    a::Value, b::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[a, b]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.intr.smax",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_smin`

"""
function intr_smin(
    a::Value, b::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[a, b]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.intr.smin",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_smul_with_overflow`

"""
function intr_smul_with_overflow(
    operand_0::Value, operand_1::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[operand_0, operand_1]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.smul.with.overflow",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_ssa_copy`

"""
function intr_ssa_copy(
    operand::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[operand,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.intr.ssa.copy",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_sshl_sat`

"""
function intr_sshl_sat(
    a::Value, b::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[a, b]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.intr.sshl.sat",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_ssub_sat`

"""
function intr_ssub_sat(
    a::Value, b::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[a, b]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.intr.ssub.sat",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_ssub_with_overflow`

"""
function intr_ssub_with_overflow(
    operand_0::Value, operand_1::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[operand_0, operand_1]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.ssub.with.overflow",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_sin`

"""
function intr_sin(
    in::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.intr.sin",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_sqrt`

"""
function intr_sqrt(
    in::Value;
    res=nothing::Union{Nothing,IR.Type},
    fastmathFlags=nothing,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.intr.sqrt",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_stackrestore`

"""
function intr_stackrestore(ptr::Value; location=Location())
    _results = IR.Type[]
    _operands = Value[ptr,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.stackrestore",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_stacksave`

"""
function intr_stacksave(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.stacksave",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_experimental_stepvector`

"""
function intr_experimental_stepvector(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.experimental.stepvector",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_threadlocal_address`

"""
function intr_threadlocal_address(global_::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[global_,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.threadlocal.address",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_trap`

"""
function intr_trap(; location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.trap",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_uadd_sat`

"""
function intr_uadd_sat(
    a::Value, b::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[a, b]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.intr.uadd.sat",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_uadd_with_overflow`

"""
function intr_uadd_with_overflow(
    operand_0::Value, operand_1::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[operand_0, operand_1]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.uadd.with.overflow",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_ubsantrap`

"""
function intr_ubsantrap(; failureKind, location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("failureKind", failureKind),]

    return IR.create_operation(
        "llvm.intr.ubsantrap",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_umax`

"""
function intr_umax(
    a::Value, b::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[a, b]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.intr.umax",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_umin`

"""
function intr_umin(
    a::Value, b::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[a, b]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.intr.umin",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_umul_with_overflow`

"""
function intr_umul_with_overflow(
    operand_0::Value, operand_1::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[operand_0, operand_1]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.umul.with.overflow",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_ushl_sat`

"""
function intr_ushl_sat(
    a::Value, b::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[a, b]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.intr.ushl.sat",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_usub_sat`

"""
function intr_usub_sat(
    a::Value, b::Value; res=nothing::Union{Nothing,IR.Type}, location=Location()
)
    _results = IR.Type[]
    _operands = Value[a, b]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.intr.usub.sat",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_usub_with_overflow`

"""
function intr_usub_with_overflow(
    operand_0::Value, operand_1::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[operand_0, operand_1]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.usub.with.overflow",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_ashr`

"""
function intr_vp_ashr(
    lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[lhs, rhs, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.ashr",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_add`

"""
function intr_vp_add(
    lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[lhs, rhs, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.add",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_and`

"""
function intr_vp_and(
    lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[lhs, rhs, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.and",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_fadd`

"""
function intr_vp_fadd(
    lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[lhs, rhs, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.fadd",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_fdiv`

"""
function intr_vp_fdiv(
    lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[lhs, rhs, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.fdiv",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_fmuladd`

"""
function intr_vp_fmuladd(
    op1::Value,
    op2::Value,
    op3::Value,
    mask::Value,
    evl::Value;
    res::IR.Type,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[op1, op2, op3, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.fmuladd",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_fmul`

"""
function intr_vp_fmul(
    lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[lhs, rhs, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.fmul",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_fneg`

"""
function intr_vp_fneg(op::Value, mask::Value, evl::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[op, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.fneg",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_fpext`

"""
function intr_vp_fpext(
    src::Value, mask::Value, evl::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[src, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.fpext",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_fptosi`

"""
function intr_vp_fptosi(
    src::Value, mask::Value, evl::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[src, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.fptosi",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_fptoui`

"""
function intr_vp_fptoui(
    src::Value, mask::Value, evl::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[src, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.fptoui",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_fptrunc`

"""
function intr_vp_fptrunc(
    src::Value, mask::Value, evl::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[src, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.fptrunc",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_frem`

"""
function intr_vp_frem(
    lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[lhs, rhs, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.frem",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_fsub`

"""
function intr_vp_fsub(
    lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[lhs, rhs, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.fsub",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_fma`

"""
function intr_vp_fma(
    op1::Value,
    op2::Value,
    op3::Value,
    mask::Value,
    evl::Value;
    res::IR.Type,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[op1, op2, op3, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.fma",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_inttoptr`

"""
function intr_vp_inttoptr(
    src::Value, mask::Value, evl::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[src, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.inttoptr",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_lshr`

"""
function intr_vp_lshr(
    lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[lhs, rhs, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.lshr",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_load`

"""
function intr_vp_load(
    ptr::Value, mask::Value, evl::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[ptr, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.load",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_merge`

"""
function intr_vp_merge(
    cond::Value,
    true_val::Value,
    false_val::Value,
    evl::Value;
    res::IR.Type,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[cond, true_val, false_val, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.merge",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_mul`

"""
function intr_vp_mul(
    lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[lhs, rhs, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.mul",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_or`

"""
function intr_vp_or(
    lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[lhs, rhs, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.or",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_ptrtoint`

"""
function intr_vp_ptrtoint(
    src::Value, mask::Value, evl::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[src, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.ptrtoint",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_reduce_add`

"""
function intr_vp_reduce_add(
    satrt_value::Value,
    val::Value,
    mask::Value,
    evl::Value;
    res::IR.Type,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[satrt_value, val, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.reduce.add",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_reduce_and`

"""
function intr_vp_reduce_and(
    satrt_value::Value,
    val::Value,
    mask::Value,
    evl::Value;
    res::IR.Type,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[satrt_value, val, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.reduce.and",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_reduce_fadd`

"""
function intr_vp_reduce_fadd(
    satrt_value::Value,
    val::Value,
    mask::Value,
    evl::Value;
    res::IR.Type,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[satrt_value, val, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.reduce.fadd",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_reduce_fmax`

"""
function intr_vp_reduce_fmax(
    satrt_value::Value,
    val::Value,
    mask::Value,
    evl::Value;
    res::IR.Type,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[satrt_value, val, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.reduce.fmax",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_reduce_fmin`

"""
function intr_vp_reduce_fmin(
    satrt_value::Value,
    val::Value,
    mask::Value,
    evl::Value;
    res::IR.Type,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[satrt_value, val, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.reduce.fmin",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_reduce_fmul`

"""
function intr_vp_reduce_fmul(
    satrt_value::Value,
    val::Value,
    mask::Value,
    evl::Value;
    res::IR.Type,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[satrt_value, val, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.reduce.fmul",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_reduce_mul`

"""
function intr_vp_reduce_mul(
    satrt_value::Value,
    val::Value,
    mask::Value,
    evl::Value;
    res::IR.Type,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[satrt_value, val, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.reduce.mul",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_reduce_or`

"""
function intr_vp_reduce_or(
    satrt_value::Value,
    val::Value,
    mask::Value,
    evl::Value;
    res::IR.Type,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[satrt_value, val, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.reduce.or",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_reduce_smax`

"""
function intr_vp_reduce_smax(
    satrt_value::Value,
    val::Value,
    mask::Value,
    evl::Value;
    res::IR.Type,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[satrt_value, val, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.reduce.smax",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_reduce_smin`

"""
function intr_vp_reduce_smin(
    satrt_value::Value,
    val::Value,
    mask::Value,
    evl::Value;
    res::IR.Type,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[satrt_value, val, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.reduce.smin",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_reduce_umax`

"""
function intr_vp_reduce_umax(
    satrt_value::Value,
    val::Value,
    mask::Value,
    evl::Value;
    res::IR.Type,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[satrt_value, val, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.reduce.umax",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_reduce_umin`

"""
function intr_vp_reduce_umin(
    satrt_value::Value,
    val::Value,
    mask::Value,
    evl::Value;
    res::IR.Type,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[satrt_value, val, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.reduce.umin",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_reduce_xor`

"""
function intr_vp_reduce_xor(
    satrt_value::Value,
    val::Value,
    mask::Value,
    evl::Value;
    res::IR.Type,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[satrt_value, val, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.reduce.xor",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_sdiv`

"""
function intr_vp_sdiv(
    lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[lhs, rhs, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.sdiv",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_sext`

"""
function intr_vp_sext(
    src::Value, mask::Value, evl::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[src, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.sext",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_sitofp`

"""
function intr_vp_sitofp(
    src::Value, mask::Value, evl::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[src, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.sitofp",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_srem`

"""
function intr_vp_srem(
    lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[lhs, rhs, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.srem",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_select`

"""
function intr_vp_select(
    cond::Value,
    true_val::Value,
    false_val::Value,
    evl::Value;
    res::IR.Type,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[cond, true_val, false_val, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.select",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_shl`

"""
function intr_vp_shl(
    lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[lhs, rhs, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.shl",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_store`

"""
function intr_vp_store(val::Value, ptr::Value, mask::Value, evl::Value; location=Location())
    _results = IR.Type[]
    _operands = Value[val, ptr, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.store",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_experimental_vp_strided_load`

"""
function intr_experimental_vp_strided_load(
    ptr::Value, stride::Value, mask::Value, evl::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[ptr, stride, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.experimental.vp.strided.load",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_experimental_vp_strided_store`

"""
function intr_experimental_vp_strided_store(
    val::Value, ptr::Value, stride::Value, mask::Value, evl::Value; location=Location()
)
    _results = IR.Type[]
    _operands = Value[val, ptr, stride, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.experimental.vp.strided.store",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_sub`

"""
function intr_vp_sub(
    lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[lhs, rhs, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.sub",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_trunc`

"""
function intr_vp_trunc(
    src::Value, mask::Value, evl::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[src, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.trunc",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_udiv`

"""
function intr_vp_udiv(
    lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[lhs, rhs, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.udiv",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_uitofp`

"""
function intr_vp_uitofp(
    src::Value, mask::Value, evl::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[src, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.uitofp",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_urem`

"""
function intr_vp_urem(
    lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[lhs, rhs, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.urem",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_xor`

"""
function intr_vp_xor(
    lhs::Value, rhs::Value, mask::Value, evl::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[lhs, rhs, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.xor",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vp_zext`

"""
function intr_vp_zext(
    src::Value, mask::Value, evl::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[src, mask, evl]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vp.zext",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vacopy`

"""
function intr_vacopy(dest_list::Value, src_list::Value; location=Location())
    _results = IR.Type[]
    _operands = Value[dest_list, src_list]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vacopy",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vaend`

"""
function intr_vaend(arg_list::Value; location=Location())
    _results = IR.Type[]
    _operands = Value[arg_list,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vaend",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vastart`

"""
function intr_vastart(arg_list::Value; location=Location())
    _results = IR.Type[]
    _operands = Value[arg_list,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vastart",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_var_annotation`

"""
function intr_var_annotation(
    val::Value,
    annotation::Value,
    fileName::Value,
    line::Value,
    attr::Value;
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[val, annotation, fileName, line, attr]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.var.annotation",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_masked_compressstore`

"""
function intr_masked_compressstore(
    operand_0::Value, operand_1::Value, operand_2::Value; location=Location()
)
    _results = IR.Type[]
    _operands = Value[operand_0, operand_1, operand_2]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.masked.compressstore",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_masked_expandload`

"""
function intr_masked_expandload(
    operand_0::Value, operand_1::Value, operand_2::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[operand_0, operand_1, operand_2]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.masked.expandload",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_masked_gather`

"""
function intr_masked_gather(
    ptrs::Value,
    mask::Value,
    pass_thru::Vector{Value};
    res::IR.Type,
    alignment,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[ptrs, mask, pass_thru...]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("alignment", alignment),]

    return IR.create_operation(
        "llvm.intr.masked.gather",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_masked_scatter`

"""
function intr_masked_scatter(
    value::Value, ptrs::Value, mask::Value; alignment, location=Location()
)
    _results = IR.Type[]
    _operands = Value[value, ptrs, mask]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("alignment", alignment),]

    return IR.create_operation(
        "llvm.intr.masked.scatter",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vector_extract`

"""
function intr_vector_extract(srcvec::Value; res::IR.Type, pos, location=Location())
    _results = IR.Type[res,]
    _operands = Value[srcvec,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("pos", pos),]

    return IR.create_operation(
        "llvm.intr.vector.extract",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vector_insert`

"""
function intr_vector_insert(
    srcvec::Value,
    dstvec::Value;
    res=nothing::Union{Nothing,IR.Type},
    pos,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[srcvec, dstvec]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("pos", pos),]
    !isnothing(res) && push!(_results, res)

    return IR.create_operation(
        "llvm.intr.vector.insert",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=(length(_results) == 0 ? nothing : _results),
        result_inference=(length(_results) == 0 ? true : false),
    )
end

"""
`intr_vector_reduce_add`

"""
function intr_vector_reduce_add(in::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vector.reduce.add",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vector_reduce_and`

"""
function intr_vector_reduce_and(in::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vector.reduce.and",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vector_reduce_fadd`

"""
function intr_vector_reduce_fadd(
    start_value::Value, input::Value; res::IR.Type, reassoc=nothing, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[start_value, input]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(reassoc) && push!(attributes, namedattribute("reassoc", reassoc))

    return IR.create_operation(
        "llvm.intr.vector.reduce.fadd",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vector_reduce_fmax`

"""
function intr_vector_reduce_fmax(
    in::Value; res::IR.Type, fastmathFlags=nothing, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.intr.vector.reduce.fmax",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vector_reduce_fmaximum`

"""
function intr_vector_reduce_fmaximum(
    in::Value; res::IR.Type, fastmathFlags=nothing, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.intr.vector.reduce.fmaximum",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vector_reduce_fmin`

"""
function intr_vector_reduce_fmin(
    in::Value; res::IR.Type, fastmathFlags=nothing, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.intr.vector.reduce.fmin",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vector_reduce_fminimum`

"""
function intr_vector_reduce_fminimum(
    in::Value; res::IR.Type, fastmathFlags=nothing, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(fastmathFlags) &&
        push!(attributes, namedattribute("fastmathFlags", fastmathFlags))

    return IR.create_operation(
        "llvm.intr.vector.reduce.fminimum",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vector_reduce_fmul`

"""
function intr_vector_reduce_fmul(
    start_value::Value, input::Value; res::IR.Type, reassoc=nothing, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[start_value, input]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(reassoc) && push!(attributes, namedattribute("reassoc", reassoc))

    return IR.create_operation(
        "llvm.intr.vector.reduce.fmul",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vector_reduce_mul`

"""
function intr_vector_reduce_mul(in::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vector.reduce.mul",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vector_reduce_or`

"""
function intr_vector_reduce_or(in::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vector.reduce.or",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vector_reduce_smax`

"""
function intr_vector_reduce_smax(in::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vector.reduce.smax",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vector_reduce_smin`

"""
function intr_vector_reduce_smin(in::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vector.reduce.smin",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vector_reduce_umax`

"""
function intr_vector_reduce_umax(in::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vector.reduce.umax",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vector_reduce_umin`

"""
function intr_vector_reduce_umin(in::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vector.reduce.umin",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vector_reduce_xor`

"""
function intr_vector_reduce_xor(in::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[in,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vector.reduce.xor",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`intr_vscale`

"""
function intr_vscale(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "llvm.intr.vscale",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

import ...IR:
    IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes

"""
`barrier0`

"""
function barrier0(; location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.barrier0",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`read_ptx_sreg_ntid_x`

"""
function read_ptx_sreg_ntid_x(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.read.ptx.sreg.ntid.x",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`read_ptx_sreg_ntid_y`

"""
function read_ptx_sreg_ntid_y(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.read.ptx.sreg.ntid.y",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`read_ptx_sreg_ntid_z`

"""
function read_ptx_sreg_ntid_z(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.read.ptx.sreg.ntid.z",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`read_ptx_sreg_ctaid_x`

"""
function read_ptx_sreg_ctaid_x(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.read.ptx.sreg.ctaid.x",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`read_ptx_sreg_ctaid_y`

"""
function read_ptx_sreg_ctaid_y(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.read.ptx.sreg.ctaid.y",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`read_ptx_sreg_ctaid_z`

"""
function read_ptx_sreg_ctaid_z(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.read.ptx.sreg.ctaid.z",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`cp_async_bulk_tensor_shared_cluster_global`

"""
function cp_async_bulk_tensor_shared_cluster_global(
    dstMem::Value,
    tmaDescriptor::Value,
    mbar::Value,
    coordinates::Vector{Value};
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[dstMem, tmaDescriptor, mbar, coordinates...]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.cp.async.bulk.tensor.shared.cluster.global",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`cp_async_commit_group`

"""
function cp_async_commit_group(; location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.cp.async.commit.group",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`cp_async_shared_global`

"""
function cp_async_shared_global(
    dst::Value,
    src::Value,
    cpSize=nothing::Union{Nothing,Value};
    size,
    modifier,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[dst, src]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("size", size), namedattribute("modifier", modifier)
    ]
    !isnothing(cpSize) && push!(operands, cpSize)

    return IR.create_operation(
        "nvvm.cp.async.shared.global",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`cp_async_wait_group`

"""
function cp_async_wait_group(; n, location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("n", n),]

    return IR.create_operation(
        "nvvm.cp.async.wait.group",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`read_ptx_sreg_nctaid_x`

"""
function read_ptx_sreg_nctaid_x(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.read.ptx.sreg.nctaid.x",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`read_ptx_sreg_nctaid_y`

"""
function read_ptx_sreg_nctaid_y(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.read.ptx.sreg.nctaid.y",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`read_ptx_sreg_nctaid_z`

"""
function read_ptx_sreg_nctaid_z(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.read.ptx.sreg.nctaid.z",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`read_ptx_sreg_laneid`

"""
function read_ptx_sreg_laneid(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.read.ptx.sreg.laneid",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`ldmatrix`

"""
function ldmatrix(ptr::Value; res::IR.Type, num, layout, location=Location())
    _results = IR.Type[res,]
    _operands = Value[ptr,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("num", num), namedattribute("layout", layout)
    ]

    return IR.create_operation(
        "nvvm.ldmatrix",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mbarrier_arrive_expect_tx`

"""
function mbarrier_arrive_expect_tx(addr::Value, txcount::Value; location=Location())
    _results = IR.Type[]
    _operands = Value[addr, txcount]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.mbarrier.arrive.expect_tx",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mbarrier_arrive_expect_tx_shared`

"""
function mbarrier_arrive_expect_tx_shared(addr::Value, txcount::Value; location=Location())
    _results = IR.Type[]
    _operands = Value[addr, txcount]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.mbarrier.arrive.expect_tx.shared",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mbarrier_arrive_nocomplete`

"""
function mbarrier_arrive_nocomplete(
    addr::Value, count::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[addr, count]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.mbarrier.arrive.nocomplete",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mbarrier_arrive_nocomplete_shared`

"""
function mbarrier_arrive_nocomplete_shared(
    addr::Value, count::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[addr, count]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.mbarrier.arrive.nocomplete.shared",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mbarrier_arrive`

"""
function mbarrier_arrive(addr::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[addr,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.mbarrier.arrive",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mbarrier_arrive_shared`

"""
function mbarrier_arrive_shared(addr::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[addr,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.mbarrier.arrive.shared",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mbarrier_init`

"""
function mbarrier_init(addr::Value, count::Value; location=Location())
    _results = IR.Type[]
    _operands = Value[addr, count]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.mbarrier.init",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mbarrier_init_shared`

"""
function mbarrier_init_shared(addr::Value, count::Value; location=Location())
    _results = IR.Type[]
    _operands = Value[addr, count]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.mbarrier.init.shared",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mbarrier_inval`

"""
function mbarrier_inval(addr::Value; location=Location())
    _results = IR.Type[]
    _operands = Value[addr,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.mbarrier.inval",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mbarrier_inval_shared`

"""
function mbarrier_inval_shared(addr::Value; location=Location())
    _results = IR.Type[]
    _operands = Value[addr,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.mbarrier.inval.shared",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mbarrier_test_wait`

"""
function mbarrier_test_wait(addr::Value, state::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[addr, state]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.mbarrier.test.wait",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mbarrier_test_wait_shared`

"""
function mbarrier_test_wait_shared(
    addr::Value, state::Value; res::IR.Type, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[addr, state]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.mbarrier.test.wait.shared",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mbarrier_try_wait_parity`

"""
function mbarrier_try_wait_parity(
    addr::Value, phase::Value, ticks::Value; location=Location()
)
    _results = IR.Type[]
    _operands = Value[addr, phase, ticks]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.mbarrier.try_wait.parity",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mbarrier_try_wait_parity_shared`

"""
function mbarrier_try_wait_parity_shared(
    addr::Value, phase::Value, ticks::Value; location=Location()
)
    _results = IR.Type[]
    _operands = Value[addr, phase, ticks]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.mbarrier.try_wait.parity.shared",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mma_sync`

The `nvvm.mma.sync` operation collectively performs the operation
`D = matmul(A, B) + C` using all threads in a warp.

All the threads in the warp must execute the same `mma.sync` operation.

For each possible multiplicand PTX data type, there are one or more possible
instruction shapes given as \"mMnNkK\". The below table describes the posssibilities
as well as the types required for the operands. Note that the data type for
C (the accumulator) and D (the result) can vary independently when there are
multiple possibilities in the \"C/D Type\" column.

When an optional attribute cannot be immediately inferred from the types of
the operands and the result during parsing or validation, an error will be
raised.

`b1Op` is only relevant when the binary (b1) type is given to
`multiplicandDataType`. It specifies how the multiply-and-acumulate is
performed and is either `xor_popc` or `and_poc`. The default is `xor_popc`.

`intOverflowBehavior` is only relevant when the `multiplicandType` attribute
is one of `u8, s8, u4, s4`, this attribute describes how overflow is handled
in the accumulator. When the attribute is `satfinite`, the accumulator values
are clamped in the int32 range on overflow. This is the default behavior.
Alternatively, accumulator behavior `wrapped` can also be specified, in
which case overflow wraps from one end of the range to the other.

`layoutA` and `layoutB` are required and should generally be set to
`#nvvm.mma_layout<row>` and `#nvvm.mma_layout<col>` respectively, but other
combinations are possible for certain layouts according to the table below.

```
| A/B Type | Shape     | ALayout | BLayout | A Type   | B Type   | C/D Type          |
|----------|-----------|---------|---------|----------|----------|-------------------|
| f64      | .m8n8k4   | row     | col     | 1x f64   | 1x f64   | 2x f64            |
| f16      | .m8n8k4   | row/col | row/col | 2x f16x2 | 2x f16x2 | 4x f16x2 or 8xf32 |
|          | .m16n8k8  | row     | col     | 2x f16x2 | 1x f16x2 | 2x f16x2 or 4 f32 |
|          | .m16n8k16 | row     | col     | 4x f16x2 | 2x f16x2 | 2x f16x2 or 4 f32 |
| bf16     | .m16n8k8  | row     | col     | 2x f16x2 | 1x f16x2 | 2x f16x2 or 4 f32 |
|          | .m16n8k16 | row     | col     | 4x f16x2 | 2x f16x2 | 2x f16x2 or 4 f32 |
| tf32     | .m16n8k4  | row     | col     | 2x i32   | 1x i32   | 4x f32            |
|          | .m16n8k8  | row     | col     | 4x i32   | 2x i32   | 2x f16x2 or 4 f32 |
| u8/s8    | .m8n8k16  | row     | col     | 1x i32   | 1x i32   | 2x i32            |
|          | .m16n8k16 | row     | col     | 2x i32   | 1x i32   | 4x i32            |
|          | .m16n8k32 | row     | col     | 4x i32   | 2x i32   | 4x i32            |
| u4/s4    | .m8n8k32  | row     | col     | 1x i32   | 1x i32   | 2x i32            |
|          | m16n8k32  | row     | col     | 2x i32   | 1x i32   | 4x i32            |
|          | m16n8k64  | row     | col     | 4x i32   | 2x i32   | 4x i32            |
| b1       | m8n8k128  | row     | col     | 1x i32   | 1x i32   | 2x i32            |
|          | m16n8k128 | row     | col     | 2x i32   | 1x i32   | 4x i32            |
```


# Example
```mlir

%128 = nvvm.mma.sync A[%120, %121, %122, %123]
                     B[%124, %125]
                     C[%126, %127]
                     {layoutA = #nvvm.mma_layout<row>,
                      layoutB = #nvvm.mma_layout<col>,
                      shape = {k = 16 : i32, m = 16 : i32, n = 8 : i32}}
    : (vector<2xf16>, vector<2xf16>, vector<2xf16>)
       -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
```
"""
function mma_sync(
    operandA::Vector{Value},
    operandB::Vector{Value},
    operandC::Vector{Value};
    res::IR.Type,
    shape,
    b1Op=nothing,
    intOverflowBehavior=nothing,
    layoutA,
    layoutB,
    multiplicandAPtxType=nothing,
    multiplicandBPtxType=nothing,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[operandA..., operandB..., operandC...]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("shape", shape),
        namedattribute("layoutA", layoutA),
        namedattribute("layoutB", layoutB),
    ]
    push!(
        attributes,
        operandsegmentsizes([length(operandA), length(operandB), length(operandC)]),
    )
    !isnothing(b1Op) && push!(attributes, namedattribute("b1Op", b1Op))
    !isnothing(intOverflowBehavior) &&
        push!(attributes, namedattribute("intOverflowBehavior", intOverflowBehavior))
    !isnothing(multiplicandAPtxType) &&
        push!(attributes, namedattribute("multiplicandAPtxType", multiplicandAPtxType))
    !isnothing(multiplicandBPtxType) &&
        push!(attributes, namedattribute("multiplicandBPtxType", multiplicandBPtxType))

    return IR.create_operation(
        "nvvm.mma.sync",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`rcp_approx_ftz_f`

"""
function rcp_approx_ftz_f(arg::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[arg,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.rcp.approx.ftz.f",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`redux_sync`

"""
function redux_sync(
    val::Value, mask_and_clamp::Value; res::IR.Type, kind, location=Location()
)
    _results = IR.Type[res,]
    _operands = Value[val, mask_and_clamp]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("kind", kind),]

    return IR.create_operation(
        "nvvm.redux.sync",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`shfl_sync`

"""
function shfl_sync(
    dst::Value,
    val::Value,
    offset::Value,
    mask_and_clamp::Value;
    res::IR.Type,
    kind,
    return_value_and_is_valid=nothing,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[dst, val, offset, mask_and_clamp]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("kind", kind),]
    !isnothing(return_value_and_is_valid) && push!(
        attributes,
        namedattribute("return_value_and_is_valid", return_value_and_is_valid),
    )

    return IR.create_operation(
        "nvvm.shfl.sync",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`bar_warp_sync`

"""
function bar_warp_sync(mask::Value; location=Location())
    _results = IR.Type[]
    _operands = Value[mask,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.bar.warp.sync",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`read_ptx_sreg_tid_x`

"""
function read_ptx_sreg_tid_x(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.read.ptx.sreg.tid.x",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`read_ptx_sreg_tid_y`

"""
function read_ptx_sreg_tid_y(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.read.ptx.sreg.tid.y",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`read_ptx_sreg_tid_z`

"""
function read_ptx_sreg_tid_z(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.read.ptx.sreg.tid.z",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`vote_ballot_sync`

"""
function vote_ballot_sync(mask::Value, pred::Value; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[mask, pred]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.vote.ballot.sync",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`wmma_load`

"""
function wmma_load(
    ptr::Value,
    stride::Value;
    res::IR.Type,
    m,
    n,
    k,
    layout,
    eltype,
    frag,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[ptr, stride]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("m", m),
        namedattribute("n", n),
        namedattribute("k", k),
        namedattribute("layout", layout),
        namedattribute("eltype", eltype),
        namedattribute("frag", frag),
    ]

    return IR.create_operation(
        "nvvm.wmma.load",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`wmma_mma`

"""
function wmma_mma(
    args::Vector{Value};
    res::IR.Type,
    m,
    n,
    k,
    layoutA,
    layoutB,
    eltypeA,
    eltypeB,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("m", m),
        namedattribute("n", n),
        namedattribute("k", k),
        namedattribute("layoutA", layoutA),
        namedattribute("layoutB", layoutB),
        namedattribute("eltypeA", eltypeA),
        namedattribute("eltypeB", eltypeB),
    ]

    return IR.create_operation(
        "nvvm.wmma.mma",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`wmma_store`

"""
function wmma_store(
    ptr::Value,
    args::Vector{Value},
    stride::Value;
    m,
    n,
    k,
    layout,
    eltype,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[ptr, args..., stride]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("m", m),
        namedattribute("n", n),
        namedattribute("k", k),
        namedattribute("layout", layout),
        namedattribute("eltype", eltype),
    ]

    return IR.create_operation(
        "nvvm.wmma.store",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`read_ptx_sreg_warpsize`

"""
function read_ptx_sreg_warpsize(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.read.ptx.sreg.warpsize",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`wgmma_fence_aligned`

Enforce an ordering of register accesses between warpgroup level matrix 
multiplication and other operations. 
See for more information:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-fence
"""
function wgmma_fence_aligned(; location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.wgmma.fence.aligned",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`wgmma_commit_group_sync_aligned`

Commits all prior uncommitted warpgroup level matrix multiplication operations.
See for more information:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-commit-group
"""
function wgmma_commit_group_sync_aligned(; location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "nvvm.wgmma.commit.group.sync.aligned",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`wgmma_wait_group_sync_aligned`

Signal the completion of a preceding warpgroup operation.
See for more information:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group
"""
function wgmma_wait_group_sync_aligned(; group, location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("group", group),]

    return IR.create_operation(
        "nvvm.wgmma.wait.group.sync.aligned",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

import ...IR:
    IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes

"""
`barrier`

"""
function barrier(; location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.barrier",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`workgroup_dim_x`

"""
function workgroup_dim_x(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.workgroup.dim.x",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`workgroup_dim_y`

"""
function workgroup_dim_y(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.workgroup.dim.y",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`workgroup_dim_z`

"""
function workgroup_dim_z(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.workgroup.dim.z",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`workgroup_id_x`

"""
function workgroup_id_x(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.workgroup.id.x",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`workgroup_id_y`

"""
function workgroup_id_y(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.workgroup.id.y",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`workgroup_id_z`

"""
function workgroup_id_z(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.workgroup.id.z",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`grid_dim_x`

"""
function grid_dim_x(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.grid.dim.x",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`grid_dim_y`

"""
function grid_dim_y(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.grid.dim.y",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`grid_dim_z`

"""
function grid_dim_z(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.grid.dim.z",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`buffer_load`

"""
function buffer_load(
    rsrc::Value,
    vindex::Value,
    offset::Value,
    glc::Value,
    slc::Value;
    res::IR.Type,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[rsrc, vindex, offset, glc, slc]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.buffer.load",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`buffer_store`

"""
function buffer_store(
    vdata::Value,
    rsrc::Value,
    vindex::Value,
    offset::Value,
    glc::Value,
    slc::Value;
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[vdata, rsrc, vindex, offset, glc, slc]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.buffer.store",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`raw_buffer_atomic_cmpswap`

"""
function raw_buffer_atomic_cmpswap(
    src::Value,
    cmp::Value,
    rsrc::Value,
    offset::Value,
    soffset::Value,
    aux::Value;
    res::IR.Type,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[src, cmp, rsrc, offset, soffset, aux]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.raw.buffer.atomic.cmpswap",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`raw_buffer_atomic_fadd`

"""
function raw_buffer_atomic_fadd(
    vdata::Value,
    rsrc::Value,
    offset::Value,
    soffset::Value,
    aux::Value;
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[vdata, rsrc, offset, soffset, aux]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.raw.buffer.atomic.fadd",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`raw_buffer_atomic_fmax`

"""
function raw_buffer_atomic_fmax(
    vdata::Value,
    rsrc::Value,
    offset::Value,
    soffset::Value,
    aux::Value;
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[vdata, rsrc, offset, soffset, aux]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.raw.buffer.atomic.fmax",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`raw_buffer_atomic_smax`

"""
function raw_buffer_atomic_smax(
    vdata::Value,
    rsrc::Value,
    offset::Value,
    soffset::Value,
    aux::Value;
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[vdata, rsrc, offset, soffset, aux]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.raw.buffer.atomic.smax",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`raw_buffer_atomic_umin`

"""
function raw_buffer_atomic_umin(
    vdata::Value,
    rsrc::Value,
    offset::Value,
    soffset::Value,
    aux::Value;
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[vdata, rsrc, offset, soffset, aux]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.raw.buffer.atomic.umin",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`raw_buffer_load`

"""
function raw_buffer_load(
    rsrc::Value,
    offset::Value,
    soffset::Value,
    aux::Value;
    res::IR.Type,
    location=Location(),
)
    _results = IR.Type[res,]
    _operands = Value[rsrc, offset, soffset, aux]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.raw.buffer.load",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`raw_buffer_store`

"""
function raw_buffer_store(
    vdata::Value,
    rsrc::Value,
    offset::Value,
    soffset::Value,
    aux::Value;
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[vdata, rsrc, offset, soffset, aux]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.raw.buffer.store",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`workitem_id_x`

"""
function workitem_id_x(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.workitem.id.x",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`workitem_id_y`

"""
function workitem_id_y(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.workitem.id.y",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`workitem_id_z`

"""
function workitem_id_z(; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.workitem.id.z",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_f32_16x16x16bf16_1k`

"""
function mfma_f32_16x16x16bf16_1k(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.f32.16x16x16bf16.1k",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_f32_16x16x16f16`

"""
function mfma_f32_16x16x16f16(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.f32.16x16x16f16",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_f32_16x16x1f32`

"""
function mfma_f32_16x16x1f32(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.f32.16x16x1f32",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_f32_16x16x2bf16`

"""
function mfma_f32_16x16x2bf16(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.f32.16x16x2bf16",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_f32_16x16x32_bf8_bf8`

"""
function mfma_f32_16x16x32_bf8_bf8(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.f32.16x16x32.bf8.bf8",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_f32_16x16x32_bf8_fp8`

"""
function mfma_f32_16x16x32_bf8_fp8(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.f32.16x16x32.bf8.fp8",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_f32_16x16x32_fp8_bf8`

"""
function mfma_f32_16x16x32_fp8_bf8(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.f32.16x16x32.fp8.bf8",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_f32_16x16x32_fp8_fp8`

"""
function mfma_f32_16x16x32_fp8_fp8(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.f32.16x16x32.fp8.fp8",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_f32_16x16x4bf16_1k`

"""
function mfma_f32_16x16x4bf16_1k(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.f32.16x16x4bf16.1k",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_f32_16x16x4f16`

"""
function mfma_f32_16x16x4f16(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.f32.16x16x4f16",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_f32_16x16x4f32`

"""
function mfma_f32_16x16x4f32(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.f32.16x16x4f32",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_f32_16x16x8_xf32`

"""
function mfma_f32_16x16x8_xf32(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.f32.16x16x8.xf32",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_f32_16x16x8bf16`

"""
function mfma_f32_16x16x8bf16(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.f32.16x16x8bf16",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_f32_32x32x16_bf8_bf8`

"""
function mfma_f32_32x32x16_bf8_bf8(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.f32.32x32x16.bf8.bf8",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_f32_32x32x16_bf8_fp8`

"""
function mfma_f32_32x32x16_bf8_fp8(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.f32.32x32x16.bf8.fp8",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_f32_32x32x16_fp8_bf8`

"""
function mfma_f32_32x32x16_fp8_bf8(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.f32.32x32x16.fp8.bf8",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_f32_32x32x16_fp8_fp8`

"""
function mfma_f32_32x32x16_fp8_fp8(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.f32.32x32x16.fp8.fp8",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_f32_32x32x1f32`

"""
function mfma_f32_32x32x1f32(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.f32.32x32x1f32",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_f32_32x32x2bf16`

"""
function mfma_f32_32x32x2bf16(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.f32.32x32x2bf16",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_f32_32x32x2f32`

"""
function mfma_f32_32x32x2f32(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.f32.32x32x2f32",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_f32_32x32x4_xf32`

"""
function mfma_f32_32x32x4_xf32(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.f32.32x32x4.xf32",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_f32_32x32x4bf16`

"""
function mfma_f32_32x32x4bf16(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.f32.32x32x4bf16",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_f32_32x32x4bf16_1k`

"""
function mfma_f32_32x32x4bf16_1k(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.f32.32x32x4bf16.1k",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_f32_32x32x4f16`

"""
function mfma_f32_32x32x4f16(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.f32.32x32x4f16",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_f32_32x32x8bf16_1k`

"""
function mfma_f32_32x32x8bf16_1k(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.f32.32x32x8bf16.1k",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_f32_32x32x8f16`

"""
function mfma_f32_32x32x8f16(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.f32.32x32x8f16",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_f32_4x4x1f32`

"""
function mfma_f32_4x4x1f32(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.f32.4x4x1f32",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_f32_4x4x2bf16`

"""
function mfma_f32_4x4x2bf16(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.f32.4x4x2bf16",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_f32_4x4x4bf16_1k`

"""
function mfma_f32_4x4x4bf16_1k(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.f32.4x4x4bf16.1k",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_f32_4x4x4f16`

"""
function mfma_f32_4x4x4f16(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.f32.4x4x4f16",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_f64_16x16x4f64`

"""
function mfma_f64_16x16x4f64(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.f64.16x16x4f64",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_f64_4x4x4f64`

"""
function mfma_f64_4x4x4f64(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.f64.4x4x4f64",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_i32_16x16x16i8`

"""
function mfma_i32_16x16x16i8(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.i32.16x16x16i8",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_i32_16x16x32_i8`

"""
function mfma_i32_16x16x32_i8(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.i32.16x16x32.i8",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_i32_16x16x4i8`

"""
function mfma_i32_16x16x4i8(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.i32.16x16x4i8",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_i32_32x32x16_i8`

"""
function mfma_i32_32x32x16_i8(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.i32.32x32x16.i8",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_i32_32x32x4i8`

"""
function mfma_i32_32x32x4i8(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.i32.32x32x4i8",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_i32_32x32x8i8`

"""
function mfma_i32_32x32x8i8(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.i32.32x32x8i8",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`mfma_i32_4x4x4i8`

"""
function mfma_i32_4x4x4i8(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.mfma.i32.4x4x4i8",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`wmma_bf16_16x16x16_bf16`

"""
function wmma_bf16_16x16x16_bf16(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.wmma.bf16.16x16x16.bf16",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`wmma_f16_16x16x16_f16`

"""
function wmma_f16_16x16x16_f16(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.wmma.f16.16x16x16.f16",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`wmma_f32_16x16x16_bf16`

"""
function wmma_f32_16x16x16_bf16(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.wmma.f32.16x16x16.bf16",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`wmma_f32_16x16x16_f16`

"""
function wmma_f32_16x16x16_f16(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.wmma.f32.16x16x16.f16",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`wmma_i32_16x16x16_iu4`

"""
function wmma_i32_16x16x16_iu4(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.wmma.i32.16x16x16.iu4",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`wmma_i32_16x16x16_iu8`

"""
function wmma_i32_16x16x16_iu8(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

    return IR.create_operation(
        "rocdl.wmma.i32.16x16x16.iu8",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

end # llvm
