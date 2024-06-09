module llvm

import ...IR: IR, NamedAttribute, Value, value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes

"""
`ashr`

"""
function ashr(lhs, rhs; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(lhs), value(rhs), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.ashr", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`access_group`

Defines an access group metadata that can be attached to any instruction
that potentially accesses memory. The access group may be attached to a
memory accessing instruction via the `llvm.access.group` metadata and
a branch instruction in the loop latch block via the
`llvm.loop.parallel_accesses` metadata.

See the following link for more details:
https://llvm.org/docs/LangRef.html#llvm-access-group-metadata
"""
function access_group(; sym_name, location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("sym_name", sym_name),]

    return IR.create_operation(
        "llvm.access_group",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`add`

"""
function add(lhs, rhs; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(lhs), value(rhs), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.add", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`addrspacecast`

"""
function addrspacecast(arg; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(arg), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.addrspacecast", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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
`alias_scope_domain`

Defines a domain that may be associated with an alias scope.

See the following link for more details:
https://llvm.org/docs/LangRef.html#noalias-and-alias-scope-metadata
"""
function alias_scope_domain(; sym_name, description=nothing, location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("sym_name", sym_name),]
    !isnothing(description) &&
        push!(_attributes, namedattribute("description", description))

    return IR.create_operation(
        "llvm.alias_scope_domain",
        location;
        operands=_operands,
        owned_regions=_owned_regions,
        successors=_successors,
        attributes=_attributes,
        results=_results,
        result_inference=false,
    )
end

"""
`alias_scope`

Defines an alias scope that can be attached to a memory-accessing operation.
Such scopes can be used in combination with `noalias` metadata to indicate
that sets of memory-affecting operations in one scope do not alias with
memory-affecting operations in another scope.

# Example
  module {
    llvm.func @foo(%ptr1 : !llvm.ptr<i32>) {
        %c0 = llvm.mlir.constant(0 : i32) : i32
        %c4 = llvm.mlir.constant(4 : i32) : i32
        %1 = llvm.ptrtoint %ptr1 : !llvm.ptr<i32> to i32
        %2 = llvm.add %1, %c1 : i32
        %ptr2 = llvm.inttoptr %2 : i32 to !llvm.ptr<i32>
        llvm.store %c0, %ptr1 { alias_scopes = [@metadata::@scope1], llvm.noalias = [@metadata::@scope2] } : !llvm.ptr<i32>
        llvm.store %c4, %ptr2 { alias_scopes = [@metadata::@scope2], llvm.noalias = [@metadata::@scope1] } : !llvm.ptr<i32>
        llvm.return
    }

    llvm.metadata @metadata {
      llvm.alias_scope_domain @unused_domain
      llvm.alias_scope_domain @domain { description = \"Optional domain description\"}
      llvm.alias_scope @scope1 { domain = @domain }
      llvm.alias_scope @scope2 { domain = @domain, description = \"Optional scope description\" }
      llvm.return
    }
  }

See the following link for more details:
https://llvm.org/docs/LangRef.html#noalias-and-alias-scope-metadata
"""
function alias_scope(; sym_name, domain, description=nothing, location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("sym_name", sym_name), namedattribute("domain", domain)
    ]
    !isnothing(description) &&
        push!(_attributes, namedattribute("description", description))

    return IR.create_operation(
        "llvm.alias_scope",
        location;
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
function alloca(arraySize; res::IR.Type, alignment=nothing, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(arraySize), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(alignment) && push!(attributes, namedattribute("alignment", alignment))
    
    IR.create_operation(
        "llvm.alloca", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`and`

"""
function and(lhs, rhs; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(lhs), value(rhs), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.and", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`intr_assume`

"""
function intr_assume(cond; location=Location())
    results = IR.Type[]
    operands = Value[value(cond), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.assume", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`cmpxchg`

"""
function cmpxchg(ptr, cmp, val; res::IR.Type, success_ordering, failure_ordering, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(ptr), value(cmp), value(val), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("success_ordering", success_ordering), namedattribute("failure_ordering", failure_ordering), ]
    
    IR.create_operation(
        "llvm.cmpxchg", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`atomicrmw`

"""
function atomicrmw(ptr, val; res::IR.Type, bin_op, ordering, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(ptr), value(val), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("bin_op", bin_op), namedattribute("ordering", ordering), ]
    
    IR.create_operation(
        "llvm.atomicrmw", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_bitreverse`

"""
function intr_bitreverse(in; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(in), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.intr.bitreverse", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`bitcast`

"""
function bitcast(arg; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(arg), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.bitcast", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`br`

"""
function br(destOperands; dest::Block, location=Location())
    results = IR.Type[]
    operands = Value[value.(destOperands)..., ]
    owned_regions = Region[]
    successors = Block[dest, ]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.br", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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
function attribute `callee`. The trailing type of the instruction is always
the MLIR function type, which may be different from the indirect callee that
has the wrapped LLVM IR function type.

Examples:

```mlir
// Direct call without arguments and with one result.
%0 = llvm.call @foo() : () -> (f32)

// Direct call with arguments and without a result.
llvm.call @bar(%0) : (f32) -> ()

// Indirect call with an argument and without a result.
llvm.call %1(%0) : (f32) -> ()
```
"""
function call(operand_0; result_0::Vector{IR.Type}, callee=nothing, fastmathFlags=nothing, location=Location())
    results = IR.Type[result_0..., ]
    operands = Value[value.(operand_0)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(callee) && push!(attributes, namedattribute("callee", callee))
    !isnothing(fastmathFlags) && push!(attributes, namedattribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.call", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`cond_br`

"""
function cond_br(condition, trueDestOperands, falseDestOperands; branch_weights=nothing, trueDest::Block, falseDest::Block, location=Location())
    results = IR.Type[]
    operands = Value[value(condition), value.(trueDestOperands)..., value.(falseDestOperands)..., ]
    owned_regions = Region[]
    successors = Block[trueDest, falseDest, ]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([1, length(trueDestOperands), length(falseDestOperands), ]))
    !isnothing(branch_weights) && push!(attributes, namedattribute("branch_weights", branch_weights))
    
    IR.create_operation(
        "llvm.cond_br", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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
`intr_copysign`

"""
function intr_copysign(a, b; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(a), value(b), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.intr.copysign", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
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
function intr_coro_begin(token, mem; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(token), value(mem), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.coro.begin", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_coro_end`

"""
function intr_coro_end(handle, unwind; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(handle), value(unwind), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.coro.end", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_coro_free`

"""
function intr_coro_free(id, handle; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(id), value(handle), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.coro.free", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_coro_id`

"""
function intr_coro_id(align, promise, coroaddr, fnaddrs; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(align), value(promise), value(coroaddr), value(fnaddrs), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.coro.id", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_coro_resume`

"""
function intr_coro_resume(handle; location=Location())
    results = IR.Type[]
    operands = Value[value(handle), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.coro.resume", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_coro_save`

"""
function intr_coro_save(handle; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(handle), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.coro.save", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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
function intr_coro_suspend(save, final; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(save), value(final), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.coro.suspend", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_cos`

"""
function intr_cos(in; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(in), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.intr.cos", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`intr_ctlz`

"""
function intr_ctlz(in, zero_undefined; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(in), value(zero_undefined), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.ctlz", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_cttz`

"""
function intr_cttz(in, zero_undefined; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(in), value(zero_undefined), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.cttz", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_ctpop`

"""
function intr_ctpop(in; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(in), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.intr.ctpop", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`intr_eh_typeid_for`

"""
function intr_eh_typeid_for(type_info; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(type_info), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.eh.typeid.for", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_exp2`

"""
function intr_exp2(in; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(in), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.intr.exp2", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`intr_exp`

"""
function intr_exp(in; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(in), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.intr.exp", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`extractelement`

"""
function extractelement(vector, position; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(vector), value(position), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.extractelement", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`extractvalue`

"""
function extractvalue(container; res::IR.Type, position, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(container), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("position", position), ]
    
    IR.create_operation(
        "llvm.extractvalue", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_fabs`

"""
function intr_fabs(in; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(in), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.intr.fabs", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`fadd`

"""
function fadd(lhs, rhs; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    results = IR.Type[]
    operands = Value[value(lhs), value(rhs), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    !isnothing(fastmathFlags) && push!(attributes, namedattribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.fadd", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`intr_ceil`

"""
function intr_ceil(in; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(in), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.intr.ceil", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`fcmp`

"""
function fcmp(lhs, rhs; res::IR.Type, predicate, fastmathFlags=nothing, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(lhs), value(rhs), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("predicate", predicate), ]
    !isnothing(fastmathFlags) && push!(attributes, namedattribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.fcmp", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`fdiv`

"""
function fdiv(lhs, rhs; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    results = IR.Type[]
    operands = Value[value(lhs), value(rhs), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    !isnothing(fastmathFlags) && push!(attributes, namedattribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.fdiv", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`intr_floor`

"""
function intr_floor(in; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(in), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.intr.floor", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`intr_fma`

"""
function intr_fma(a, b, c; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(a), value(b), value(c), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.intr.fma", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`intr_fmuladd`

"""
function intr_fmuladd(a, b, c; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(a), value(b), value(c), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.intr.fmuladd", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`fmul`

"""
function fmul(lhs, rhs; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    results = IR.Type[]
    operands = Value[value(lhs), value(rhs), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    !isnothing(fastmathFlags) && push!(attributes, namedattribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.fmul", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`fneg`

"""
function fneg(operand; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    results = IR.Type[]
    operands = Value[value(operand), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    !isnothing(fastmathFlags) && push!(attributes, namedattribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.fneg", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`fpext`

"""
function fpext(arg; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(arg), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.fpext", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`fptosi`

"""
function fptosi(arg; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(arg), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.fptosi", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`fptoui`

"""
function fptoui(arg; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(arg), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.fptoui", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`fptrunc`

"""
function fptrunc(arg; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(arg), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.fptrunc", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`frem`

"""
function frem(lhs, rhs; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    results = IR.Type[]
    operands = Value[value(lhs), value(rhs), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    !isnothing(fastmathFlags) && push!(attributes, namedattribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.frem", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`fsub`

"""
function fsub(lhs, rhs; res=nothing::Union{Nothing, IR.Type}, fastmathFlags=nothing, location=Location())
    results = IR.Type[]
    operands = Value[value(lhs), value(rhs), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    !isnothing(fastmathFlags) && push!(attributes, namedattribute("fastmathFlags", fastmathFlags))
    
    IR.create_operation(
        "llvm.fsub", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`fence`

"""
function fence(; ordering, syncscope, location=Location())
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[
        namedattribute("ordering", ordering), namedattribute("syncscope", syncscope)
    ]

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
function freeze(val; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(val), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.freeze", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`getelementptr`

"""
function getelementptr(base, indices; res::IR.Type, structIndices, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(base), value.(indices)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("structIndices", structIndices), ]
    
    IR.create_operation(
        "llvm.getelementptr", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_get_active_lane_mask`

"""
function intr_get_active_lane_mask(base, n; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(base), value(n), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.get.active.lane.mask", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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
    value=nothing,
    alignment=nothing,
    addr_space=nothing,
    unnamed_addr=nothing,
    section=nothing,
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
    !isnothing(constant) && push!(_attributes, namedattribute("constant", constant))
    !isnothing(dso_local) && push!(_attributes, namedattribute("dso_local", dso_local))
    !isnothing(value) && push!(_attributes, namedattribute("value", value))
    !isnothing(alignment) && push!(_attributes, namedattribute("alignment", alignment))
    !isnothing(addr_space) && push!(_attributes, namedattribute("addr_space", addr_space))
    !isnothing(unnamed_addr) &&
        push!(_attributes, namedattribute("unnamed_addr", unnamed_addr))
    !isnothing(section) && push!(_attributes, namedattribute("section", section))

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
function icmp(lhs, rhs; res::IR.Type, predicate, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(lhs), value(rhs), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("predicate", predicate), ]
    
    IR.create_operation(
        "llvm.icmp", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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
function inline_asm(operands_; res=nothing::Union{Nothing, IR.Type}, asm_string, constraints, has_side_effects=nothing, is_align_stack=nothing, asm_dialect=nothing, operand_attrs=nothing, location=Location())
    results = IR.Type[]
    operands = Value[value.(operands_)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("asm_string", asm_string), namedattribute("constraints", constraints), ]
    !isnothing(res) && push!(results, res)
    !isnothing(has_side_effects) && push!(attributes, namedattribute("has_side_effects", has_side_effects))
    !isnothing(is_align_stack) && push!(attributes, namedattribute("is_align_stack", is_align_stack))
    !isnothing(asm_dialect) && push!(attributes, namedattribute("asm_dialect", asm_dialect))
    !isnothing(operand_attrs) && push!(attributes, namedattribute("operand_attrs", operand_attrs))
    
    IR.create_operation(
        "llvm.inline_asm", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`insertelement`

"""
function insertelement(vector, value, position; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(vector), value(value), value(position), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.insertelement", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`insertvalue`

"""
function insertvalue(container, value; res::IR.Type, position, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(container), value(value), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("position", position), ]
    
    IR.create_operation(
        "llvm.insertvalue", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`inttoptr`

"""
function inttoptr(arg; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(arg), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.inttoptr", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`invoke`

"""
function invoke(callee_operands, normalDestOperands, unwindDestOperands; result_0::Vector{IR.Type}, callee=nothing, normalDest::Block, unwindDest::Block, location=Location())
    results = IR.Type[result_0..., ]
    operands = Value[value.(callee_operands)..., value.(normalDestOperands)..., value.(unwindDestOperands)..., ]
    owned_regions = Region[]
    successors = Block[normalDest, unwindDest, ]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(callee_operands), length(normalDestOperands), length(unwindDestOperands), ]))
    !isnothing(callee) && push!(attributes, namedattribute("callee", callee))
    
    IR.create_operation(
        "llvm.invoke", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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
    linkage=nothing,
    dso_local=nothing,
    personality=nothing,
    passthrough=nothing,
    body::Region,
    location=Location(),
)
    _results = IR.Type[]
    _operands = Value[]
    _owned_regions = Region[body,]
    _successors = Block[]
    _attributes = NamedAttribute[]
    !isnothing(linkage) && push!(_attributes, namedattribute("linkage", linkage))
    !isnothing(dso_local) && push!(_attributes, namedattribute("dso_local", dso_local))
    !isnothing(personality) &&
        push!(_attributes, namedattribute("personality", personality))
    !isnothing(passthrough) &&
        push!(_attributes, namedattribute("passthrough", passthrough))

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
function lshr(lhs, rhs; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(lhs), value(rhs), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.lshr", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`landingpad`

"""
function landingpad(operand_0; res::IR.Type, cleanup=nothing, location=Location())
    results = IR.Type[res, ]
    operands = Value[value.(operand_0)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(cleanup) && push!(attributes, namedattribute("cleanup", cleanup))
    
    IR.create_operation(
        "llvm.landingpad", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`load`

"""
function load(addr; res::IR.Type, access_groups=nothing, alias_scopes=nothing, noalias_scopes=nothing, alignment=nothing, volatile_=nothing, nontemporal=nothing, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(addr), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(access_groups) && push!(attributes, namedattribute("access_groups", access_groups))
    !isnothing(alias_scopes) && push!(attributes, namedattribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, namedattribute("noalias_scopes", noalias_scopes))
    !isnothing(alignment) && push!(attributes, namedattribute("alignment", alignment))
    !isnothing(volatile_) && push!(attributes, namedattribute("volatile_", volatile_))
    !isnothing(nontemporal) && push!(attributes, namedattribute("nontemporal", nontemporal))
    
    IR.create_operation(
        "llvm.load", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_log10`

"""
function intr_log10(in; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(in), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.intr.log10", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`intr_log2`

"""
function intr_log2(in; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(in), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.intr.log2", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`intr_log`

"""
function intr_log(in; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(in), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.intr.log", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`intr_masked_load`

"""
function intr_masked_load(data, mask, pass_thru; res::IR.Type, alignment, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(data), value(mask), value.(pass_thru)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("alignment", alignment), ]
    
    IR.create_operation(
        "llvm.intr.masked.load", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_masked_store`

"""
function intr_masked_store(value, data, mask; alignment, location=Location())
    results = IR.Type[]
    operands = Value[value(value), value(data), value(mask), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("alignment", alignment), ]
    
    IR.create_operation(
        "llvm.intr.masked.store", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_matrix_column_major_load`

"""
function intr_matrix_column_major_load(data, stride; res::IR.Type, isVolatile, rows, columns, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(data), value(stride), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("isVolatile", isVolatile), namedattribute("rows", rows), namedattribute("columns", columns), ]
    
    IR.create_operation(
        "llvm.intr.matrix.column.major.load", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_matrix_column_major_store`

"""
function intr_matrix_column_major_store(matrix, data, stride; isVolatile, rows, columns, location=Location())
    results = IR.Type[]
    operands = Value[value(matrix), value(data), value(stride), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("isVolatile", isVolatile), namedattribute("rows", rows), namedattribute("columns", columns), ]
    
    IR.create_operation(
        "llvm.intr.matrix.column.major.store", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_matrix_multiply`

"""
function intr_matrix_multiply(lhs, rhs; res::IR.Type, lhs_rows, lhs_columns, rhs_columns, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(lhs), value(rhs), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("lhs_rows", lhs_rows), namedattribute("lhs_columns", lhs_columns), namedattribute("rhs_columns", rhs_columns), ]
    
    IR.create_operation(
        "llvm.intr.matrix.multiply", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_matrix_transpose`

"""
function intr_matrix_transpose(matrix; res::IR.Type, rows, columns, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(matrix), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("rows", rows), namedattribute("columns", columns), ]
    
    IR.create_operation(
        "llvm.intr.matrix.transpose", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_maxnum`

"""
function intr_maxnum(a, b; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(a), value(b), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.intr.maxnum", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`intr_maximum`

"""
function intr_maximum(a, b; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(a), value(b), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.intr.maximum", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`intr_memcpy_inline`

"""
function intr_memcpy_inline(dst, src, len, isVolatile; location=Location())
    results = IR.Type[]
    operands = Value[value(dst), value(src), value(len), value(isVolatile), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.memcpy.inline", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_memcpy`

"""
function intr_memcpy(dst, src, len, isVolatile; location=Location())
    results = IR.Type[]
    operands = Value[value(dst), value(src), value(len), value(isVolatile), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.memcpy", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_memmove`

"""
function intr_memmove(dst, src, len, isVolatile; location=Location())
    results = IR.Type[]
    operands = Value[value(dst), value(src), value(len), value(isVolatile), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.memmove", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_memset`

"""
function intr_memset(dst, val, len, isVolatile; location=Location())
    results = IR.Type[]
    operands = Value[value(dst), value(val), value(len), value(isVolatile), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.memset", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`metadata`

llvm.metadata op defines one or more metadata nodes.

# Example
  llvm.metadata @metadata {
    llvm.access_group @group1
    llvm.access_group @group2
    llvm.return
  }
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
`intr_minnum`

"""
function intr_minnum(a, b; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(a), value(b), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.intr.minnum", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`intr_minimum`

"""
function intr_minimum(a, b; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(a), value(b), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.intr.minimum", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`mul`

"""
function mul(lhs, rhs; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(lhs), value(rhs), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.mul", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
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
function or(lhs, rhs; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(lhs), value(rhs), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.or", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`intr_powi`

"""
function intr_powi(a, b; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(a), value(b), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.powi", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_pow`

"""
function intr_pow(a, b; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(a), value(b), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.intr.pow", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`intr_prefetch`

"""
function intr_prefetch(addr, rw, hint, cache; location=Location())
    results = IR.Type[]
    operands = Value[value(addr), value(rw), value(hint), value(cache), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.prefetch", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`ptrtoint`

"""
function ptrtoint(arg; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(arg), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.ptrtoint", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`resume`

"""
function resume(value; location=Location())
    results = IR.Type[]
    operands = Value[value(value), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.resume", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`return_`

"""
function return_(args; location=Location())
    results = IR.Type[]
    operands = Value[value.(args)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.return", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_sadd_with_overflow`

"""
function intr_sadd_with_overflow(operand_0, operand_1; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(operand_0), value(operand_1), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.sadd.with.overflow", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`sdiv`

"""
function sdiv(lhs, rhs; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(lhs), value(rhs), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.sdiv", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`sext`

"""
function sext(arg; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(arg), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.sext", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`sitofp`

"""
function sitofp(arg; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(arg), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.sitofp", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_smax`

"""
function intr_smax(a, b; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(a), value(b), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.intr.smax", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`intr_smin`

"""
function intr_smin(a, b; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(a), value(b), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.intr.smin", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`intr_smul_with_overflow`

"""
function intr_smul_with_overflow(operand_0, operand_1; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(operand_0), value(operand_1), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.smul.with.overflow", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`srem`

"""
function srem(lhs, rhs; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(lhs), value(rhs), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.srem", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`intr_ssub_with_overflow`

"""
function intr_ssub_with_overflow(operand_0, operand_1; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(operand_0), value(operand_1), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.ssub.with.overflow", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`select`

"""
function select(condition, trueValue, falseValue; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(condition), value(trueValue), value(falseValue), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.select", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`shl`

"""
function shl(lhs, rhs; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(lhs), value(rhs), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.shl", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`shufflevector`

"""
function shufflevector(v1, v2; res::IR.Type, mask, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(v1), value(v2), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("mask", mask), ]
    
    IR.create_operation(
        "llvm.shufflevector", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_sin`

"""
function intr_sin(in; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(in), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.intr.sin", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`intr_sqrt`

"""
function intr_sqrt(in; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(in), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.intr.sqrt", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`intr_stackrestore`

"""
function intr_stackrestore(ptr; location=Location())
    results = IR.Type[]
    operands = Value[value(ptr), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.stackrestore", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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
`store`

"""
function store(value, addr; access_groups=nothing, alias_scopes=nothing, noalias_scopes=nothing, alignment=nothing, volatile_=nothing, nontemporal=nothing, location=Location())
    results = IR.Type[]
    operands = Value[value(value), value(addr), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(access_groups) && push!(attributes, namedattribute("access_groups", access_groups))
    !isnothing(alias_scopes) && push!(attributes, namedattribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, namedattribute("noalias_scopes", noalias_scopes))
    !isnothing(alignment) && push!(attributes, namedattribute("alignment", alignment))
    !isnothing(volatile_) && push!(attributes, namedattribute("volatile_", volatile_))
    !isnothing(nontemporal) && push!(attributes, namedattribute("nontemporal", nontemporal))
    
    IR.create_operation(
        "llvm.store", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`sub`

"""
function sub(lhs, rhs; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(lhs), value(rhs), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.sub", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`switch`

"""
function switch(value, defaultOperands, caseOperands; case_values=nothing, case_operand_segments, branch_weights=nothing, defaultDestination::Block, caseDestinations::Vector{Block}, location=Location())
    results = IR.Type[]
    operands = Value[value(value), value.(defaultOperands)..., value.(caseOperands)..., ]
    owned_regions = Region[]
    successors = Block[defaultDestination, caseDestinations..., ]
    attributes = NamedAttribute[namedattribute("case_operand_segments", case_operand_segments), ]
    push!(attributes, operandsegmentsizes([1, length(defaultOperands), length(caseOperands), ]))
    !isnothing(case_values) && push!(attributes, namedattribute("case_values", case_values))
    !isnothing(branch_weights) && push!(attributes, namedattribute("branch_weights", branch_weights))
    
    IR.create_operation(
        "llvm.switch", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`trunc`

"""
function trunc(arg; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(arg), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.trunc", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_uadd_with_overflow`

"""
function intr_uadd_with_overflow(operand_0, operand_1; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(operand_0), value(operand_1), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.uadd.with.overflow", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`udiv`

"""
function udiv(lhs, rhs; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(lhs), value(rhs), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.udiv", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`uitofp`

"""
function uitofp(arg; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(arg), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.uitofp", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_umax`

"""
function intr_umax(a, b; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(a), value(b), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.intr.umax", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`intr_umin`

"""
function intr_umin(a, b; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(a), value(b), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.intr.umin", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`intr_umul_with_overflow`

"""
function intr_umul_with_overflow(operand_0, operand_1; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(operand_0), value(operand_1), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.umul.with.overflow", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`urem`

"""
function urem(lhs, rhs; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(lhs), value(rhs), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.urem", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`intr_usub_with_overflow`

"""
function intr_usub_with_overflow(operand_0, operand_1; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(operand_0), value(operand_1), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.usub.with.overflow", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`mlir_undef`

Unlike LLVM IR, MLIR does not have first-class undefined values. Such values
must be created as SSA values using `llvm.mlir.undef`. This operation has no
operands or attributes. It creates an undefined value of the specified LLVM
IR dialect type wrapping an LLVM IR structure type.

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
function xor(lhs, rhs; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[value(lhs), value(rhs), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    IR.create_operation(
        "llvm.xor", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`zext`

"""
function zext(arg; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(arg), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.zext", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_masked_compressstore`

"""
function intr_masked_compressstore(operand_0, operand_1, operand_2; location=Location())
    results = IR.Type[]
    operands = Value[value(operand_0), value(operand_1), value(operand_2), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.masked.compressstore", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_masked_expandload`

"""
function intr_masked_expandload(operand_0, operand_1, operand_2; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(operand_0), value(operand_1), value(operand_2), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.masked.expandload", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_masked_gather`

"""
function intr_masked_gather(ptrs, mask, pass_thru; res::IR.Type, alignment, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(ptrs), value(mask), value.(pass_thru)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("alignment", alignment), ]
    
    IR.create_operation(
        "llvm.intr.masked.gather", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_masked_scatter`

"""
function intr_masked_scatter(value, ptrs, mask; alignment, location=Location())
    results = IR.Type[]
    operands = Value[value(value), value(ptrs), value(mask), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("alignment", alignment), ]
    
    IR.create_operation(
        "llvm.intr.masked.scatter", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_vector_reduce_add`

"""
function intr_vector_reduce_add(operand_0; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(operand_0), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vector.reduce.add", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_vector_reduce_and`

"""
function intr_vector_reduce_and(operand_0; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(operand_0), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vector.reduce.and", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_vector_reduce_fadd`

"""
function intr_vector_reduce_fadd(operand_0, operand_1; res::IR.Type, reassoc=nothing, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(operand_0), value(operand_1), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(reassoc) && push!(attributes, namedattribute("reassoc", reassoc))
    
    IR.create_operation(
        "llvm.intr.vector.reduce.fadd", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_vector_reduce_fmax`

"""
function intr_vector_reduce_fmax(operand_0; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(operand_0), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vector.reduce.fmax", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_vector_reduce_fmin`

"""
function intr_vector_reduce_fmin(operand_0; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(operand_0), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vector.reduce.fmin", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_vector_reduce_fmul`

"""
function intr_vector_reduce_fmul(operand_0, operand_1; res::IR.Type, reassoc=nothing, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(operand_0), value(operand_1), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(reassoc) && push!(attributes, namedattribute("reassoc", reassoc))
    
    IR.create_operation(
        "llvm.intr.vector.reduce.fmul", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_vector_reduce_mul`

"""
function intr_vector_reduce_mul(operand_0; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(operand_0), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vector.reduce.mul", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_vector_reduce_or`

"""
function intr_vector_reduce_or(operand_0; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(operand_0), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vector.reduce.or", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_vector_reduce_smax`

"""
function intr_vector_reduce_smax(operand_0; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(operand_0), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vector.reduce.smax", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_vector_reduce_smin`

"""
function intr_vector_reduce_smin(operand_0; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(operand_0), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vector.reduce.smin", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_vector_reduce_umax`

"""
function intr_vector_reduce_umax(operand_0; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(operand_0), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vector.reduce.umax", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_vector_reduce_umin`

"""
function intr_vector_reduce_umin(operand_0; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(operand_0), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vector.reduce.umin", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_vector_reduce_xor`

"""
function intr_vector_reduce_xor(operand_0; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[value(operand_0), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    IR.create_operation(
        "llvm.intr.vector.reduce.xor", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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
function cp_async_shared_global(dst::Value, src::Value; size, location=Location())
    _results = IR.Type[]
    _operands = Value[dst, src]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[namedattribute("size", size),]

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
`mma_sync`

"""
function mma_sync(args::Vector{Value}; res::IR.Type, location=Location())
    _results = IR.Type[res,]
    _operands = Value[args...,]
    _owned_regions = Region[]
    _successors = Block[]
    _attributes = NamedAttribute[]

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
        _attributes,
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

end # llvm
