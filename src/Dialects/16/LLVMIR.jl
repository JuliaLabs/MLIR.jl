module llvm

import ...IR: NamedAttribute, MLIRType, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


"""
`ashr`

"""
function ashr(lhs::Value, rhs::Value; res=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    create_operation(
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
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sym_name", sym_name), ]
    
    create_operation(
        "llvm.access_group", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`add`

"""
function add(lhs::Value, rhs::Value; res=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    create_operation(
        "llvm.add", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`addrspacecast`

"""
function addrspacecast(arg::Value; res::MLIRType, location=Location())
    results = MLIRType[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
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
function mlir_addressof(; res::MLIRType, global_name, location=Location())
    results = MLIRType[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("global_name", global_name), ]
    
    create_operation(
        "llvm.mlir.addressof", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`alias_scope_domain`

Defines a domain that may be associated with an alias scope.

See the following link for more details:
https://llvm.org/docs/LangRef.html#noalias-and-alias-scope-metadata
"""
function alias_scope_domain(; sym_name, description=nothing, location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sym_name", sym_name), ]
    !isnothing(description) && push!(attributes, namedattribute("description", description))
    
    create_operation(
        "llvm.alias_scope_domain", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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
    }
  }

See the following link for more details:
https://llvm.org/docs/LangRef.html#noalias-and-alias-scope-metadata
"""
function alias_scope(; sym_name, domain, description=nothing, location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sym_name", sym_name), namedattribute("domain", domain), ]
    !isnothing(description) && push!(attributes, namedattribute("description", description))
    
    create_operation(
        "llvm.alias_scope", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`alloca`

"""
function alloca(arraySize::Value; res::MLIRType, alignment=nothing, elem_type=nothing, location=Location())
    results = MLIRType[res, ]
    operands = Value[arraySize, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(alignment) && push!(attributes, namedattribute("alignment", alignment))
    !isnothing(elem_type) && push!(attributes, namedattribute("elem_type", elem_type))
    
    create_operation(
        "llvm.alloca", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`and`

"""
function and(lhs::Value, rhs::Value; res=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    create_operation(
        "llvm.and", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`cmpxchg`

"""
function cmpxchg(ptr::Value, cmp::Value, val::Value; res::MLIRType, success_ordering, failure_ordering, location=Location())
    results = MLIRType[res, ]
    operands = Value[ptr, cmp, val, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("success_ordering", success_ordering), namedattribute("failure_ordering", failure_ordering), ]
    
    create_operation(
        "llvm.cmpxchg", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`atomicrmw`

"""
function atomicrmw(ptr::Value, val::Value; res::MLIRType, bin_op, ordering, location=Location())
    results = MLIRType[res, ]
    operands = Value[ptr, val, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("bin_op", bin_op), namedattribute("ordering", ordering), ]
    
    create_operation(
        "llvm.atomicrmw", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`bitcast`

"""
function bitcast(arg::Value; res::MLIRType, location=Location())
    results = MLIRType[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.bitcast", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`br`

"""
function br(destOperands::Vector{Value}; dest::Block, location=Location())
    results = MLIRType[]
    operands = Value[destOperands..., ]
    owned_regions = Region[]
    successors = Block[dest, ]
    attributes = NamedAttribute[]
    
    create_operation(
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
function call(operand_0::Vector{Value}; result=nothing::Union{Nothing, MLIRType}, callee=nothing, fastmathFlags=nothing, branch_weights=nothing, location=Location())
    results = MLIRType[]
    operands = Value[operand_0..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(callee) && push!(attributes, namedattribute("callee", callee))
    !isnothing(fastmathFlags) && push!(attributes, namedattribute("fastmathFlags", fastmathFlags))
    !isnothing(branch_weights) && push!(attributes, namedattribute("branch_weights", branch_weights))
    
    create_operation(
        "llvm.call", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`cond_br`

"""
function cond_br(condition::Value, trueDestOperands::Vector{Value}, falseDestOperands::Vector{Value}; branch_weights=nothing, trueDest::Block, falseDest::Block, location=Location())
    results = MLIRType[]
    operands = Value[condition, trueDestOperands..., falseDestOperands..., ]
    owned_regions = Region[]
    successors = Block[trueDest, falseDest, ]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([1, length(trueDestOperands), length(falseDestOperands), ]))
    !isnothing(branch_weights) && push!(attributes, namedattribute("branch_weights", branch_weights))
    
    create_operation(
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
function mlir_constant(; res::MLIRType, value, location=Location())
    results = MLIRType[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("value", value), ]
    
    create_operation(
        "llvm.mlir.constant", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`extractelement`

"""
function extractelement(vector::Value, position::Value; res=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[vector, position, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    create_operation(
        "llvm.extractelement", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`extractvalue`

"""
function extractvalue(container::Value; res::MLIRType, position, location=Location())
    results = MLIRType[res, ]
    operands = Value[container, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("position", position), ]
    
    create_operation(
        "llvm.extractvalue", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`fadd`

"""
function fadd(lhs::Value, rhs::Value; res=nothing::Union{Nothing, MLIRType}, fastmathFlags=nothing, location=Location())
    results = MLIRType[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    !isnothing(fastmathFlags) && push!(attributes, namedattribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.fadd", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`fcmp`

"""
function fcmp(lhs::Value, rhs::Value; res=nothing::Union{Nothing, MLIRType}, predicate, fastmathFlags=nothing, location=Location())
    results = MLIRType[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("predicate", predicate), ]
    !isnothing(res) && push!(results, res)
    !isnothing(fastmathFlags) && push!(attributes, namedattribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.fcmp", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`fdiv`

"""
function fdiv(lhs::Value, rhs::Value; res=nothing::Union{Nothing, MLIRType}, fastmathFlags=nothing, location=Location())
    results = MLIRType[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    !isnothing(fastmathFlags) && push!(attributes, namedattribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.fdiv", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`fmul`

"""
function fmul(lhs::Value, rhs::Value; res=nothing::Union{Nothing, MLIRType}, fastmathFlags=nothing, location=Location())
    results = MLIRType[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    !isnothing(fastmathFlags) && push!(attributes, namedattribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.fmul", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`fneg`

"""
function fneg(operand::Value; res=nothing::Union{Nothing, MLIRType}, fastmathFlags=nothing, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    !isnothing(fastmathFlags) && push!(attributes, namedattribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.fneg", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`fpext`

"""
function fpext(arg::Value; res::MLIRType, location=Location())
    results = MLIRType[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.fpext", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`fptosi`

"""
function fptosi(arg::Value; res::MLIRType, location=Location())
    results = MLIRType[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.fptosi", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`fptoui`

"""
function fptoui(arg::Value; res::MLIRType, location=Location())
    results = MLIRType[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.fptoui", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`fptrunc`

"""
function fptrunc(arg::Value; res::MLIRType, location=Location())
    results = MLIRType[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.fptrunc", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`frem`

"""
function frem(lhs::Value, rhs::Value; res=nothing::Union{Nothing, MLIRType}, fastmathFlags=nothing, location=Location())
    results = MLIRType[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    !isnothing(fastmathFlags) && push!(attributes, namedattribute("fastmathFlags", fastmathFlags))
    
    create_operation(
        "llvm.frem", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`fsub`

"""
function fsub(lhs::Value, rhs::Value; res=nothing::Union{Nothing, MLIRType}, fastmathFlags=nothing, location=Location())
    results = MLIRType[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    !isnothing(fastmathFlags) && push!(attributes, namedattribute("fastmathFlags", fastmathFlags))
    
    create_operation(
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
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("ordering", ordering), namedattribute("syncscope", syncscope), ]
    
    create_operation(
        "llvm.fence", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`freeze`

"""
function freeze(val::Value; res=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[val, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    create_operation(
        "llvm.freeze", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
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
function getelementptr(base::Value, dynamicIndices::Vector{Value}; res::MLIRType, rawConstantIndices, elem_type=nothing, inbounds=nothing, location=Location())
    results = MLIRType[res, ]
    operands = Value[base, dynamicIndices..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("rawConstantIndices", rawConstantIndices), ]
    !isnothing(elem_type) && push!(attributes, namedattribute("elem_type", elem_type))
    !isnothing(inbounds) && push!(attributes, namedattribute("inbounds", inbounds))
    
    create_operation(
        "llvm.getelementptr", location;
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
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("ctors", ctors), namedattribute("priorities", priorities), ]
    
    create_operation(
        "llvm.mlir.global_ctors", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dtors", dtors), namedattribute("priorities", priorities), ]
    
    create_operation(
        "llvm.mlir.global_dtors", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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
function mlir_global(; global_type, constant=nothing, sym_name, linkage, dso_local=nothing, thread_local_=nothing, value=nothing, alignment=nothing, addr_space=nothing, unnamed_addr=nothing, section=nothing, initializer::Region, location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[initializer, ]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("global_type", global_type), namedattribute("sym_name", sym_name), namedattribute("linkage", linkage), ]
    !isnothing(constant) && push!(attributes, namedattribute("constant", constant))
    !isnothing(dso_local) && push!(attributes, namedattribute("dso_local", dso_local))
    !isnothing(thread_local_) && push!(attributes, namedattribute("thread_local_", thread_local_))
    !isnothing(value) && push!(attributes, namedattribute("value", value))
    !isnothing(alignment) && push!(attributes, namedattribute("alignment", alignment))
    !isnothing(addr_space) && push!(attributes, namedattribute("addr_space", addr_space))
    !isnothing(unnamed_addr) && push!(attributes, namedattribute("unnamed_addr", unnamed_addr))
    !isnothing(section) && push!(attributes, namedattribute("section", section))
    
    create_operation(
        "llvm.mlir.global", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`icmp`

"""
function icmp(lhs::Value, rhs::Value; res=nothing::Union{Nothing, MLIRType}, predicate, location=Location())
    results = MLIRType[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("predicate", predicate), ]
    !isnothing(res) && push!(results, res)
    
    create_operation(
        "llvm.icmp", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
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
function inline_asm(operands::Vector{Value}; res=nothing::Union{Nothing, MLIRType}, asm_string, constraints, has_side_effects=nothing, is_align_stack=nothing, asm_dialect=nothing, operand_attrs=nothing, location=Location())
    results = MLIRType[]
    operands = Value[operands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("asm_string", asm_string), namedattribute("constraints", constraints), ]
    !isnothing(res) && push!(results, res)
    !isnothing(has_side_effects) && push!(attributes, namedattribute("has_side_effects", has_side_effects))
    !isnothing(is_align_stack) && push!(attributes, namedattribute("is_align_stack", is_align_stack))
    !isnothing(asm_dialect) && push!(attributes, namedattribute("asm_dialect", asm_dialect))
    !isnothing(operand_attrs) && push!(attributes, namedattribute("operand_attrs", operand_attrs))
    
    create_operation(
        "llvm.inline_asm", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`insertelement`

"""
function insertelement(vector::Value, value::Value, position::Value; res=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[vector, value, position, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    create_operation(
        "llvm.insertelement", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`insertvalue`

"""
function insertvalue(container::Value, value::Value; res=nothing::Union{Nothing, MLIRType}, position, location=Location())
    results = MLIRType[]
    operands = Value[container, value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("position", position), ]
    !isnothing(res) && push!(results, res)
    
    create_operation(
        "llvm.insertvalue", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`inttoptr`

"""
function inttoptr(arg::Value; res::MLIRType, location=Location())
    results = MLIRType[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.inttoptr", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`invoke`

"""
function invoke(callee_operands::Vector{Value}, normalDestOperands::Vector{Value}, unwindDestOperands::Vector{Value}; result_0::Vector{MLIRType}, callee=nothing, branch_weights=nothing, normalDest::Block, unwindDest::Block, location=Location())
    results = MLIRType[result_0..., ]
    operands = Value[callee_operands..., normalDestOperands..., unwindDestOperands..., ]
    owned_regions = Region[]
    successors = Block[normalDest, unwindDest, ]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(callee_operands), length(normalDestOperands), length(unwindDestOperands), ]))
    !isnothing(callee) && push!(attributes, namedattribute("callee", callee))
    !isnothing(branch_weights) && push!(attributes, namedattribute("branch_weights", branch_weights))
    
    create_operation(
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
function func(; sym_name, function_type, linkage=nothing, dso_local=nothing, CConv=nothing, personality=nothing, garbageCollector=nothing, passthrough=nothing, arg_attrs=nothing, res_attrs=nothing, function_entry_count=nothing, memory=nothing, body::Region, location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[body, ]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sym_name", sym_name), namedattribute("function_type", function_type), ]
    !isnothing(linkage) && push!(attributes, namedattribute("linkage", linkage))
    !isnothing(dso_local) && push!(attributes, namedattribute("dso_local", dso_local))
    !isnothing(CConv) && push!(attributes, namedattribute("CConv", CConv))
    !isnothing(personality) && push!(attributes, namedattribute("personality", personality))
    !isnothing(garbageCollector) && push!(attributes, namedattribute("garbageCollector", garbageCollector))
    !isnothing(passthrough) && push!(attributes, namedattribute("passthrough", passthrough))
    !isnothing(arg_attrs) && push!(attributes, namedattribute("arg_attrs", arg_attrs))
    !isnothing(res_attrs) && push!(attributes, namedattribute("res_attrs", res_attrs))
    !isnothing(function_entry_count) && push!(attributes, namedattribute("function_entry_count", function_entry_count))
    !isnothing(memory) && push!(attributes, namedattribute("memory", memory))
    
    create_operation(
        "llvm.func", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`lshr`

"""
function lshr(lhs::Value, rhs::Value; res=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    create_operation(
        "llvm.lshr", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`landingpad`

"""
function landingpad(operand_0::Vector{Value}; res::MLIRType, cleanup=nothing, location=Location())
    results = MLIRType[res, ]
    operands = Value[operand_0..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(cleanup) && push!(attributes, namedattribute("cleanup", cleanup))
    
    create_operation(
        "llvm.landingpad", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`load`

"""
function load(addr::Value; res::MLIRType, access_groups=nothing, alias_scopes=nothing, noalias_scopes=nothing, alignment=nothing, volatile_=nothing, nontemporal=nothing, location=Location())
    results = MLIRType[res, ]
    operands = Value[addr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(access_groups) && push!(attributes, namedattribute("access_groups", access_groups))
    !isnothing(alias_scopes) && push!(attributes, namedattribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, namedattribute("noalias_scopes", noalias_scopes))
    !isnothing(alignment) && push!(attributes, namedattribute("alignment", alignment))
    !isnothing(volatile_) && push!(attributes, namedattribute("volatile_", volatile_))
    !isnothing(nontemporal) && push!(attributes, namedattribute("nontemporal", nontemporal))
    
    create_operation(
        "llvm.load", location;
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
  }
"""
function metadata(; sym_name, body::Region, location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[body, ]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sym_name", sym_name), ]
    
    create_operation(
        "llvm.metadata", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`mul`

"""
function mul(lhs::Value, rhs::Value; res=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    create_operation(
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
function mlir_null(; res::MLIRType, location=Location())
    results = MLIRType[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.mlir.null", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`or`

"""
function or(lhs::Value, rhs::Value; res=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    create_operation(
        "llvm.or", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`ptrtoint`

"""
function ptrtoint(arg::Value; res::MLIRType, location=Location())
    results = MLIRType[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.ptrtoint", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`resume`

"""
function resume(value::Value; location=Location())
    results = MLIRType[]
    operands = Value[value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.resume", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`return_`

"""
function return_(arg=nothing::Union{Nothing, Value}; location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(arg) && push!(operands, arg)
    
    create_operation(
        "llvm.return", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`sdiv`

"""
function sdiv(lhs::Value, rhs::Value; res=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    create_operation(
        "llvm.sdiv", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`sext`

"""
function sext(arg::Value; res::MLIRType, location=Location())
    results = MLIRType[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.sext", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`sitofp`

"""
function sitofp(arg::Value; res::MLIRType, location=Location())
    results = MLIRType[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.sitofp", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`srem`

"""
function srem(lhs::Value, rhs::Value; res=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    create_operation(
        "llvm.srem", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`select`

"""
function select(condition::Value, trueValue::Value, falseValue::Value; res=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[condition, trueValue, falseValue, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    create_operation(
        "llvm.select", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`shl`

"""
function shl(lhs::Value, rhs::Value; res=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    create_operation(
        "llvm.shl", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`shufflevector`

"""
function shufflevector(v1::Value, v2::Value; res::MLIRType, mask, location=Location())
    results = MLIRType[res, ]
    operands = Value[v1, v2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("mask", mask), ]
    
    create_operation(
        "llvm.shufflevector", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`store`

"""
function store(value::Value, addr::Value; access_groups=nothing, alias_scopes=nothing, noalias_scopes=nothing, alignment=nothing, volatile_=nothing, nontemporal=nothing, location=Location())
    results = MLIRType[]
    operands = Value[value, addr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(access_groups) && push!(attributes, namedattribute("access_groups", access_groups))
    !isnothing(alias_scopes) && push!(attributes, namedattribute("alias_scopes", alias_scopes))
    !isnothing(noalias_scopes) && push!(attributes, namedattribute("noalias_scopes", noalias_scopes))
    !isnothing(alignment) && push!(attributes, namedattribute("alignment", alignment))
    !isnothing(volatile_) && push!(attributes, namedattribute("volatile_", volatile_))
    !isnothing(nontemporal) && push!(attributes, namedattribute("nontemporal", nontemporal))
    
    create_operation(
        "llvm.store", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`sub`

"""
function sub(lhs::Value, rhs::Value; res=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    create_operation(
        "llvm.sub", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`switch`

"""
function switch(value::Value, defaultOperands::Vector{Value}, caseOperands::Vector{Value}; case_values=nothing, case_operand_segments, branch_weights=nothing, defaultDestination::Block, caseDestinations::Vector{Block}, location=Location())
    results = MLIRType[]
    operands = Value[value, defaultOperands..., caseOperands..., ]
    owned_regions = Region[]
    successors = Block[defaultDestination, caseDestinations..., ]
    attributes = NamedAttribute[namedattribute("case_operand_segments", case_operand_segments), ]
    push!(attributes, operandsegmentsizes([1, length(defaultOperands), length(caseOperands), ]))
    !isnothing(case_values) && push!(attributes, namedattribute("case_values", case_values))
    !isnothing(branch_weights) && push!(attributes, namedattribute("branch_weights", branch_weights))
    
    create_operation(
        "llvm.switch", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`tbaa_root`

Defines a TBAA root node.

# Example
  llvm.metadata @tbaa {
    llvm.tbaa_root @tbaa_root_0 {identity = \"Simple C/C++ TBAA\"}
  }

See the following link for more details:
https://llvm.org/docs/LangRef.html#tbaa-metadata
"""
function tbaa_root(; sym_name, identity, location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sym_name", sym_name), namedattribute("identity", identity), ]
    
    create_operation(
        "llvm.tbaa_root", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`tbaa_tag`

Defines a TBAA node describing a memory access.

# Example
  llvm.metadata @tbaa {
    llvm.tbaa_root @tbaa_root_0 {identity = \"Simple C/C++ TBAA\"}
    llvm.tbaa_type_desc @tbaa_type_desc_1 {
        identity = \"omnipotent char\",
        members = [@tbaa_root_0],
        offsets = array<i64: 0>
    }
    llvm.tbaa_type_desc @tbaa_type_desc_2 {
        identity = \"long long\",
        members = [@tbaa_type_desc_1],
        offsets = array<i64: 0>
    }
    llvm.tbaa_type_desc @tbaa_type_desc_3 {
        identity = \"agg2_t\",
        members = [@tbaa_type_desc_2, @tbaa_type_desc_2],
        offsets = array<i64: 0, 8>
    }
    llvm.tbaa_tag @tbaa_tag_4 {
        access_type = @tbaa_type_desc_2,
        base_type = @tbaa_type_desc_3,
        offset = 8 : i64
    }
    llvm.tbaa_type_desc @tbaa_type_desc_5 {
        identity = \"int\",
        members = [@tbaa_type_desc_1],
        offsets = array<i64: 0>
    }
    llvm.tbaa_type_desc @tbaa_type_desc_6 {
        identity = \"agg1_t\",
        members = [@tbaa_type_desc_5, @tbaa_type_desc_5],
        offsets = array<i64: 0, 4>
    }
    llvm.tbaa_tag @tbaa_tag_7 {
        access_type = @tbaa_type_desc_5,
        base_type = @tbaa_type_desc_6,
        offset = 0 : i64
    }
  }

See the following link for more details:
https://llvm.org/docs/LangRef.html#tbaa-metadata
"""
function tbaa_tag(; sym_name, base_type, access_type, offset, constant=nothing, location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sym_name", sym_name), namedattribute("base_type", base_type), namedattribute("access_type", access_type), namedattribute("offset", offset), ]
    !isnothing(constant) && push!(attributes, namedattribute("constant", constant))
    
    create_operation(
        "llvm.tbaa_tag", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`tbaa_type_desc`

Defines a TBAA node describing a type.

# Example
  llvm.metadata @tbaa {
    llvm.tbaa_root @tbaa_root_0 {identity = \"Simple C/C++ TBAA\"}
    llvm.tbaa_type_desc @tbaa_type_desc_1 {
        identity = \"omnipotent char\",
        members = [@tbaa_root_0],
        offsets = array<i64: 0>
    }
    llvm.tbaa_type_desc @tbaa_type_desc_2 {
        identity = \"long long\",
        members = [@tbaa_type_desc_1],
        offsets = array<i64: 0>
    }
    llvm.tbaa_type_desc @tbaa_type_desc_3 {
        identity = \"agg2_t\",
        members = [@tbaa_type_desc_2, @tbaa_type_desc_2],
        offsets = array<i64: 0, 8>
    }
    llvm.tbaa_type_desc @tbaa_type_desc_5 {
        identity = \"int\",
        members = [@tbaa_type_desc_1],
        offsets = array<i64: 0>
    }
    llvm.tbaa_type_desc @tbaa_type_desc_6 {
        identity = \"agg1_t\",
        members = [@tbaa_type_desc_5, @tbaa_type_desc_5],
        offsets = array<i64: 0, 4>
    }
  }

See the following link for more details:
https://llvm.org/docs/LangRef.html#tbaa-metadata
"""
function tbaa_type_desc(; sym_name, identity=nothing, members, offsets, location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sym_name", sym_name), namedattribute("members", members), namedattribute("offsets", offsets), ]
    !isnothing(identity) && push!(attributes, namedattribute("identity", identity))
    
    create_operation(
        "llvm.tbaa_type_desc", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`trunc`

"""
function trunc(arg::Value; res::MLIRType, location=Location())
    results = MLIRType[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.trunc", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`udiv`

"""
function udiv(lhs::Value, rhs::Value; res=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    create_operation(
        "llvm.udiv", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`uitofp`

"""
function uitofp(arg::Value; res::MLIRType, location=Location())
    results = MLIRType[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.uitofp", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`urem`

"""
function urem(lhs::Value, rhs::Value; res=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    create_operation(
        "llvm.urem", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
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
function mlir_undef(; res::MLIRType, location=Location())
    results = MLIRType[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.mlir.undef", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`unreachable`

"""
function unreachable(; location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.unreachable", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`xor`

"""
function xor(lhs::Value, rhs::Value; res=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    create_operation(
        "llvm.xor", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`zext`

"""
function zext(arg::Value; res::MLIRType, location=Location())
    results = MLIRType[res, ]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.zext", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

end # llvm
