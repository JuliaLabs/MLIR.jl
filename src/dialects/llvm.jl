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
    (res != nothing) && push!(results, res)
    
    create_operation(
        "llvm.ashr", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
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
    (res != nothing) && push!(results, res)
    
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
function mlir_addressof(; res::MLIRType, global_name::Union{Attribute, NamedAttribute}, location=Location())
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
`alloca`

"""
function alloca(arraySize::Value; res::MLIRType, alignment=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, elem_type=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, inalloca=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, location=Location())
    results = MLIRType[res, ]
    operands = Value[arraySize, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (alignment != nothing) && push!(attributes, namedattribute("alignment", alignment))
    (elem_type != nothing) && push!(attributes, namedattribute("elem_type", elem_type))
    (inalloca != nothing) && push!(attributes, namedattribute("inalloca", inalloca))
    
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
    (res != nothing) && push!(results, res)
    
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
function cmpxchg(ptr::Value, cmp::Value, val::Value; res=nothing::Union{Nothing, MLIRType}, success_ordering::Union{Attribute, NamedAttribute}, failure_ordering::Union{Attribute, NamedAttribute}, syncscope=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, alignment=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, weak=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, volatile_=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, access_groups=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, alias_scopes=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, noalias_scopes=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, tbaa=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, location=Location())
    results = MLIRType[]
    operands = Value[ptr, cmp, val, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("success_ordering", success_ordering), namedattribute("failure_ordering", failure_ordering), ]
    (res != nothing) && push!(results, res)
    (syncscope != nothing) && push!(attributes, namedattribute("syncscope", syncscope))
    (alignment != nothing) && push!(attributes, namedattribute("alignment", alignment))
    (weak != nothing) && push!(attributes, namedattribute("weak", weak))
    (volatile_ != nothing) && push!(attributes, namedattribute("volatile_", volatile_))
    (access_groups != nothing) && push!(attributes, namedattribute("access_groups", access_groups))
    (alias_scopes != nothing) && push!(attributes, namedattribute("alias_scopes", alias_scopes))
    (noalias_scopes != nothing) && push!(attributes, namedattribute("noalias_scopes", noalias_scopes))
    (tbaa != nothing) && push!(attributes, namedattribute("tbaa", tbaa))
    
    create_operation(
        "llvm.cmpxchg", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`atomicrmw`

"""
function atomicrmw(ptr::Value, val::Value; res=nothing::Union{Nothing, MLIRType}, bin_op::Union{Attribute, NamedAttribute}, ordering::Union{Attribute, NamedAttribute}, syncscope=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, alignment=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, volatile_=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, access_groups=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, alias_scopes=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, noalias_scopes=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, tbaa=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, location=Location())
    results = MLIRType[]
    operands = Value[ptr, val, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("bin_op", bin_op), namedattribute("ordering", ordering), ]
    (res != nothing) && push!(results, res)
    (syncscope != nothing) && push!(attributes, namedattribute("syncscope", syncscope))
    (alignment != nothing) && push!(attributes, namedattribute("alignment", alignment))
    (volatile_ != nothing) && push!(attributes, namedattribute("volatile_", volatile_))
    (access_groups != nothing) && push!(attributes, namedattribute("access_groups", access_groups))
    (alias_scopes != nothing) && push!(attributes, namedattribute("alias_scopes", alias_scopes))
    (noalias_scopes != nothing) && push!(attributes, namedattribute("noalias_scopes", noalias_scopes))
    (tbaa != nothing) && push!(attributes, namedattribute("tbaa", tbaa))
    
    create_operation(
        "llvm.atomicrmw", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
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
function br(destOperands::Vector{Value}; loop_annotation=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, dest::Block, location=Location())
    results = MLIRType[]
    operands = Value[destOperands..., ]
    owned_regions = Region[]
    successors = Block[dest, ]
    attributes = NamedAttribute[]
    (loop_annotation != nothing) && push!(attributes, namedattribute("loop_annotation", loop_annotation))
    
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
function call(operand_0::Vector{Value}; result=nothing::Union{Nothing, MLIRType}, callee=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, fastmathFlags=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, branch_weights=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, access_groups=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, alias_scopes=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, noalias_scopes=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, tbaa=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, location=Location())
    results = MLIRType[]
    operands = Value[operand_0..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    (callee != nothing) && push!(attributes, namedattribute("callee", callee))
    (fastmathFlags != nothing) && push!(attributes, namedattribute("fastmathFlags", fastmathFlags))
    (branch_weights != nothing) && push!(attributes, namedattribute("branch_weights", branch_weights))
    (access_groups != nothing) && push!(attributes, namedattribute("access_groups", access_groups))
    (alias_scopes != nothing) && push!(attributes, namedattribute("alias_scopes", alias_scopes))
    (noalias_scopes != nothing) && push!(attributes, namedattribute("noalias_scopes", noalias_scopes))
    (tbaa != nothing) && push!(attributes, namedattribute("tbaa", tbaa))
    
    create_operation(
        "llvm.call", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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
function comdat(; sym_name::Union{Attribute, NamedAttribute}, body::Region, location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[body, ]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sym_name", sym_name), ]
    
    create_operation(
        "llvm.comdat", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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
function comdat_selector(; sym_name::Union{Attribute, NamedAttribute}, comdat::Union{Attribute, NamedAttribute}, location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sym_name", sym_name), namedattribute("comdat", comdat), ]
    
    create_operation(
        "llvm.comdat_selector", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`cond_br`

"""
function cond_br(condition::Value, trueDestOperands::Vector{Value}, falseDestOperands::Vector{Value}; branch_weights=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, loop_annotation=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, trueDest::Block, falseDest::Block, location=Location())
    results = MLIRType[]
    operands = Value[condition, trueDestOperands..., falseDestOperands..., ]
    owned_regions = Region[]
    successors = Block[trueDest, falseDest, ]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([1, length(trueDestOperands), length(falseDestOperands), ]))
    (branch_weights != nothing) && push!(attributes, namedattribute("branch_weights", branch_weights))
    (loop_annotation != nothing) && push!(attributes, namedattribute("loop_annotation", loop_annotation))
    
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
function mlir_constant(; res::MLIRType, value::Union{Attribute, NamedAttribute}, location=Location())
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
    (res != nothing) && push!(results, res)
    
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
function extractvalue(container::Value; res::MLIRType, position::Union{Attribute, NamedAttribute}, location=Location())
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
function fadd(lhs::Value, rhs::Value; res=nothing::Union{Nothing, MLIRType}, fastmathFlags=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, location=Location())
    results = MLIRType[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (res != nothing) && push!(results, res)
    (fastmathFlags != nothing) && push!(attributes, namedattribute("fastmathFlags", fastmathFlags))
    
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
function fcmp(lhs::Value, rhs::Value; res=nothing::Union{Nothing, MLIRType}, predicate::Union{Attribute, NamedAttribute}, fastmathFlags=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, location=Location())
    results = MLIRType[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("predicate", predicate), ]
    (res != nothing) && push!(results, res)
    (fastmathFlags != nothing) && push!(attributes, namedattribute("fastmathFlags", fastmathFlags))
    
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
function fdiv(lhs::Value, rhs::Value; res=nothing::Union{Nothing, MLIRType}, fastmathFlags=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, location=Location())
    results = MLIRType[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (res != nothing) && push!(results, res)
    (fastmathFlags != nothing) && push!(attributes, namedattribute("fastmathFlags", fastmathFlags))
    
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
function fmul(lhs::Value, rhs::Value; res=nothing::Union{Nothing, MLIRType}, fastmathFlags=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, location=Location())
    results = MLIRType[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (res != nothing) && push!(results, res)
    (fastmathFlags != nothing) && push!(attributes, namedattribute("fastmathFlags", fastmathFlags))
    
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
function fneg(operand::Value; res=nothing::Union{Nothing, MLIRType}, fastmathFlags=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, location=Location())
    results = MLIRType[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (res != nothing) && push!(results, res)
    (fastmathFlags != nothing) && push!(attributes, namedattribute("fastmathFlags", fastmathFlags))
    
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
function frem(lhs::Value, rhs::Value; res=nothing::Union{Nothing, MLIRType}, fastmathFlags=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, location=Location())
    results = MLIRType[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (res != nothing) && push!(results, res)
    (fastmathFlags != nothing) && push!(attributes, namedattribute("fastmathFlags", fastmathFlags))
    
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
function fsub(lhs::Value, rhs::Value; res=nothing::Union{Nothing, MLIRType}, fastmathFlags=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, location=Location())
    results = MLIRType[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (res != nothing) && push!(results, res)
    (fastmathFlags != nothing) && push!(attributes, namedattribute("fastmathFlags", fastmathFlags))
    
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
function fence(; ordering::Union{Attribute, NamedAttribute}, syncscope=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("ordering", ordering), ]
    (syncscope != nothing) && push!(attributes, namedattribute("syncscope", syncscope))
    
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
    (res != nothing) && push!(results, res)
    
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
function getelementptr(base::Value, dynamicIndices::Vector{Value}; res::MLIRType, rawConstantIndices::Union{Attribute, NamedAttribute}, elem_type=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, inbounds=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, location=Location())
    results = MLIRType[res, ]
    operands = Value[base, dynamicIndices..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("rawConstantIndices", rawConstantIndices), ]
    (elem_type != nothing) && push!(attributes, namedattribute("elem_type", elem_type))
    (inbounds != nothing) && push!(attributes, namedattribute("inbounds", inbounds))
    
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
function mlir_global_ctors(; ctors::Union{Attribute, NamedAttribute}, priorities::Union{Attribute, NamedAttribute}, location=Location())
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
function mlir_global_dtors(; dtors::Union{Attribute, NamedAttribute}, priorities::Union{Attribute, NamedAttribute}, location=Location())
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
function mlir_global(; global_type::Union{Attribute, NamedAttribute}, constant=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, sym_name::Union{Attribute, NamedAttribute}, linkage::Union{Attribute, NamedAttribute}, dso_local=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, thread_local_=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, value=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, alignment=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, addr_space=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, unnamed_addr=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, section=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, comdat=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, visibility_=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, initializer::Region, location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[initializer, ]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("global_type", global_type), namedattribute("sym_name", sym_name), namedattribute("linkage", linkage), ]
    (constant != nothing) && push!(attributes, namedattribute("constant", constant))
    (dso_local != nothing) && push!(attributes, namedattribute("dso_local", dso_local))
    (thread_local_ != nothing) && push!(attributes, namedattribute("thread_local_", thread_local_))
    (value != nothing) && push!(attributes, namedattribute("value", value))
    (alignment != nothing) && push!(attributes, namedattribute("alignment", alignment))
    (addr_space != nothing) && push!(attributes, namedattribute("addr_space", addr_space))
    (unnamed_addr != nothing) && push!(attributes, namedattribute("unnamed_addr", unnamed_addr))
    (section != nothing) && push!(attributes, namedattribute("section", section))
    (comdat != nothing) && push!(attributes, namedattribute("comdat", comdat))
    (visibility_ != nothing) && push!(attributes, namedattribute("visibility_", visibility_))
    
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
function icmp(lhs::Value, rhs::Value; res=nothing::Union{Nothing, MLIRType}, predicate::Union{Attribute, NamedAttribute}, location=Location())
    results = MLIRType[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("predicate", predicate), ]
    (res != nothing) && push!(results, res)
    
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
function inline_asm(operands::Vector{Value}; res=nothing::Union{Nothing, MLIRType}, asm_string::Union{Attribute, NamedAttribute}, constraints::Union{Attribute, NamedAttribute}, has_side_effects=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, is_align_stack=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, asm_dialect=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, operand_attrs=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, location=Location())
    results = MLIRType[]
    operands = Value[operands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("asm_string", asm_string), namedattribute("constraints", constraints), ]
    (res != nothing) && push!(results, res)
    (has_side_effects != nothing) && push!(attributes, namedattribute("has_side_effects", has_side_effects))
    (is_align_stack != nothing) && push!(attributes, namedattribute("is_align_stack", is_align_stack))
    (asm_dialect != nothing) && push!(attributes, namedattribute("asm_dialect", asm_dialect))
    (operand_attrs != nothing) && push!(attributes, namedattribute("operand_attrs", operand_attrs))
    
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
    (res != nothing) && push!(results, res)
    
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
function insertvalue(container::Value, value::Value; res=nothing::Union{Nothing, MLIRType}, position::Union{Attribute, NamedAttribute}, location=Location())
    results = MLIRType[]
    operands = Value[container, value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("position", position), ]
    (res != nothing) && push!(results, res)
    
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
function invoke(callee_operands::Vector{Value}, normalDestOperands::Vector{Value}, unwindDestOperands::Vector{Value}; result_0::Vector{MLIRType}, callee=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, branch_weights=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, normalDest::Block, unwindDest::Block, location=Location())
    results = MLIRType[result_0..., ]
    operands = Value[callee_operands..., normalDestOperands..., unwindDestOperands..., ]
    owned_regions = Region[]
    successors = Block[normalDest, unwindDest, ]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(callee_operands), length(normalDestOperands), length(unwindDestOperands), ]))
    (callee != nothing) && push!(attributes, namedattribute("callee", callee))
    (branch_weights != nothing) && push!(attributes, namedattribute("branch_weights", branch_weights))
    
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
function func(; sym_name::Union{Attribute, NamedAttribute}, function_type::Union{Attribute, NamedAttribute}, linkage=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, dso_local=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, CConv=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, comdat=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, personality=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, garbageCollector=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, passthrough=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, arg_attrs=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, res_attrs=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, function_entry_count=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, memory=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, visibility_=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, arm_streaming=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, arm_locally_streaming=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, section=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, unnamed_addr=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, alignment=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, body::Region, location=Location())
    results = MLIRType[]
    operands = Value[]
    owned_regions = Region[body, ]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sym_name", sym_name), namedattribute("function_type", function_type), ]
    (linkage != nothing) && push!(attributes, namedattribute("linkage", linkage))
    (dso_local != nothing) && push!(attributes, namedattribute("dso_local", dso_local))
    (CConv != nothing) && push!(attributes, namedattribute("CConv", CConv))
    (comdat != nothing) && push!(attributes, namedattribute("comdat", comdat))
    (personality != nothing) && push!(attributes, namedattribute("personality", personality))
    (garbageCollector != nothing) && push!(attributes, namedattribute("garbageCollector", garbageCollector))
    (passthrough != nothing) && push!(attributes, namedattribute("passthrough", passthrough))
    (arg_attrs != nothing) && push!(attributes, namedattribute("arg_attrs", arg_attrs))
    (res_attrs != nothing) && push!(attributes, namedattribute("res_attrs", res_attrs))
    (function_entry_count != nothing) && push!(attributes, namedattribute("function_entry_count", function_entry_count))
    (memory != nothing) && push!(attributes, namedattribute("memory", memory))
    (visibility_ != nothing) && push!(attributes, namedattribute("visibility_", visibility_))
    (arm_streaming != nothing) && push!(attributes, namedattribute("arm_streaming", arm_streaming))
    (arm_locally_streaming != nothing) && push!(attributes, namedattribute("arm_locally_streaming", arm_locally_streaming))
    (section != nothing) && push!(attributes, namedattribute("section", section))
    (unnamed_addr != nothing) && push!(attributes, namedattribute("unnamed_addr", unnamed_addr))
    (alignment != nothing) && push!(attributes, namedattribute("alignment", alignment))
    
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
    (res != nothing) && push!(results, res)
    
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
function landingpad(operand_0::Vector{Value}; res::MLIRType, cleanup=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, location=Location())
    results = MLIRType[res, ]
    operands = Value[operand_0..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (cleanup != nothing) && push!(attributes, namedattribute("cleanup", cleanup))
    
    create_operation(
        "llvm.landingpad", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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
function load(addr::Value; res::MLIRType, alignment=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, volatile_=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, nontemporal=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, ordering=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, syncscope=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, access_groups=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, alias_scopes=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, noalias_scopes=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, tbaa=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, location=Location())
    results = MLIRType[res, ]
    operands = Value[addr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (alignment != nothing) && push!(attributes, namedattribute("alignment", alignment))
    (volatile_ != nothing) && push!(attributes, namedattribute("volatile_", volatile_))
    (nontemporal != nothing) && push!(attributes, namedattribute("nontemporal", nontemporal))
    (ordering != nothing) && push!(attributes, namedattribute("ordering", ordering))
    (syncscope != nothing) && push!(attributes, namedattribute("syncscope", syncscope))
    (access_groups != nothing) && push!(attributes, namedattribute("access_groups", access_groups))
    (alias_scopes != nothing) && push!(attributes, namedattribute("alias_scopes", alias_scopes))
    (noalias_scopes != nothing) && push!(attributes, namedattribute("noalias_scopes", noalias_scopes))
    (tbaa != nothing) && push!(attributes, namedattribute("tbaa", tbaa))
    
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
```mlir
llvm.metadata @metadata {
  llvm.access_group @group1
  llvm.access_group @group2
}
```
"""
function metadata(; sym_name::Union{Attribute, NamedAttribute}, body::Region, location=Location())
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
    (res != nothing) && push!(results, res)
    
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
    (res != nothing) && push!(results, res)
    
    create_operation(
        "llvm.or", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
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
function mlir_poison(; res::MLIRType, location=Location())
    results = MLIRType[res, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "llvm.mlir.poison", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
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
    (arg != nothing) && push!(operands, arg)
    
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
    (res != nothing) && push!(results, res)
    
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
    (res != nothing) && push!(results, res)
    
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
function select(condition::Value, trueValue::Value, falseValue::Value; res=nothing::Union{Nothing, MLIRType}, fastmathFlags=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, location=Location())
    results = MLIRType[]
    operands = Value[condition, trueValue, falseValue, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (res != nothing) && push!(results, res)
    (fastmathFlags != nothing) && push!(attributes, namedattribute("fastmathFlags", fastmathFlags))
    
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
    (res != nothing) && push!(results, res)
    
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
function shufflevector(v1::Value, v2::Value; res::MLIRType, mask::Union{Attribute, NamedAttribute}, location=Location())
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
function store(value::Value, addr::Value; alignment=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, volatile_=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, nontemporal=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, ordering=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, syncscope=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, access_groups=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, alias_scopes=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, noalias_scopes=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, tbaa=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, location=Location())
    results = MLIRType[]
    operands = Value[value, addr, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (alignment != nothing) && push!(attributes, namedattribute("alignment", alignment))
    (volatile_ != nothing) && push!(attributes, namedattribute("volatile_", volatile_))
    (nontemporal != nothing) && push!(attributes, namedattribute("nontemporal", nontemporal))
    (ordering != nothing) && push!(attributes, namedattribute("ordering", ordering))
    (syncscope != nothing) && push!(attributes, namedattribute("syncscope", syncscope))
    (access_groups != nothing) && push!(attributes, namedattribute("access_groups", access_groups))
    (alias_scopes != nothing) && push!(attributes, namedattribute("alias_scopes", alias_scopes))
    (noalias_scopes != nothing) && push!(attributes, namedattribute("noalias_scopes", noalias_scopes))
    (tbaa != nothing) && push!(attributes, namedattribute("tbaa", tbaa))
    
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
    (res != nothing) && push!(results, res)
    
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
function switch(value::Value, defaultOperands::Vector{Value}, caseOperands::Vector{Value}; case_values=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, case_operand_segments::Union{Attribute, NamedAttribute}, branch_weights=nothing::Union{Nothing, Union{Attribute, NamedAttribute}}, defaultDestination::Block, caseDestinations::Vector{Block}, location=Location())
    results = MLIRType[]
    operands = Value[value, defaultOperands..., caseOperands..., ]
    owned_regions = Region[]
    successors = Block[defaultDestination, caseDestinations..., ]
    attributes = NamedAttribute[namedattribute("case_operand_segments", case_operand_segments), ]
    push!(attributes, operandsegmentsizes([1, length(defaultOperands), length(caseOperands), ]))
    (case_values != nothing) && push!(attributes, namedattribute("case_values", case_values))
    (branch_weights != nothing) && push!(attributes, namedattribute("branch_weights", branch_weights))
    
    create_operation(
        "llvm.switch", location;
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
    (res != nothing) && push!(results, res)
    
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
    (res != nothing) && push!(results, res)
    
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
IR dialect type.

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
    (res != nothing) && push!(results, res)
    
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
